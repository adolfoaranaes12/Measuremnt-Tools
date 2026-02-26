#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <time.h>

// ---------------------------------------------------------------------------
// cpu_cores.c  —  P-core / E-core topology probe for MPI layout decisions
//
// Reads Linux sysfs topology (no root required).
// Outputs: core map, cache sharing, and optimal mpirun affinity for OpenFOAM.
// Build:  gcc -O2 -o cpu_cores cpu_cores.c
// ---------------------------------------------------------------------------

#define MAX_CPUS 256

typedef struct {
    int logical_id;
    int physical_core;
    int max_freq_khz;
    int l3_id;           // LLC (last-level cache) id
    int is_pcre;         // 1 = P-core, 0 = E-core
    int sibling;         // HT sibling logical id (-1 if none)
} CpuInfo;

// Read a single integer from a sysfs file; return -1 on failure
static int read_sysfs_int(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val = -1;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

// Read a comma/range list (e.g. "0,2-5") and return first two values
// Returns count of siblings found
static int read_thread_siblings(const char *path, int *out, int max_out) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[256];
    fgets(buf, sizeof(buf), f);
    fclose(f);

    int count = 0;
    char *tok = strtok(buf, ",\n");
    while (tok && count < max_out) {
        // handle ranges like "4-5"
        int a, b;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) {
            for (int i = a; i <= b && count < max_out; i++)
                out[count++] = i;
        } else if (sscanf(tok, "%d", &a) == 1) {
            out[count++] = a;
        }
        tok = strtok(NULL, ",\n");
    }
    return count;
}

// Measure memory bandwidth via repeated sequential reads (~64 MB array)
static double measure_bandwidth_gbs(void) {
    const size_t N = 8 * 1024 * 1024;  // 8M doubles = 64 MB
    double *arr = (double *)malloc(N * sizeof(double));
    if (!arr) return -1.0;

    // warm up
    for (size_t i = 0; i < N; i++) arr[i] = (double)i;

    struct timespec t0, t1;
    volatile double sink = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    // 4 passes to average out noise
    for (int pass = 0; pass < 4; pass++)
        for (size_t i = 0; i < N; i++) sink += arr[i];
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double bytes = 4.0 * N * sizeof(double);
    free(arr);
    (void)sink;
    return bytes / elapsed / 1e9;
}

int main(void) {
    CpuInfo cpus[MAX_CPUS];
    int ncpus = 0;

    // Enumerate /sys/devices/system/cpu/cpu<N>
    char path[512];
    for (int id = 0; id < MAX_CPUS; id++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", id);
        int core = read_sysfs_int(path);
        if (core < 0) continue;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", id);
        int freq = read_sysfs_int(path);

        // L3 / LLC id
        int l3 = -1;
        for (int lev = 0; lev < 6; lev++) {
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/cache/index%d/level", id, lev);
            int level = read_sysfs_int(path);
            if (level == 3) {
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/cache/index%d/id", id, lev);
                l3 = read_sysfs_int(path);
                break;
            }
        }

        cpus[ncpus].logical_id    = id;
        cpus[ncpus].physical_core = core;
        cpus[ncpus].max_freq_khz  = freq;
        cpus[ncpus].l3_id         = l3;
        cpus[ncpus].is_pcre       = (freq >= 4000000) ? 1 : 0;
        cpus[ncpus].sibling       = -1;
        ncpus++;
    }

    // Resolve HT siblings (same physical_core, different logical_id)
    for (int i = 0; i < ncpus; i++) {
        for (int j = i + 1; j < ncpus; j++) {
            if (cpus[i].physical_core == cpus[j].physical_core &&
                cpus[i].is_pcre && cpus[j].is_pcre) {
                cpus[i].sibling = cpus[j].logical_id;
                cpus[j].sibling = cpus[i].logical_id;
            }
        }
    }

    // Count P/E cores
    int n_pcores = 0, n_ecores = 0;
    int p_logical = 0, e_logical = 0;
    for (int i = 0; i < ncpus; i++) {
        if (cpus[i].is_pcre) {
            p_logical++;
            // count unique physical cores
            int dup = 0;
            for (int j = 0; j < i; j++)
                if (cpus[j].physical_core == cpus[i].physical_core && cpus[j].is_pcre)
                    dup = 1;
            if (!dup) n_pcores++;
        } else {
            e_logical++;
            n_ecores++;
        }
    }

    printf("=== CPU Core Topology for MPI Layout ===\n\n");
    printf("Total logical CPUs:  %d\n", ncpus);
    printf("P-cores (HT):        %d physical  /  %d logical  @ %.1f MHz\n",
           n_pcores, p_logical,
           cpus[0].max_freq_khz / 1000.0);
    printf("E-cores (no HT):     %d physical  /  %d logical  @ %.1f MHz\n",
           n_ecores, e_logical,
           (e_logical > 0) ? cpus[p_logical].max_freq_khz / 1000.0 : 0.0);

    printf("\n--- P-core map (logical CPU → physical core, HT sibling) ---\n");
    for (int i = 0; i < ncpus; i++) {
        if (!cpus[i].is_pcre) continue;
        printf("  cpu%-3d  phys_core=%-3d  sibling=cpu%-3d  freq=%d MHz\n",
               cpus[i].logical_id,
               cpus[i].physical_core,
               cpus[i].sibling,
               cpus[i].max_freq_khz / 1000);
    }

    printf("\n--- E-core map (logical CPU → physical core) ---\n");
    for (int i = 0; i < ncpus; i++) {
        if (cpus[i].is_pcre) continue;
        printf("  cpu%-3d  phys_core=%-3d  freq=%d MHz\n",
               cpus[i].logical_id,
               cpus[i].physical_core,
               cpus[i].max_freq_khz / 1000);
    }

    // Build cpu-set strings for mpirun --cpu-set
    // Option A: one rank per P-core physical core (no HT sharing of FPU)
    char cpuset_one_per_pcore[256] = "";
    // Option B: all P-core HT threads (both siblings)
    char cpuset_all_p[256] = "";

    int first_a = 1, first_b = 1;
    for (int i = 0; i < ncpus; i++) {
        if (!cpus[i].is_pcre) continue;
        // all P logical
        if (!first_b) strcat(cpuset_all_p, ",");
        char tmp[8]; snprintf(tmp, sizeof(tmp), "%d", cpus[i].logical_id);
        strcat(cpuset_all_p, tmp);
        first_b = 0;
        // one per physical: take the lower logical id of each pair
        if (cpus[i].sibling < 0 || cpus[i].logical_id < cpus[i].sibling) {
            if (!first_a) strcat(cpuset_one_per_pcore, ",");
            strcat(cpuset_one_per_pcore, tmp);
            first_a = 0;
        }
    }

    printf("\n--- Memory Bandwidth (single-threaded sequential read, ~64 MB) ---\n");
    double bw = measure_bandwidth_gbs();
    if (bw > 0)
        printf("  Measured:  %.1f GB/s\n", bw);
    else
        printf("  Measurement failed (malloc error)\n");

    printf("\n=== OpenFOAM MPI Recommendations ===\n\n");
    printf("Strategy A — FP64 compute-bound (no HT, 1 rank/P-core):\n");
    printf("  Ranks:       %d\n", n_pcores);
    printf("  mpirun -np %d --bind-to core --map-by core \\\n", n_pcores);
    printf("         --cpu-set %s simpleFoam -parallel\n\n", cpuset_one_per_pcore);

    printf("Strategy B — Memory-bandwidth-bound (all P-core HT threads):\n");
    printf("  Ranks:       %d\n", p_logical);
    printf("  mpirun -np %d --bind-to hwthread --map-by hwthread \\\n", p_logical);
    printf("         --cpu-set %s simpleFoam -parallel\n\n", cpuset_all_p);

    printf("Recommendation for simpleFoam (FP64, memory-bound pressure solve):\n");
    printf("  Use Strategy B (%d ranks, P-cores only).\n", p_logical);
    printf("  E-cores excluded: %.1f MHz vs %.1f MHz P-cores would bottleneck MPI sync.\n",
           (e_logical > 0) ? cpus[p_logical].max_freq_khz / 1000.0 : 0.0,
           cpus[0].max_freq_khz / 1000.0);

    printf("\ndecomposeParDict:  numberOfSubdomains %d;\n", p_logical);
    printf("decomposeParDict:  method scotch;\n");

    return 0;
}
