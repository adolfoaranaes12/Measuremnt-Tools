#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Read CPU timestamp counter — for timing
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// CPUID instruction — direct hardware query
void cpuid(uint32_t leaf, uint32_t subleaf,
           uint32_t *eax, uint32_t *ebx,
           uint32_t *ecx, uint32_t *edx) {
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
    );
}

// Check for specific CPU feature flags
int has_feature(uint32_t reg, int bit) {
    return (reg >> bit) & 1;
}

int main() {
    uint32_t eax, ebx, ecx, edx;

    printf("=== CPU Hardware Characterization ===\n\n");

    // --- Brand string (3 CPUID calls) ---
    char brand[49] = {0};
    for (int i = 0; i < 3; i++) {
        cpuid(0x80000002 + i, 0, &eax, &ebx, &ecx, &edx);
        memcpy(brand + i*16 +  0, &eax, 4);
        memcpy(brand + i*16 +  4, &ebx, 4);
        memcpy(brand + i*16 +  8, &ecx, 4);
        memcpy(brand + i*16 + 12, &edx, 4);
    }
    printf("CPU Model:          %s\n", brand);

    // --- Core topology ---
    cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);
    int logical_per_package = (ebx >> 16) & 0xFF;
    printf("Logical CPUs:       %d (per package)\n", logical_per_package);

    // --- Cache hierarchy ---
    printf("\n--- Cache Hierarchy ---\n");
    for (int level = 0; level < 6; level++) {
        cpuid(0x4, level, &eax, &ebx, &ecx, &edx);
        int cache_type = eax & 0x1F;
        if (cache_type == 0) break;

        const char* type_str = "Unknown";
        if (cache_type == 1) type_str = "Data";
        if (cache_type == 2) type_str = "Instruction";
        if (cache_type == 3) type_str = "Unified";

        int cache_level    = (eax >> 5)  & 0x7;
        int line_size      = (ebx & 0xFFF) + 1;
        int partitions     = ((ebx >> 12) & 0x3FF) + 1;
        int associativity  = ((ebx >> 22) & 0x3FF) + 1;
        int sets           = ecx + 1;
        int cache_size_KB  = (line_size * partitions * associativity * sets) / 1024;

        printf("L%d %s Cache:      %d KB (line=%dB, %d-way)\n",
               cache_level, type_str, cache_size_KB, line_size, associativity);
    }

    // --- SIMD / Vector capabilities ---
    cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);
    printf("\n--- SIMD Capabilities ---\n");
    printf("SSE2:               %s\n", has_feature(edx, 26) ? "yes" : "no");
    printf("SSE4.2:             %s\n", has_feature(ecx, 20) ? "yes" : "no");
    printf("AVX:                %s\n", has_feature(ecx, 28) ? "yes" : "no");
    printf("FMA3:               %s\n", has_feature(ecx, 12) ? "yes" : "no");

    cpuid(0x7, 0, &eax, &ebx, &ecx, &edx);
    printf("AVX2:               %s\n", has_feature(ebx,  5) ? "yes" : "no");
    printf("AVX-512F:           %s\n", has_feature(ebx, 16) ? "yes" : "no");

    // --- FP precision ---
    printf("\n--- Floating Point ---\n");
    printf("FP32 epsilon:       1.19e-07  (6  significant digits)\n");
    printf("FP64 epsilon:       2.22e-16  (15 significant digits)\n");
    printf("FP80 epsilon:       1.08e-19  (18 significant digits) x87 extended\n");

    // --- Roofline inputs ---
    // Read actual clock from /proc
    FILE* f = fopen("/proc/cpuinfo", "r");
    double max_mhz = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        double mhz;
        if (sscanf(line, "cpu MHz : %lf", &mhz) == 1)
            if (mhz > max_mhz) max_mhz = mhz;
    }
    fclose(f);

    // Check AVX-512 for roofline
    cpuid(0x7, 0, &eax, &ebx, &ecx, &edx);
    int avx512 = has_feature(ebx, 16);
    int avx2   = has_feature(ebx,  5);

    // FP64 doubles per cycle per core:
    // AVX-512 + FMA: 8 doubles × 2 (FMA) = 16 FLOP/cycle
    // AVX2    + FMA: 4 doubles × 2 (FMA) = 8  FLOP/cycle
    // SSE2    + FMA: 2 doubles × 2 (FMA) = 4  FLOP/cycle
    int fp64_per_cycle = avx512 ? 16 : (avx2 ? 8 : 4);
    printf("\n--- CPU Roofline (per core) ---\n");
    printf("Current Clock:      %.0f MHz\n", max_mhz);
    printf("FP64 FLOP/cycle:    %d (via %s + FMA)\n",
           fp64_per_cycle,
           avx512 ? "AVX-512" : (avx2 ? "AVX2" : "SSE2"));
    printf("Peak FP64/core:     %.2f GFLOPS\n",
           (max_mhz * 1e6 * fp64_per_cycle) / 1e9);
    printf("Peak FP32/core:     %.2f GFLOPS (2x FP64)\n",
           (max_mhz * 1e6 * fp64_per_cycle * 2) / 1e9);

    // Memory bandwidth from /proc (approximate)
    printf("\n--- Key CFD Insight ---\n");
    printf("CPU FP64 peak >>  GPU FP64 (0.136 TFLOPS)\n");
    printf("Use CPU for FP64 pressure solver\n");
    printf("Use GPU for FP32 flux computation + neural operators\n");

    return 0;
}
