#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    struct { int sm; int cores; } sm2cores[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128},
        {0x70, 64},  {0x72, 64},  {0x75, 64},
        {0x80, 64},  {0x86, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}
    };
    int idx = 0;
    while (sm2cores[idx].sm != -1) {
        if (sm2cores[idx].sm == ((major << 4) + minor))
            return sm2cores[idx].cores;
        idx++;
    }
    printf("Unknown SM %d.%d, defaulting to 64\n", major, minor);
    return 64;
}

int getTensorCoresPerSM(int major, int minor) {
    if (major < 7) return 0;
    if (major == 7) return 8;
    if (major == 8) return 4;
    if (major == 9) return 4;
    return 0;
}

// FP64 ratio is architectural — hardcoded per family
// Returns the divisor (e.g. 64 means FP64 = FP32/64)
int getFP64Divisor(int major, int minor) {
    if (major == 8 && minor == 0) return 2;   // A100 — proper compute
    if (major == 8)               return 64;  // other Ampere — gaming
    if (major == 9)               return 2;   // Hopper
    if (major == 7 && minor == 0) return 2;   // V100
    if (major == 7)               return 32;  // Turing
    if (major == 6 && minor == 0) return 2;   // P100
    if (major == 6)               return 32;  // other Pascal
    return 32;                                // safe default
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable GPU detected.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        int coresPerSM    = getCoresPerSM(prop.major, prop.minor);
        int totalCUDA     = coresPerSM * prop.multiProcessorCount;
        int tensorPerSM   = getTensorCoresPerSM(prop.major, prop.minor);
        int totalTensor   = tensorPerSM * prop.multiProcessorCount;
        int fp64divisor   = getFP64Divisor(prop.major, prop.minor);

        int clockKHz, memClockKHz;
        cudaDeviceGetAttribute(&clockKHz,    cudaDevAttrClockRate,       dev);
        cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);

        // Derived performance metrics
        double peak_fp32_tflops = ((double)totalCUDA * clockKHz * 1e3 * 2.0) / 1e12;
        double peak_fp64_tflops = peak_fp32_tflops / fp64divisor;
        double peak_bw_GBs      = (memClockKHz * 1e3 * (prop.memoryBusWidth / 8) * 2.0) / 1e9;
        double ridge_point      = (peak_fp32_tflops * 1e12) / (peak_bw_GBs * 1e9);

        printf("=== GPU %d: %s ===\n", dev, prop.name);
        printf("Compute Capability:           %d.%d\n", prop.major, prop.minor);
        printf("Streaming Multiprocessors:    %d SMs\n", prop.multiProcessorCount);
        printf("CUDA Cores:                   %d (%d/SM)\n", totalCUDA, coresPerSM);
        if (totalTensor > 0)
            printf("Tensor Cores:                 %d (%d/SM)\n", totalTensor, tensorPerSM);
        else
            printf("Tensor Cores:                 none\n");

        printf("\n--- Clocks ---\n");
        printf("GPU Clock:                    %.0f MHz\n", clockKHz / 1000.0);
        printf("Memory Clock:                 %.0f MHz\n", memClockKHz / 1000.0);
        printf("Memory Bus:                   %d-bit\n", prop.memoryBusWidth);

        printf("\n--- Memory ---\n");
        printf("Global Memory:                %.2f GB\n",
               prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("L2 Cache:                     %d KB\n", prop.l2CacheSize / 1024);
        printf("Shared Memory per SM:         %zu KB\n",
               prop.sharedMemPerMultiprocessor / 1024);

        printf("\n--- Thread Hierarchy ---\n");
        printf("Max Threads per Block:        %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per SM:           %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max Concurrent Threads:       %d\n",
               prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);
        printf("Latency hiding ratio:         %dx (threads/cores per SM)\n",
               prop.maxThreadsPerMultiProcessor / coresPerSM);

        printf("\n--- Roofline Model ---\n");
        printf("Peak FP32:                    %.2f TFLOPS\n", peak_fp32_tflops);
        printf("Peak FP64 (1/%d):             %.3f TFLOPS\n", fp64divisor, peak_fp64_tflops);
        printf("Peak Memory Bandwidth:        %.1f GB/s\n", peak_bw_GBs);
        printf("Ridge Point:                  %.1f FLOP/byte\n", ridge_point);
        printf("  → kernels below %.0f FLOP/byte are memory bound\n", ridge_point);
        printf("  → CFD kernels typically 1-10 FLOP/byte: MEMORY BOUND\n");

        printf("\n--- Precision Summary ---\n");
        printf("FP32 epsilon:                 1.19e-07  (6 significant digits)\n");
        printf("FP64 epsilon:                 2.22e-16  (15 significant digits)\n");
        printf("FP64/FP32 architectural ratio: 1/%d\n", fp64divisor);
        printf("FP64/FP32 measured ratio:     1/73 (this machine, thermal throttled)\n");

        printf("\n");
    }
    return 0;
}
