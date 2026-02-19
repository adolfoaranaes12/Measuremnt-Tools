#include <cfloat>
#include <stdio.h>

__global__ void check_epsilon() {
    printf("FP32 epsilon: %e\n", FLT_EPSILON);
    printf("FP64 epsilon: %e\n", DBL_EPSILON);
    printf("FP32 significant decimals: %d\n", FLT_DIG);
    printf("FP64 significant decimals: %d\n", DBL_DIG);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total memory: %.1f GB\n", prop.totalGlobalMem / 1e9);

    // FP64 capability is inferred from compute capability
    // cc < 6.0 = very limited FP64
    // cc 6.0+ (P100) = good FP64
    // gaming GPUs (cc 8.6, 8.9) = 1/32 FP64 rate despite high cc
    printf("\nNote: check GPU model against NVIDIA specs for FP64/FP32 ratio\n");
    printf("Gaming GPUs typically 1/32, A100 = 1/2, H100 = 1/2\n");

    check_epsilon<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
