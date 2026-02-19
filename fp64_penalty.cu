#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) {                                          \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error at %s:%d — %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                    \
    }                                                               \
}

__global__ void fp32_kernel(float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = (float)(idx + 1) * 0.0001f;
        for (volatile int i = 0; i < 10000; i++)
            x = x * 1.0000001f + 0.0000001f;
        out[idx] = x;
    }
}

__global__ void fp64_kernel(double* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double x = (double)(idx + 1) * 0.0001;
        for (volatile int i = 0; i < 10000; i++)
            x = x * 1.0000001 + 0.0000001;
        out[idx] = x;
    }
}

int main() {
    int N = 1 << 22;
    float*  d_f32; CUDA_CHECK(cudaMalloc(&d_f32, N * sizeof(float)));
    double* d_f64; CUDA_CHECK(cudaMalloc(&d_f64, N * sizeof(double)));
    float*  h_f32 = new float[N];
    double* h_f64 = new double[N];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Warmup
    fp32_kernel<<<blocks, threads>>>(d_f32, N);
    fp64_kernel<<<blocks, threads>>>(d_f64, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time FP32
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        fp32_kernel<<<blocks, threads>>>(d_f32, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms32;
    cudaEventElapsedTime(&ms32, start, stop);

    // Time FP64
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        fp64_kernel<<<blocks, threads>>>(d_f64, N);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms64;
    cudaEventElapsedTime(&ms64, start, stop);

    CUDA_CHECK(cudaMemcpy(h_f32, d_f32, N * sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f64, d_f64, N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("FP32 results: [0]=%.6f  [100]=%.6f\n",    h_f32[0], h_f32[100]);
    printf("FP64 results: [0]=%.15f [100]=%.15f\n",   h_f64[0], h_f64[100]);
    printf("\nDrift at index 100: %.2e\n",
           (double)h_f32[100] - h_f64[100]);  // this IS the accumulated error

    printf("\nFP32 time (20 runs): %.1f ms\n", ms32);
    printf("FP64 time (20 runs): %.1f ms\n", ms64);
    printf("Measured FP64/FP32 ratio: 1/%.1f\n", ms64 / ms32);

    cudaFree(d_f32);
    cudaFree(d_f64);
    delete[] h_f32;
    delete[] h_f64;
    return 0;
}
