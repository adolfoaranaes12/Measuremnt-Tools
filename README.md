# GPU/CPU Compute Stack Characterization

Empirical benchmarks for understanding floating point precision
and performance across CPU and GPU hardware.

## Tools
- `epsilon.cu` — GPU machine epsilon (FP32/FP64)
- `cpu_epsilon.c` — CPU machine epsilon (FP32/FP64/FP80)
- `fp64_penalty.cu` — Measured FP64/FP32 performance ratio
- `gpu_info.cu` — GPU roofline model + hardware profile
- `cpu_info.c` — CPU roofline model + SIMD capabilities

## Requirements
- CUDA Toolkit 13.0+
- Driver 580+
- gcc with AVX2 support

## Results (RTX 3050 Ti + i7-12700H)
| Metric              | Value         |
|---------------------|---------------|
| GPU FP32 epsilon    | 1.19e-07      |
| GPU FP64 epsilon    | 2.22e-16      |
| CPU FP80 epsilon    | 1.08e-19      |
| FP64/FP32 ratio     | 1/73 measured |
| FP32 drift (10k)    | 4.54e-06      |
| CPU bandwidth       | 44.6 GB/s     |
| GPU bandwidth       | ~192 GB/s     |
