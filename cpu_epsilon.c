#include <stdio.h>

int main() {
    // FP32
    float eps32 = 1.0f;
    while ((1.0f + eps32 / 2.0f) != 1.0f)
        eps32 /= 2.0f;
    
    // FP64
    double eps64 = 1.0;
    while ((1.0 + eps64 / 2.0) != 1.0)
        eps64 /= 2.0;

    // x86 also has 80-bit extended precision (long double)
    long double eps80 = 1.0L;
    while ((1.0L + eps80 / 2.0L) != 1.0L)
        eps80 /= 2.0L;

    printf("CPU Machine Epsilon\n");
    printf("===================\n");
    printf("FP32  (float):       %.10e  (%d significant decimals)\n", eps32,  6);
    printf("FP64  (double):      %.20e  (%d significant decimals)\n", eps64, 15);
    printf("FP80  (long double): %.20Le  (%d significant decimals)\n", eps80, 18);

    return 0;
}
