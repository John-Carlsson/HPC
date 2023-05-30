// Multiplication of two array of complex numbers (a+bi)(c+di)in SoA layout
// (ac-bd)+(ad+bc)i
#include <arm_neon.h>
# include <stdio.h>

int main(void) {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {1,2,3,4,5,6,7,8};
    float c[8] = {1,2,3,4,5,6,7,8};
    float d[8] = {1,2,3,4,5,6,7,8};

    float32x4_t v1, v2, v3, v4;
    v1 = vld1q_f32(a);
    v2 = vld1q_f32(b);
    v3 = vld1q_f32(c);
    v4 = vld1q_f32(d);

    float32x4_t ac = vmulq_f32(v1,v3);
    float32x4_t bd = vmulq_f32(v2,v4);
    float32x4_t ad = vmulq_f32(v1,v4);
    float32x4_t bc = vmulq_f32(v2,v3);

    float32x4_t v5 = vsubq_f32(ac,bd);
    float32x4_t v6 = vaddq_f32(ad,bc);
    float32x4_t v7 = vaddq_f32(v5,v6);

    double* res = (double*)&v7;
    for (size_t i = 0; i < 4; i++)
    {
        fprintf(stdout, "%f\n", res[i]);

    }
    
    




}