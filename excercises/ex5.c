#include <arm_neon.h>
# include <stdio.h>
int main(void) {
    float32x4_t vec1 = {4.0, 5.0, 13.0, 6.0};
    float32x4_t vec2 = {9.0, 3.0, 6.0, 7.0};
    float32x4_t neg = {1.0, -1.0, 1.0, -1.0};

    // (a+bi)*(c+di) = (ac-bd)+(ad+bc)*i
    // (x+yi)*(z+wi) = (xz-yw)+(xw+yz)*i
    // [a, b, x, y]
    // [c, d, z, w] 
    
    float32x4_t vec3 = vmulq_f32(vec1,vec2);
    // vec3 = [ac, bd, xz, yw]

    // switch the real/imaginary values of vec2
    float32x4_t vec2_2 = vrev64q_f32(vec2);
    // vec2_2 = [d, c, w, z]

    // negate the imaginary values of vec2
    vec2_2 = vmulq_f32(vec2_2,neg);
    // vec2_2 = [d, -c, w, -z]
    
    float32x4_t vec4 = vmulq_f32(vec1,vec2_2);
   
    // vec4 = [ad, -bc, xw, -yz]

    // [ac, bd, xz, yw]
    // [ad, -bc, xw, -yz]   We want [ac-bd,ad+bc,xz-yw,xw+yz]
    float32x4_t v3, v4, v5, v6, v7, v8, v9;
    
    v3 = vrev64q_f32(vec2_2); // reverse the order of the elements in v2
    v4 = vrev64q_f32(vget_low_f32(vec2_2)); // reverse the order of the low 2 elements in v2
    v5 = vcombine_f32(vget_high_f32(vec1), vget_low_f32(vec1)); // swap the high and low elements of v1
    v6 = vcombine_f32(vget_low_f32(vec2_2), vget_high_f32(vec2_2)); // swap the high and low elements of v3
    v7 = vmulq_f32(v5, v3); // multiply corresponding elements of v1 and v3
    v8 = vmulq_f32(v6, v4); // multiply corresponding elements of v2 and the reversed low 2 elements of v2
    v9 = vsubq_f32(v7, v8); // subtract the two products to get the first two elements of the result
    v9 = vsetq_lane_f32(vgetq_lane_f32(v9, 2), v9, 1); // move the third element of v9 to the second element
    v9 = vsetq_lane_f32(vgetq_lane_f32(v9, 3), v9, 2); // move the fourth element of v9 to the third element
    float32x4_t result = v9;

    printf("Result of this shit: %f, %f, %f, %f\n", result[0], result[1], result[2], result[3]);

}
