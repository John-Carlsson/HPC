void vector_add_cpu(float *out, float *a, float *b, int n);

void matrix_multiplication_cpu(float *out, float *a, float *b, int n);

void matrix_multiplication_add_cpu(float *out, float *a, float *b, float *c, int n);

#ifdef __cplusplus

#include "math_opts.hpp"

#define CPU_VECTOR_BYTE_SIZE 64
#define __ALIGNED __attribute__((aligned(CPU_VECTOR_BYTE_SIZE)))
class MMAOptCPU : public MMAOperation
{
private:
    unsigned int _matrixSize;

    float *_A;
    float *_B;
    float *_C;
    float *_Out;

    float *_AlignedA __ALIGNED = nullptr;
    float *_AlignedB __ALIGNED = nullptr;
    float *_AlignedC __ALIGNED = nullptr;
    float *_AlignedOut __ALIGNED = nullptr;

    const char *name = "MMA CPU";

public:
    MMAOptCPU(float *A, float *B, float *C, float *Out, unsigned int size);

    ~MMAOptCPU()
    {
        Cleanup();
    };

    const char *GetOPTMame();

    void Import();

    void Compute();

    void Export();

    void ComputeNTime(unsigned int loopCount);

    void Cleanup();
};

#endif