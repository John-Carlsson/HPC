#ifdef __CUDACC__  // NVCC
extern "C"
{
#endif
#ifdef __cplusplus  // cpp
extern "C"
{
#endif
void vector_add_cuda(float *out, float *a, float *b, int n);

void matrix_multiplication_cuda(float *out, float *a, float *b, int n);

void matrix_multiplication_add_cuda(float *out, float *a, float *b, float *c, int n);

#ifdef __cplusplus // cpp
}
#endif
#ifdef __CUDACC__ // NVCC
}
#endif

#ifdef __cplusplus

#include "math_opts.hpp"

class MMAOptCUDA : public MMAOperation
{
protected:
    unsigned int __matrixSize;

    float *_A;
    float *_B;
    float *_C;
    float *_Out;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_out = nullptr;

    const char* name = "MMA CUDA Shared";

public:
    MMAOptCUDA(float *A, float *B, float *C, float *Out, unsigned int size);
    
    ~MMAOptCUDA()
    {
        Cleanup();
    };

    virtual const char *GetOPTMame();

    void Import();

    virtual void Compute();

    void Export();

    virtual void ComputeNTime(unsigned int loopCount);

    void Cleanup();
};

class MMAOptCUDAGlobMem : public MMAOptCUDA
{
private:
    const char* name = "MMA CUDA";

public:
    MMAOptCUDAGlobMem(float *A, float *B, float *C, float *Out, unsigned int size):
        MMAOptCUDA(A,B,C,Out,size)
    {}

    const char *GetOPTMame();

    void Compute() override;

    void ComputeNTime(unsigned int loopCount) override;

};

#endif