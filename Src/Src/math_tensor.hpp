#ifdef __CUDACC__ // NVCC
extern "C"
{
#endif
#ifdef __cplusplus  // cpp
extern "C"
{
#endif


void vector_add_tensor(float *out, float *a, float *b, int n);

void matrix_multiplication_tensor(float *out, float *a, float *b, int n);

void matrix_multiplication_add_tensor(float *out, float *a, float *b, float *c, int n);

#ifdef __cplusplus // cpp
}
#endif
#ifdef __CUDACC__ // NVCC
}
#endif

#ifdef __cplusplus

#include "math_opts.hpp"

class MMAOptTensor : public MMAOperation
{
protected:
    unsigned int __matrixSize;

    float *_A;
    float *_B;
    float *_C;
    float *_Out;

    //half types are cuda
    void *d_a = nullptr, *d_b = nullptr;
    float *d_c = nullptr, *d_out = nullptr;

    const char* name = "MMA Tensor";

public:
    MMAOptTensor(float *A, float *B, float *C, float *Out, unsigned int size);
    
    ~MMAOptTensor()
    {
        Cleanup();
    };

    const char *GetOPTMame();

    double Import();

    double Compute();

    double Export();

    void ComputeNTime(unsigned int loopCount);

    void Cleanup();
};

class MMAOptTensorShared : public MMAOperation
{
private:
    unsigned int __matrixSize;

    float *_A;
    float *_B;
    float *_C;
    float *_Out;

    //half types are cuda
    void *d_a = nullptr, *d_b = nullptr;
    float *d_c = nullptr, *d_out = nullptr;

    const char* name = "MMA Tensor Shared";

public:
    MMAOptTensorShared(float *A, float *B, float *C, float *Out, unsigned int size);
    
    ~MMAOptTensorShared()
    {
        Cleanup();
    };

    const char *GetOPTMame();

    double Import();

    double Compute();

    double Export();

    void ComputeNTime(unsigned int loopCount);

    void Cleanup();
};


#endif