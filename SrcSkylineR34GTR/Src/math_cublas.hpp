
#ifdef __cplusplus
#include "math_opts.hpp"
class MMAOptCublas : public MMAOperation
{
private:
    unsigned int __matrixSize;

    float *_A;
    float *_B;
    float *_C;
    float *_Out;

    //half types are cuda
    void *d_a = nullptr, *d_b = nullptr;
    void* cublasHandlerPtr;
    float *d_c = nullptr, *d_out = nullptr;

    const char* name = "MMA Cublas";

public:
    MMAOptCublas(float *A, float *B, float *C, float *Out, unsigned int size);
    
    ~MMAOptCublas()
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