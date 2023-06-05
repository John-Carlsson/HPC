#include "math_cublas.hpp"
#include "benchmarking.hpp"
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
using namespace nvcuda;

MMAOptCublas::MMAOptCublas(float *A, float *B, float *C, float *Out, unsigned int size) :
        __matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasHandlerPtr = &cublasHandle;
}

const char *MMAOptCublas::GetOPTMame()
{
    return name;
}

double MMAOptCublas::Import()
{
    double time;
    BENCH_STORE(
    half *t_a = (half *)malloc(sizeof(half) * __matrixSize * __matrixSize);
    for (size_t i = 0; i < __matrixSize * __matrixSize; i++)
    {
        t_a[i] = (half)this->_A[i];
    }
    half *t_b = (half *)malloc(sizeof(half) * __matrixSize * __matrixSize);
    for (size_t i = 0; i < __matrixSize * __matrixSize; i++)
    {
        t_b[i] = (half)this->_B[i];
    }

    if (d_a == nullptr) cudaMalloc((void **)&(this->d_a), sizeof(half) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_a, t_a, __matrixSize*__matrixSize*sizeof(half), cudaMemcpyHostToDevice);

    if (d_b == nullptr) cudaMalloc((void **)(&this->d_b), sizeof(half) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_b, t_b, __matrixSize*__matrixSize*sizeof(half), cudaMemcpyHostToDevice);

    if (d_c == nullptr) cudaMalloc((void **)&(this->d_c), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_c, this->_C, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    free(t_a);
    free(t_b);
    ,time)
    return time;
}

double MMAOptCublas::Compute()
{
    float time;

    cublasHandle_t cublasHandle = *(cublasHandle_t*)cublasHandlerPtr;


    float alpha = 1, beta = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                __matrixSize, __matrixSize, __matrixSize,
                &alpha,
                d_a, CUDA_R_16F, __matrixSize,
                d_b, CUDA_R_16F, __matrixSize,
                &beta,
                d_c, CUDA_R_32F, __matrixSize,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

double MMAOptCublas::Export()
{
    double time;
    BENCH_STORE(
    cudaMemcpy(this->_Out, this->d_c, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
    ,time)
    return time;
}

void MMAOptCublas::ComputeNTime(unsigned int loopCount)
{


    cublasHandle_t cublasHandle = *(cublasHandle_t*)cublasHandlerPtr;
    float alpha = 1, beta = 1;

    for (unsigned int i = 0; i < loopCount; i++)
    {
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                __matrixSize, __matrixSize, __matrixSize,
                &alpha,
                d_a, CUDA_R_16F, __matrixSize,
                d_b, CUDA_R_16F, __matrixSize,
                &beta,
                d_c, CUDA_R_32F, __matrixSize,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);    
        }
    
}

void MMAOptCublas::Cleanup()
{
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    d_out = nullptr;
}