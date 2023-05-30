#include "math_cpu.hpp"

#include <stdlib.h>
#include <omp.h>

#include <cstring>

#define TILE_SIZE 32



void vector_add_cpu(float *out, float *a, float *b, int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

void matrix_multiplication_cpu(float *out, float *a, float *b, int n)
{
    unsigned int ih, jh, kh, il, kl, jl;
    #pragma omp parallel for collapse(3)
    for (ih = 0; ih < n; ih += TILE_SIZE)
        for (jh = 0; jh < n; jh += TILE_SIZE)
            for (kh = 0; kh < n; kh += TILE_SIZE)
                for (il = 0; il < TILE_SIZE; il ++)
                    for (kl = 0; kl < TILE_SIZE; kl ++)
                        for (jl = 0; jl < TILE_SIZE; jl ++)
                            out[(ih+il)*n+jh+jl] += a[(ih+il)*n+kh+kl] * b[(kh+kl)*n+jh+jl];
}

void matrix_multiplication_add_cpu(float *out, float *a, float *b, float *c, int n)
{
    unsigned int ih, jh, kh, il, kl, jl;
    float tile[TILE_SIZE][TILE_SIZE];
    #pragma omp parallel for collapse(3) private(tile, ih, jh, kh, il, kl, jl)
    for (ih = 0; ih < n; ih += TILE_SIZE)
        for (jh = 0; jh < n; jh += TILE_SIZE)
            for (kh = 0; kh < n; kh += TILE_SIZE)
            {
                for (il = 0; il < TILE_SIZE; il ++)
                    #pragma unroll
                    for (jl = 0; jl < TILE_SIZE; jl ++)
                        tile[il][jl] = 0.;

                for (il = 0; il < TILE_SIZE; il ++)
                    for (kl = 0; kl < TILE_SIZE; kl ++)
                        #pragma unroll
                        for (jl = 0; jl < TILE_SIZE; jl ++)
                            tile[il][jl] += a[(ih+il)*n+kh+kl] * b[(kh+kl)*n+jh+jl];

                for (il = 0; il < TILE_SIZE; il ++)
                    #pragma unroll
                    for (jl = 0; jl < TILE_SIZE; jl ++)
                        out[(ih+il)*n+jh+jl] += tile[il][jl];
            }

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            out[i*n+j] += c[i*n+j];
        }
        
    }
}

__attribute__((always_inline))
inline void matrix_multiplication_add_cpu_aligned(float *__ALIGNED AlignedO, float *__ALIGNED AlignedA, float *__ALIGNED AlignedB, float *__ALIGNED AlignedC, unsigned int _matrixSize)
{
    float tile[TILE_SIZE * TILE_SIZE] __ALIGNED;
    #pragma omp parallel for \
            private(tile) \
            collapse(3) 
        for (unsigned int ih = 0; ih < _matrixSize; ih += TILE_SIZE)
            for (unsigned int jh = 0; jh < _matrixSize; jh += TILE_SIZE)
                for (unsigned int kh = 0; kh < _matrixSize; kh += TILE_SIZE)
                {
                    #pragma omp simd \
                        aligned(tile:CPU_VECTOR_BYTE_SIZE) \
                        collapse(2)
                    for (unsigned int il = 0; il < TILE_SIZE; il ++)
                        for (unsigned int jl = 0; jl < TILE_SIZE; jl ++)
                            tile[il * TILE_SIZE + jl] = 0.;

                    #pragma omp simd \
                        aligned(AlignedA:CPU_VECTOR_BYTE_SIZE) \
                        aligned(AlignedB:CPU_VECTOR_BYTE_SIZE) \
                        aligned(AlignedO:CPU_VECTOR_BYTE_SIZE) \
                        collapse(3)
                    for (unsigned int il = 0; il < TILE_SIZE; il ++)
                        for (unsigned int kl = 0; kl < TILE_SIZE; kl ++)
                            for (unsigned int jl = 0; jl < TILE_SIZE; jl ++)
                                tile[il * TILE_SIZE + jl] += AlignedA[(ih+il)*_matrixSize+kh+kl] * AlignedB[(kh+kl)*_matrixSize+jh+jl];

                    #pragma omp simd \
                        aligned(AlignedO:CPU_VECTOR_BYTE_SIZE) \
                        aligned(tile:CPU_VECTOR_BYTE_SIZE) \
                        collapse(2)
                    for (unsigned int il = 0; il < TILE_SIZE; il ++)
                        for (unsigned int jl = 0; jl < TILE_SIZE; jl ++)
                            AlignedO[(ih+il)*_matrixSize+jh+jl] += tile[il * TILE_SIZE + jl];
                }

    #pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < _matrixSize; i++)
        for (unsigned int j = 0; j < _matrixSize; j++)
            AlignedO[i*_matrixSize+j] += AlignedC[i*_matrixSize+j];
}

MMAOptCPU::MMAOptCPU(float *A, float *B, float *C, float *Out, unsigned int size):
        _matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{}

const char *MMAOptCPU::GetOPTMame()
{
    return name;
}

void MMAOptCPU::Import()
{
    if (_AlignedA == nullptr) _AlignedA = (float*)aligned_alloc(CPU_VECTOR_BYTE_SIZE, _matrixSize * _matrixSize * sizeof(float));
    std::memcpy(_AlignedA, _A, _matrixSize * _matrixSize *sizeof(float));
    if (_AlignedB == nullptr) _AlignedB = (float*)aligned_alloc(CPU_VECTOR_BYTE_SIZE, _matrixSize * _matrixSize * sizeof(float));
    std::memcpy(_AlignedB, _B, _matrixSize * _matrixSize *sizeof(float));
    if (_AlignedC == nullptr) _AlignedC = (float*)aligned_alloc(CPU_VECTOR_BYTE_SIZE, _matrixSize * _matrixSize * sizeof(float));
    std::memcpy(_AlignedC, _C, _matrixSize * _matrixSize *sizeof(float));
    if (_AlignedOut == nullptr) _AlignedOut = (float*)aligned_alloc(CPU_VECTOR_BYTE_SIZE, _matrixSize * _matrixSize * sizeof(float));
    std::memcpy(_AlignedOut, _Out, _matrixSize * _matrixSize *sizeof(float));
}

void MMAOptCPU::Compute()
{
    if(_AlignedA == nullptr) return;

    float *AlignedA __ALIGNED = _AlignedA;
    float *AlignedB __ALIGNED = _AlignedB;
    float *AlignedC __ALIGNED = _AlignedC;
    float *AlignedO __ALIGNED = _AlignedOut;

    matrix_multiplication_add_cpu_aligned(AlignedO, AlignedA, AlignedB, AlignedC, _matrixSize);
}

void MMAOptCPU::Export()
{
    std::memcpy(_Out, _AlignedOut, _matrixSize * _matrixSize *sizeof(float));
}

void MMAOptCPU::ComputeNTime(unsigned int loopCount)
{
    if(_AlignedA == nullptr) return;

    float *AlignedA __ALIGNED = _AlignedA;
    float *AlignedB __ALIGNED = _AlignedB;
    float *AlignedC __ALIGNED = _AlignedC;
    float *AlignedO __ALIGNED = _AlignedOut;

    for (unsigned int i = 0; i < loopCount; i++)
    {
        matrix_multiplication_add_cpu_aligned(AlignedO, AlignedA, AlignedB, AlignedC, _matrixSize);
    }
   }

void MMAOptCPU::Cleanup()
{
    if (_AlignedA != nullptr) free(_AlignedA);
    if (_AlignedB != nullptr) free(_AlignedB);
    if (_AlignedC != nullptr) free(_AlignedC);
    if (_AlignedOut != nullptr) free(_AlignedOut);
    _AlignedA = nullptr;
    _AlignedB = nullptr;
    _AlignedC = nullptr;
    _AlignedOut = nullptr;
}
