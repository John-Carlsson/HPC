#include "benchmarking.hpp"
#include "math_tensor.hpp"
#include "math_cuda.hpp"
#include <stdio.h>

// Define some error checking macros.
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

void vector_add_tensor(float *out, float *a, float *b, int n)
{
    vector_add_cuda(out, a, b, n);
}

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

using namespace nvcuda;



#define WMMA_TILE_SIZE 16

__global__ void mat_mul_add_tensor(half *a, half *b, float *c, float *d, int N)
{
    // Tile using a 2D grid
    int WarpX = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int WarpY = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, float> c_frag;

    if(c != nullptr)
        wmma::load_matrix_sync(c_frag, c + WarpX * 16 + WarpY * 16 * N, N, wmma::mem_row_major);
    else
        wmma::fill_fragment(c_frag, 0.0f);


    // Loop over k
    for (int i = 0; i < N; i += WMMA_TILE_SIZE)
    {

        wmma::load_matrix_sync(a_frag, a + WarpY * 16 * N + i, N); // WarpY * 16 * N + i
        wmma::load_matrix_sync(b_frag, b + WarpX * 16 + i * N, N); // WarpX * 16 + i * N
        
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    int cRow = WarpX * WMMA_TILE_SIZE;
    int cCol = WarpY * WMMA_TILE_SIZE;
    wmma::store_matrix_sync(d + cCol + cRow * N, c_frag, N, wmma::mem_row_major);
}

template <unsigned int BLOCK_SIZE, unsigned int WMMA_SIZE>
__global__ void mat_mul_add_tensor_shared_mem(half *a, half *b, float *c, float *d, int N)
{
    __shared__ half a_shared [BLOCK_SIZE * BLOCK_SIZE];
    __shared__ half b_shared [BLOCK_SIZE * BLOCK_SIZE];

    // The kernel/thread global id, row i of left matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile using a 2D grid
    unsigned int WarpX = col / warpSize;
    unsigned int WarpY = row;

    // Tile using a 2D grid
    unsigned int LocalWarpXCount = blockDim.x / warpSize;
    unsigned int LocalWarpX = threadIdx.x / warpSize;
    unsigned int LocalWarpY = threadIdx.y;

    unsigned int LocalThreadId = threadIdx.x % warpSize;
    unsigned int loadTileCount = LocalWarpXCount * blockDim.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float> c_frag;

    if(c != nullptr)
        wmma::load_matrix_sync(c_frag, c + WarpX * 16 + WarpY * 16 * N, N, wmma::mem_row_major);
    else
        wmma::fill_fragment(c_frag, 0.0f);


    // Loop over k
    for (int i = 0; i < N; i += BLOCK_SIZE)
    {
        for (size_t l = 0; l < BLOCK_SIZE; l+=loadTileCount)
        {
            unsigned int local_row = l * (LocalWarpXCount * LocalWarpY + LocalWarpX);
            unsigned int local_col = LocalThreadId;
            int shared_mem_index = (local_row)*BLOCK_SIZE + local_col;
            printf("%d %d <- A%d %d B%d %d\n", local_row , local_col, blockIdx.y * BLOCK_SIZE + local_row, i + local_col,i + local_row,threadIdx.y* BLOCK_SIZE  + local_col);
            a_shared[shared_mem_index] = a[(blockIdx.y * BLOCK_SIZE + local_row)*N + i + local_col];
            b_shared[shared_mem_index] = b[(i + local_row) *N + threadIdx.y* BLOCK_SIZE  + local_col];
        }
        

        // load into shared memory, coalesced
        //a_shared[threadIdx.y *BLOCK_SIZE + threadIdx.x] = a[row*N + i + threadIdx.x];
        //b_shared[threadIdx.y *BLOCK_SIZE + threadIdx.x] = b[i * N +threadIdx.y* BLOCK_SIZE  + threadIdx.x];
        
        // sync before computation
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k+=WMMA_SIZE)
        {
            wmma::load_matrix_sync(a_frag, a_shared + k + LocalWarpY * WMMA_SIZE, BLOCK_SIZE); // WarpY * 16 * N + i
            wmma::load_matrix_sync(b_frag, b_shared + LocalWarpX + k, BLOCK_SIZE); // WarpX * 16 + i * N
            
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        

        
    }
    int cRow = WarpX * WMMA_SIZE;
    int cCol = WarpY * WMMA_SIZE;
    wmma::store_matrix_sync(d + cCol + cRow * N, c_frag, N, wmma::mem_row_major);
}

void matrix_multiplication_tensor(float *out, float *a, float *b, int n)
{
    half *t_a = (half *)malloc(sizeof(half) * n * n);
    for (size_t i = 0; i < n * n; i++)
    {
        t_a[i] = (half)a[i];
    }
    half *t_b = (half *)malloc(sizeof(half) * n * n);
    for (size_t i = 0; i < n * n; i++)
    {
        t_b[i] = (half)b[i];
    }

    half *d_a, *d_b;
    float *d_out;
    cudaErrCheck(cudaMalloc((void **)&d_a, sizeof(half) * n * n));
    cudaErrCheck(cudaMemcpy(d_a, t_a, n * n * sizeof(half), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&d_b, sizeof(half) * n * n));
    cudaErrCheck(cudaMemcpy(d_b, t_b, n * n * sizeof(half), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&d_out, sizeof(float) * n * n));
    cudaErrCheck(cudaMemcpy(d_out, out, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    //gridDim.x = (n + (WMMA_TILE_SIZE * blockDim.x / 32) - 1) / (WMMA_TILE_SIZE * blockDim.x / 32);
    //gridDim.y = (n + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    gridDim.x = n/(blockDim.x/32 * 16);
    gridDim.y = n/(blockDim.y*16);

    mat_mul_add_tensor<<<gridDim, blockDim>>>(d_a, d_b, nullptr, d_out, n);

    cudaErrCheck(cudaMemcpy(out, d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(t_a);
    free(t_b);
}

void matrix_multiplication_add_tensor(float *out, float *a, float *b, float *c, int n)
{

    half *t_a = (half *)malloc(sizeof(half) * n * n);
    for (size_t i = 0; i < n * n; i++)
    {
        t_a[i] = (half)a[i];
    }
    half *t_b = (half *)malloc(sizeof(half) * n * n);
    for (size_t i = 0; i < n * n; i++)
    {
        t_b[i] = (half)b[i];
    }

    half *d_a, *d_b;
    float *d_c, *d_out;
    cudaErrCheck(cudaMalloc((void **)&d_a, sizeof(half) * n * n));
    cudaErrCheck(cudaMemcpy(d_a, t_a, n * n * sizeof(half), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&d_b, sizeof(half) * n * n));
    cudaErrCheck(cudaMemcpy(d_b, t_b, n * n * sizeof(half), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&d_out, sizeof(float) * n * n));
    cudaErrCheck(cudaMemcpy(d_out, out, n * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&d_c, sizeof(float) * n * n));
    cudaErrCheck(cudaMemcpy(d_c, c, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    //gridDim.x = (n + (WMMA_TILE_SIZE * blockDim.x / 32) - 1) / (WMMA_TILE_SIZE * blockDim.x / 32);
    //gridDim.y = (n + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    gridDim.x = n/(blockDim.x/32 * 16);
    gridDim.y = n/(blockDim.y*16);

    mat_mul_add_tensor<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_out, n);

    cudaErrCheck(cudaMemcpy(out, d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
    free(t_a);
    free(t_b);
}


MMAOptTensor::MMAOptTensor(float *A, float *B, float *C, float *Out, unsigned int size) :
        __matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{
    
}

const char *MMAOptTensor::GetOPTMame()
{
    return name;
}

double MMAOptTensor::Import()
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

    if (d_out == nullptr) cudaMalloc((void **)&(this->d_out), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_out, this->_Out, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_c == nullptr) cudaMalloc((void **)&(this->d_c), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_c, this->_C, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    free(t_a);
    free(t_b);
    ,time)
    return time;
}

double MMAOptTensor::Compute()
{
    
    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 256;
    blockDim.y = 4;//8

    gridDim.x = this->__matrixSize/(blockDim.x/32 * 16);
    gridDim.y = this->__matrixSize/(blockDim.y*16);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add_tensor<<<gridDim, blockDim>>>((half*)this->d_a, (half*)this->d_b, this->d_c, this->d_out, this->__matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

double MMAOptTensor::Export()
{
    double time;
    BENCH_STORE(
    cudaMemcpy(this->_Out, this->d_out, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
    ,time)
    return time;
}

void MMAOptTensor::ComputeNTime(unsigned int loopCount)
{
    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = this->__matrixSize/(blockDim.x/32 * 16);
    gridDim.y = this->__matrixSize/(blockDim.y*16);
    for (unsigned int i = 0; i < loopCount; i++)
    {
        mat_mul_add_tensor<<<gridDim, blockDim>>>((half*)this->d_a, (half*)this->d_b, this->d_c, this->d_out, this->__matrixSize);
    }
    
}

void MMAOptTensor::Cleanup()
{
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
    if (d_out != nullptr) cudaFree(d_out);
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    d_out = nullptr;
}

MMAOptTensorShared::MMAOptTensorShared(float *A, float *B, float *C, float *Out, unsigned int size) :
        __matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{
    
}

const char *MMAOptTensorShared::GetOPTMame()
{
    return name;
}

double MMAOptTensorShared::Import()
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

    if (d_out == nullptr) cudaMalloc((void **)&(this->d_out), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_out, this->_Out, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_c == nullptr) cudaMalloc((void **)&(this->d_c), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_c, this->_C, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    free(t_a);
    free(t_b);
    ,time)
    return time;
}

double MMAOptTensorShared::Compute()
{
    float time;
    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 64;
    blockDim.y = blockDim.x/32;

    gridDim.x = this->__matrixSize/(blockDim.x/32 * 16);
    gridDim.y = this->__matrixSize/(blockDim.y*16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add_tensor_shared_mem<32,16><<<gridDim, blockDim>>>((half*)this->d_a, (half*)this->d_b, this->d_c, this->d_out, this->__matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

double MMAOptTensorShared::Export()
{
    double time;
    BENCH_STORE(
    cudaMemcpy(this->_Out, this->d_out, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
    ,time)
    return time;
}

void MMAOptTensorShared::ComputeNTime(unsigned int loopCount)
{
    dim3 gridDim, blockDim;
    // 16 warps in one block
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 64;
    blockDim.y = blockDim.x/32;

    gridDim.x = this->__matrixSize/(blockDim.x/32 * 16);
    gridDim.y = this->__matrixSize/(blockDim.y*16);
    for (unsigned int i = 0; i < loopCount; i++)
    {
        mat_mul_add_tensor_shared_mem<32,16><<<gridDim, blockDim>>>((half*)this->d_a, (half*)this->d_b, this->d_c, this->d_out, this->__matrixSize);
    }
    
}

void MMAOptTensorShared::Cleanup()
{
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
    if (d_out != nullptr) cudaFree(d_out);
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    d_out = nullptr;
}


