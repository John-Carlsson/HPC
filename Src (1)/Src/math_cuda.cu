#include "math_cuda.hpp"

#include <cuda_runtime_api.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;


#define MAT_MUL_ADD_ITERATION 2

__global__ void vector_add(float *out, float *a, float *b, int n)
{
    unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int task_count = blockDim.x * gridDim.x;
    unsigned int task_space = n / task_count;

    for (unsigned int i = task_id * task_space; i < (task_id + 1) * task_space; i++)
    {
        out[i] = a[i] + b[i];
    }
}

__global__ void mat_mul(float *out, float *a, float *b, int n)
{
    unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int task_count = blockDim.x * gridDim.x;
    unsigned int task_space = n / task_count;

    unsigned int i, j, k, s = task_id * task_space, e = (task_id + 1) * task_space;
    for (i = s; i < e; i ++)
        for (k = 0; k < n; k ++)
            for (j = 0; j < n; j ++)
                out[i*n+j] += a[i*n+k] * b[k*n+j];
}

#if MAT_MUL_ADD_ITERATION == 1
__global__ void mat_mul_add(float *out, float *a, float *b, float *c, int n)
{   
    // The kernel global id
    unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;

    // The amount of kernels
    unsigned int task_count = blockDim.x * gridDim.x;

    // The amount of work for a kernel
    unsigned int task_space = n / task_count;

    unsigned int i, j, k, s = task_id * task_space, e = (task_id + 1) * task_space;
    for (i = s; i < e; i ++)
        for (k = 0; k < n; k ++)
            for (j = 0; j < n; j ++)
                out[i*n+j] += a[i*n+k] * b[k*n+j];

    for (i = s; i < e; i++)
        for (j = 0; j < n; j++)
            out[i*n+j] += c[i*n+j];
}
#elif MAT_MUL_ADD_ITERATION ==2
__global__ void mat_mul_add_global_mem(float *out, float *a, float *b, float *c, int n)
{   
    // The kernel global id
    unsigned int task_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int task_idy = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0;
    for (unsigned int k = 0; k < n; k ++){
        float el_a = a[task_idy * n + k];
        float el_b = b[task_idx + n * k];
        value += el_a * el_b;
    }

    out[task_idy*n+task_idx] = c[task_idy*n+task_idx] + value;
}

#define BLOCK_SIZE 16
__global__ void mat_mul_add(float *out, float *A, float *B, float *c, int n)
{
    __shared__ float a_shared [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared [BLOCK_SIZE][BLOCK_SIZE];
    
    // The kernel/thread global id, row i of left matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    // Loop over the tiles
    for (int tileNum = 0; tileNum < n/BLOCK_SIZE; tileNum++)
    {   
        
        // j is the column index of the left matrix
        int j = tileNum*BLOCK_SIZE + threadIdx.x;
        int i = tileNum*BLOCK_SIZE + threadIdx.y;

        // load into shared memory, coalesced
        a_shared[threadIdx.y][threadIdx.x] = A[row*n + j];
        b_shared[threadIdx.y][threadIdx.x] = B[i*n + col];
        
        // sync before computation
        __syncthreads();
        

        
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            c[row*n + col] += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
            
        }
    __syncthreads();
    out[row*n + col] = c[row*n + col];
}
    
}   
        
#endif

void vector_add_cuda(float *out, float *a, float *b, int n)
{
    float *d_a;
    cudaMalloc((void **)&d_a, sizeof(float) * n);

    cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice);

    vector_add<<<128, 64>>>(out, d_a, b, n);

    cudaFree(d_a);
}

void matrix_multiplication_cuda(float *out, float *a, float *b, float *c, int n)
{
    float *d_a, *d_b, *d_out;

    cudaMalloc((void **)&d_a, sizeof(float) * n*n);
    cudaMemcpy(d_a, a, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_b, sizeof(float) * n*n);
    cudaMemcpy(d_b, b, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_out, sizeof(float) * n*n);
    cudaMemcpy(d_out, out, n*n*sizeof(float), cudaMemcpyHostToDevice);

    mat_mul<<<32, 32>>>(d_out, d_a, d_b, n);

    cudaMemcpy(out, d_out, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void matrix_multiplication_add_cuda(float *out, float *a, float *b, float *c, int n)
{
    float *d_a, *d_b, *d_c, *d_out;

    cudaMalloc((void **)&d_a, sizeof(float) * n*n);
    cudaMemcpy(d_a, a, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_b, sizeof(float) * n*n);
    cudaMemcpy(d_b, b, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_out, sizeof(float) * n*n);
    cudaMemcpy(d_out, out, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_c, sizeof(float) * n*n);
    cudaMemcpy(d_c, c, n*n*sizeof(float), cudaMemcpyHostToDevice);


#if     MAT_MUL_ADD_ITERATION == 1
    mat_mul_add<<<32, 32>>>(d_out, d_a, d_b, d_c, n);
#elif   MAT_MUL_ADD_ITERATION == 2
    dim3 numblocks(n/32,n/32);
    dim3 thread_per_block(32,32);
    mat_mul_add<<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, n);
#elif   MAT_MUL_ADD_ITERATION == 3
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 16;
    blockDim.y = 16;

    gridDim.x = (n/blockDim.x);
    gridDim.y = (n/blockDim.y);
    mat_mul_add<<<gridDim, blockDim>>>(d_out, d_a, d_b, d_c, n);
#endif
    cudaMemcpy(out, d_out, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
}

MMAOptCUDA::MMAOptCUDA(float *A, float *B, float *C, float *Out, unsigned int size) :
        __matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{
}

void MMAOptCUDA::Import()
{
    if (d_a == nullptr) cudaMalloc((void **)&(this->d_a), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_a, this->_A, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_b == nullptr) cudaMalloc((void **)(&this->d_b), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_b, this->_B, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_out == nullptr) cudaMalloc((void **)&(this->d_out), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_out, this->_Out, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_c == nullptr) cudaMalloc((void **)&(this->d_c), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_c, this->_C, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);
}

const char *MMAOptCUDA::GetOPTMame()
{
    return name;
}

void MMAOptCUDA::Compute()
{
    if(d_out == nullptr) return;

    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 16;
    blockDim.y = 16;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/blockDim.y);
    mat_mul_add<<<gridDim, blockDim>>>(this->d_out, this->d_a, this->d_b, this->d_c, this->__matrixSize);
}

void MMAOptCUDA::Export()
{
    cudaMemcpy(this->_Out, this->d_out, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
}

void MMAOptCUDA::ComputeNTime(unsigned int loopCount)
{
    if(d_out == nullptr) return;
    
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 16;
    blockDim.y = 16;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/blockDim.y);
    for (unsigned int i = 0; i < loopCount; i++)
    {
        mat_mul_add<<<gridDim, blockDim>>>(this->d_out, this->d_a, this->d_b, this->d_c, this->__matrixSize);
    }
    
}

void MMAOptCUDA::Cleanup()
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

void MMAOptCUDAGlobMem::Compute()
{
    if(d_out == nullptr) return;

    dim3 numblocks(__matrixSize/32,__matrixSize/32);
    dim3 thread_per_block(32,32);
    mat_mul_add_global_mem<<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, __matrixSize);
   
}

void MMAOptCUDAGlobMem::ComputeNTime(unsigned int loopCount)
{
    if(d_out == nullptr) return;

    dim3 numblocks(__matrixSize/32,__matrixSize/32);
    dim3 thread_per_block(32,32);
   
    for (unsigned int i = 0; i < loopCount; i++)
        mat_mul_add_global_mem<<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, __matrixSize);
    
    
}

const char *MMAOptCUDAGlobMem::GetOPTMame()
{
    return name;
}