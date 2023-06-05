#include "math_cuda.hpp"
#include "benchmarking.hpp"

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
template<typename In, typename Out>
__global__ void mat_mul_add_global_mem(Out *out, In *a, In *b, Out *c, int n)
{   
    // The kernel global id
    unsigned int task_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int task_idy = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0;
    for (unsigned int k = 0; k < n; k ++){
        float el_a = a[task_idy * n + k];
        float el_b = b[task_idx + n * k];
        value += (Out)el_a * el_b;
    }

    out[task_idy*n+task_idx] = c[task_idy*n+task_idx] + value;
}

template<typename In, typename Out, unsigned int TILE_SIZE>
__global__ void mat_mul_add(Out *out, In *A, In *B, Out *c, int n)
{
    __shared__ In a_shared [TILE_SIZE][TILE_SIZE];
    __shared__ In b_shared [TILE_SIZE][TILE_SIZE];
    
    // The kernel/thread global id, row i of left matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;


    Out sum_acc = row < n && col < n ? c[row*n + col] : 0;
    for (size_t tileNum = 0; tileNum < n/TILE_SIZE; tileNum++)
    {

        // j is the column index of the left matrix
        int j = tileNum*TILE_SIZE + threadIdx.x;
        int i = tileNum*TILE_SIZE + threadIdx.y;

        // load into shared memory, coalesced
        a_shared[threadIdx.y][threadIdx.x] = A[row*n + j];
        b_shared[threadIdx.y][threadIdx.x] = B[i*n + col];
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum_acc += (Out)a_shared[threadIdx.y][k] * (Out)b_shared[k][threadIdx.x];
            
        }
    }

    out[row*n + col] = sum_acc;
    
}   

template<typename In, typename Out, unsigned int TILE_SIZE, unsigned int FACTOR>
__global__ void mat_mul_add_rect(Out *out, In *A, In *B, Out *c, int n)
{
    __shared__ In a_shared [FACTOR][TILE_SIZE][TILE_SIZE];
    __shared__ In b_shared [TILE_SIZE][TILE_SIZE];
    
    // The kernel/thread global id, row i of left matrix
    unsigned int row = blockIdx.y * blockDim.y * FACTOR + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    Out accs[FACTOR];
    for (unsigned int subthread = 0; subthread < FACTOR; subthread++)
        accs[subthread] = c[(row + TILE_SIZE * subthread)*n + col];

    for (unsigned int tileNum = 0; tileNum < n/TILE_SIZE; tileNum++)
    {
        unsigned int BRow = tileNum*TILE_SIZE + threadIdx.y;
        unsigned int BCol = col;

        b_shared[threadIdx.y][threadIdx.x] = (BRow < n && BCol < n) ? B[BRow*n + BCol] : (In)0;
        for (unsigned int subthread = 0; subthread < FACTOR; subthread++)
        {
            unsigned int ARow = row + TILE_SIZE * subthread;
            unsigned int ACol = tileNum*TILE_SIZE + threadIdx.x;
            a_shared[subthread][threadIdx.y][threadIdx.x] = (ARow < n && ACol < n) ? A[ARow*n + ACol] : (In)0;
            //printf("%d %d %d %d %d\n",subthread, row, col, ARow, ACol);
        }
        __syncthreads();
        for (unsigned int subthread = 0; subthread < FACTOR; subthread++)
            for (int k = 0; k < TILE_SIZE; k++)
            {
                accs[subthread] += (Out)(a_shared[subthread][threadIdx.y][k] * b_shared[k][threadIdx.x]);
            }
    }
    for (unsigned int subthread = 0; subthread < FACTOR; subthread++)
        out[(row + TILE_SIZE * subthread)*n + col] = accs[subthread];


    /*
    const unsigned int ratio = TILE_SIZE_Y/TILE_SIZE;

    for (unsigned int tileNumYA = 0; tileNumYA < n/(TILE_SIZE * FACTOR); tileNumYA++)
    {
        for (unsigned int tileNum  = 0; tileNum < FACTOR; tileNum++)
        {
            unsigned int ARow = row + TILE_SIZE * tileNum;
            unsigned int ACol = (tileNumYA * ratio + tileNum)*TILE_SIZE + threadIdx.x;
            unsigned int BRow = (tileNumYA * ratio + tileNum)*TILE_SIZE + threadIdx.y;
            unsigned int BCol = col;

            a_shared[tileNum][threadIdx.y][threadIdx.x] = (ARow < n && ACol < n) ? A[ARow*n + ACol] : (In)0;
            if (!tileNum)
                b_shared[threadIdx.y][threadIdx.x] = (BRow < n && BCol < n) ? B[BRow*n + BCol] : (In)0;

            __syncthreads();
        }
        for (unsigned int tileNum = 0; tileNum < ratio; tileNum++)
            for (int k = 0; k < TILE_SIZE; k++)
            {
                accs[tileNum] += (Out)(a_shared[tileNum][threadIdx.y][k] * b_shared[k][threadIdx.x]);
                
            }
    }
    for (unsigned int tileNum = 0; tileNum < ratio; tileNum++)
        out[(row + TILE_SIZE * tileNum)*n + col] = accs[tileNum];
*/
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
    mat_mul_add<float, float, 32><<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, n);
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

double MMAOptCUDA::Import()
{
    double time;
    BENCH_STORE(
    if (d_a == nullptr) cudaMalloc((void **)&(this->d_a), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_a, this->_A, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_b == nullptr) cudaMalloc((void **)(&this->d_b), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_b, this->_B, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_out == nullptr) cudaMalloc((void **)&(this->d_out), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_out, this->_Out, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    if (d_c == nullptr) cudaMalloc((void **)&(this->d_c), sizeof(float) * __matrixSize*__matrixSize);
    cudaMemcpy(this->d_c, this->_C, __matrixSize*__matrixSize*sizeof(float), cudaMemcpyHostToDevice);

    ,time)
    return time;
}

const char *MMAOptCUDA::GetOPTMame()
{
    return name;
}

double MMAOptCUDA::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 32;
    blockDim.y = 32;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add<float, float, 32><<<gridDim, blockDim>>>(this->d_out, this->d_a, this->d_b, this->d_c, this->__matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

double MMAOptCUDA::Export()
{
    double time;
    BENCH_STORE(
    cudaMemcpy(this->_Out, this->d_out, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
        ,time)
    return time;
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
        mat_mul_add<float, float, 16><<<gridDim, blockDim>>>(this->d_out, this->d_a, this->d_b, this->d_c, this->__matrixSize);
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

double MMAOptCUDAGlobMem::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 numblocks(__matrixSize/32,__matrixSize/32);
    dim3 thread_per_block(32,32);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add_global_mem<float, float><<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, __matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
   
}

void MMAOptCUDAGlobMem::ComputeNTime(unsigned int loopCount)
{
    if(d_out == nullptr) return;

    dim3 numblocks(__matrixSize/32,__matrixSize/32);
    dim3 thread_per_block(32,32);
   
    for (unsigned int i = 0; i < loopCount; i++)
        mat_mul_add_global_mem<float, float><<<numblocks, thread_per_block>>>(d_out, d_a, d_b, d_c, __matrixSize);
    
    
}

const char *MMAOptCUDAGlobMem::GetOPTMame()
{
    return name;
}

MMAOptCUDAH::MMAOptCUDAH(float *A, float *B, float *C, float *Out, unsigned int size) :
        __matrixSize(size),
        _A(A),
        _B(B),
        _C(C),
        _Out(Out)
{
}

double MMAOptCUDAH::Import()
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

const char *MMAOptCUDAH::GetOPTMame()
{
    return name;
}

double MMAOptCUDAH::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 32;
    blockDim.y = 32;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/(blockDim.y*2));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add_rect<half, float, 32, 2><<<gridDim, blockDim>>>(d_out, (half*)d_a, (half*)d_b, d_c, __matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

double MMAOptCUDAH::Export()
{
    double time;
    BENCH_STORE(
    cudaMemcpy(this->_Out, this->d_out, this->__matrixSize*this->__matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
        ,time)
    return time;
}

void MMAOptCUDAH::ComputeNTime(unsigned int loopCount)
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
        mat_mul_add<half, float, 16><<<gridDim, blockDim>>>(d_out, (half*)d_a, (half*)d_b, d_c, __matrixSize);
    }
    
}

void MMAOptCUDAH::Cleanup()
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


const char *MMAOptCUDASF16::GetOPTMame()
{
    return name;
}

double MMAOptCUDASF16::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 16;
    blockDim.y = 16;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add<float, float, 16><<<gridDim, blockDim>>>(this->d_out, this->d_a, this->d_b, this->d_c, this->__matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}



const char *MMAOptCUDASH32::GetOPTMame()
{
    return name;
}

double MMAOptCUDASH32::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 32;
    blockDim.y = 32;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add<half, float, 32><<<gridDim, blockDim>>>(d_out, (half*)d_a, (half*)d_b, d_c, __matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}

const char *MMAOptCUDAH2::GetOPTMame()
{
    return name;
}

double MMAOptCUDAH2::Compute()
{
    if(d_out == nullptr) return -1;

    float time;
    dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 32;
    blockDim.y = 32;

    gridDim.x = (this->__matrixSize/blockDim.x);
    gridDim.y = (this->__matrixSize/(blockDim.y*3));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul_add_rect<half, float, 32, 3><<<gridDim, blockDim>>>(d_out, (half*)d_a, (half*)d_b, d_c, __matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return (double)time;
}