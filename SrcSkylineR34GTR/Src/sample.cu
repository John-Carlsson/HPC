//////////////////////////////////////////////////////////////////////
// A simple example to show how CUDA WMMA API works with Tensor Cores
//    Created by Zong-Sheng Wang @ 2018/11/25
// Performance Tips:
//    To minimize bank conflicts, you should try to shift row or 
// column of matrics in shared memory
// cmd: 
//    $ nvcc -o main main.cu -arch sm_75

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 16
#define N_TILES 16
#define K_TILES 16

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

//__global__ void WMMAINT8()
using namespace nvcuda;

__host__ void matrix_multiplication_add_cpu(float *out, float *a, float *b, float *c, int n)
{
    unsigned int ih, jh, kh, il, kl, jl;
    float tile[M][M];
    //#pragma omp parallel for collapse(3) private(tile)
    for (ih = 0; ih < n; ih += M)
        for (jh = 0; jh < n; jh += M)
            for (kh = 0; kh < n; kh += M)
            {
                for (il = 0; il < M; il ++)
                    #pragma unroll
                    for (jl = 0; jl < M; jl ++)
                        tile[il][jl] = 0.;

                for (il = 0; il < M; il ++)
                    for (kl = 0; kl < M; kl ++)
                        #pragma unroll
                        for (jl = 0; jl < M; jl ++)
                            tile[il][jl] += a[(ih+il)*n+kh+kl] * b[(kh+kl)*n+jh+jl];

                for (il = 0; il < M; il ++)
                    #pragma unroll
                    for (jl = 0; jl < M; jl ++)
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

void print_m(float *OUT, size_t nx, size_t ny)
{
    for (size_t i = 0; i < nx*ny; i++)
        printf("%f,%s", OUT[i], (i + 1) % nx == 0 ? "\n":"");
}

void print_m2(half *OUT, size_t nx, size_t ny)
{
    for (size_t i = 0; i < nx*ny; i++)
        printf("%f,%s", OUT[i], (i + 1) % nx == 0 ? "\n":"");
}


__host__ void InitMatrix(half *A, half *B, float *C, float* A2, float* B2, float* C2)
{
	for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
    {
        half t = __float2half(rand() % 1000 / 1000.0f);
		A[i] = t;
        A2[i] = (float)t;
    }
        
	for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
    {
        half t = __float2half(rand() % 1000 / 1000.0f);
		B[i] = t;
        B2[i] = (float)t;
    }
        
	for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
    {
        float t = rand() % 1000 / 1000.0f;
		C[i] = t;
        C2[i] = t;
    }

}



__global__ void WMMAF16TensorCore(half *A, half *B, float *C, float *D)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);
	
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
	
	wmma::fill_fragment(ab_frag, 0.0f);

	// AB = A*B
	int a_col, a_row, b_col, b_row, c_col, c_row;
	a_row = ix * M;
	b_col = iy * N;
	for (int k=0; k<K_TOTAL; k+=K) {
		a_col = b_row = k;

		if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, A + a_row + a_col * M_TOTAL, M_TOTAL);
			wmma::load_matrix_sync(b_frag, B + b_row + b_col * K_TOTAL, K_TOTAL);

			// Perform the matrix multiplication
			wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	// D = AB + C
	c_col = b_row;
	c_row = a_row;
	if (c_row < M_TOTAL && c_col < N_TOTAL) {
		wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
	}
}

cudaError_t CalcWMMA(half *A, half *B, float *C, float *D)
{
	cudaError_t cuda_status;
	dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 4 * WARP_SIZE; 
	blockDim.y = 4;

	gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
	gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

	// for Performance Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D);
	cuda_status = cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// for Performance Metrics
	printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
	// references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2) / milliseconds / 1e9);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return cuda_status;
}




int main()
{
	cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		printf("cudaSetDevice failed! ");
		return 1;
	}


	// Matrix on device
	half *A;
	half *B;
	float *C;
	float *D;
	float *D2  = (float*)malloc(sizeof(float) * M_TOTAL * N_TOTAL);
    float *A2 = (float*)malloc(sizeof(float) * M_TOTAL * K_TOTAL);
	float *B2 = (float*)malloc(sizeof(float) * K_TOTAL * N_TOTAL);
	float *C2 = (float*)malloc(sizeof(float) * M_TOTAL * N_TOTAL);
	float *E  = (float*)malloc(sizeof(float) * M_TOTAL * N_TOTAL);

	// CUDA Unified Memory 
	cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * K_TOTAL);
	cudaMallocManaged((void **)&B, sizeof(half) * K_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&C, sizeof(float) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&D, sizeof(float) * M_TOTAL * N_TOTAL);
	
	// Init matrix A B C on host
	//InitHostMatrix(host_A, host_B, host_C);
	printf("[*] Initializing Matrix...\n");
	InitMatrix(A, B, C, A2, B2, C2);
	printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
	printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
	printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);
	
	// computing gemm using tensor core
	printf("[*] Computing D = A * B +C with Tensor Cores...\n");
	// D = A * B +C, D holds the result after ret
	cuda_status = CalcWMMA(A, B, C, D);
    cudaMemcpy(D, D2, M_TOTAL * N_TOTAL*sizeof(float), cudaMemcpyDeviceToHost);
    matrix_multiplication_add_cpu(E,A2,B2,C2,M_TOTAL);
	
	cuda_status = cudaDeviceReset();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceReset failed! ");
		return 1;
	}
	// Todo: Add a function to verify the result by using the result of CPU version implementation.
    int is_bad = 0;
    for (size_t i = 0; i < N * N; i++)
    {
        if (E[i]!=D2[i])
        {
            is_bad = 1;
            break;
        }
    }
    if (is_bad)
    {
        printf("CPU, \n");
        print_m(E, N, N);
        printf("Tensor \n");
        print_m(D2, N, N);
    }
    


	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	cudaFree(E);

	return 0;
}