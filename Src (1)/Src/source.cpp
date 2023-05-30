//to run the program you can run the command "make all", results will be outputed in results.csv

#include "math_opts.hpp"
#include "math_cpu.hpp"
#include "math_cuda.hpp"
#include "math_tensor.hpp"
#include "benchmarking.hpp"


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <vector>
#include <numeric>

//#define N 256
#define PLATFORM CPU
#define ExecutionCount 50

#define MakeMatrix(name, sizeX, sizeY, cell_value)\
float *name = NULL;\
name = (float *)malloc(sizeof(float) * sizeX * sizeY);\
for (size_t name##_i = 0; name##_i < sizeX*sizeY; name##_i++)\
        name[name##_i] = cell_value;\

void print_m(float *OUT, size_t nx, size_t ny)
{
    for (size_t i = 0; i < nx*ny; i++)
        printf("%f,%s", OUT[i], (i + 1) % nx == 0 ? "\n":"");
}

void test(float *OUT, size_t nx, size_t ny)
{
    float e = OUT[0];
    for (size_t i = 0; i < nx*ny; i++)
    {
        if (e!=OUT[i])
        {
            printf("invalid, %f at %ld %ld\n",OUT[i], i/nx, i%nx);
        }
    }
    
    printf("%f \n", OUT[0]);
#if N < 5
    print_m(OUT, nx, ny);
#endif
}
#include <omp.h>
double getAverage(float* v, size_t size)
{
    double av =0.;
    #pragma omp parallel for reduction(+:av)
    for (size_t i = 0; i < size; i++)
    {
        av += (double)abs(v[i]);
    }
    av/= (double)size;
    return av;
}

double getStdDev(float* v, size_t size, double mean)
{
    double s =0.;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < size; i++)
    {
        s += pow((double)v[i] - mean, 2.);
    }
    s/= (double)size;
    return sqrt(s); 
}

void testCycle(size_t N, size_t CPU_threshold, int debugMatrices)
{
    MakeMatrix(A, N, N, (float)rand()/(float)(RAND_MAX))
    MakeMatrix(B, N, N, (float)rand()/(float)(RAND_MAX))
    MakeMatrix(C, N, N, (float)rand()/(float)(RAND_MAX))

    MakeMatrix(OutCPU, N, N, 0)
    MakeMatrix(OutCudaGM, N, N, 0)
    MakeMatrix(OutCuda, N, N, 0)
    MakeMatrix(OutTensor, N, N, 0)
    MakeMatrix(OutTensorS, N, N, 0)

    MMAOptCPU MMACpu(A, B, C, OutCPU, N);
    MMAOptCUDAGlobMem MMACudaGM(A, B, C, OutCudaGM, N);
    MMAOptCUDA MMACuda(A, B, C, OutCuda, N);
    MMAOptTensor MMATemsor(A, B, C, OutTensor, N);
    MMAOptTensorShared MMATemsorS(A, B, C, OutTensorS, N);
    std::vector<MMAOperation*> Ops = {&MMACpu, &MMACudaGM, &MMACuda, &MMATemsor, &MMATemsorS};
    std::vector<float*> OutBuffs = {OutCPU, OutCudaGM, OutCuda, OutTensor, OutTensorS};

    MakeMatrix(Res, Ops.size() * 3, ExecutionCount, -1.)
    MakeMatrix(ResRR, Ops.size() * 3, ExecutionCount, -1.)

    printf("\n");
    printf("Matrixes size,%ld,%ld,\n", N,N);

    printf("Single run computation test,\n");
    printf("Name,preparing time (ms),computation time (ms),gathering restuls time (ms),validity,error mean (to CPU),error deviation (to CPU),\n");

    for (size_t i = 0; i < Ops.size(); i++)
    {
        printf("%s,", Ops[i]->GetOPTMame());
        BENCH(Ops[i]->Import();)
        BENCH(Ops[i]->Compute();)
        BENCH(Ops[i]->Export();)
        Ops[i]->Cleanup();

        if (i == 0) 
        {
            printf("\n");
            continue;
        }

        MakeMatrix(DeviationMap, N, N, 0)
        float* Out = OutBuffs[i];

        int is_bad = 0;
        #pragma omp parallel for shared(is_bad)
        for (size_t j = 0; j < N * N; j++)
        {
            if (OutCPU[j]!=Out[j])
            {
                is_bad = 1;
                DeviationMap[j] = OutCPU[j] - Out[j];
            }
        }
        printf("%s,",is_bad ? "False": "True");
        if (!is_bad)
        {
            printf("0,0,\n");
            free(DeviationMap);
            continue;
        }
        double mean = getAverage(DeviationMap, N*N);
        printf("%f,",mean);
        double deviation = getStdDev(DeviationMap, N*N, mean);
        printf("%f,",deviation);
        printf("\n");
        free(DeviationMap);

    }

    if (debugMatrices)
    {
        MakeMatrix(OutTensor2, N, N, 0)
        matrix_multiplication_add_tensor(OutTensor2, A, B, C, N);
        for (size_t i = 0; i < Ops.size(); i++)
        {
            printf("%s,\n", Ops[i]->GetOPTMame());
            print_m(OutBuffs[i],N, N);
        }        

        printf("Tensor comp,");
        MakeMatrix(DeviationMap, N, N, 0)
        int is_bad = 0;
        for (size_t j = 0; j < N * N; j++)
        {
            if (OutTensor[j]!=OutTensor2[j])
            {
                is_bad = 1;
                DeviationMap[j] = OutTensor[j] - OutTensor2[j];
            }
        }
        printf("%s,",is_bad ? "False": "True");
        double mean = getAverage(DeviationMap, N*N);
        printf("%f,",mean);
        double deviation = getStdDev(DeviationMap, N*N, mean);
        printf("%f,",deviation);
        printf("\n");
    
    }

    printf("\n");
    printf("Multi run computation  test, execution count :%d,\n", ExecutionCount);
    for (size_t i = N > CPU_threshold ? 1 : 0; i < Ops.size(); i++)
    {
        
        for (size_t k = 0; k < ExecutionCount; k++)
        {
            BENCH_STORE(Ops[i]->Import();, Res[(i * 3 + 0)* ExecutionCount + k])
            BENCH_STORE(Ops[i]->Compute();, Res[(i * 3 + 1)* ExecutionCount + k])
            BENCH_STORE(Ops[i]->Export();, Res[(i * 3 + 2)* ExecutionCount + k])
        }
        Ops[i]->Cleanup();
    }

    for (size_t i = 0; i < Ops.size(); i++)
        printf("%s Loading,%s Computing,%s Outputing,", Ops[i]->GetOPTMame(), Ops[i]->GetOPTMame(), Ops[i]->GetOPTMame());
    printf("\n");
    for (size_t i = 0; i < ExecutionCount; i++)
        for (size_t j = 0; j < Ops.size() * 3; j++)
        {
            printf("%f,",Res[j* ExecutionCount + i]);
            if (j + 1 == Ops.size() * 3)
            {
                printf("\n");
            }
            
        }

    
    printf("\n");
    printf("Multi run computation test (with deallocation), execution count :%d,\n", ExecutionCount);
    for (size_t i = N > CPU_threshold ? 1 : 0; i < Ops.size(); i++)
    {

        for (size_t k = 0; k < ExecutionCount; k++)
        {
            BENCH_STORE(Ops[i]->Import();, ResRR[(i * 3 + 0)* ExecutionCount + k])
            BENCH_STORE(Ops[i]->Compute();, ResRR[(i * 3 + 1)* ExecutionCount + k])
            BENCH_STORE(Ops[i]->Export();, ResRR[(i * 3 + 2)* ExecutionCount + k])

            Ops[i]->Cleanup();
        }

    }

    for (size_t i = 0; i < Ops.size(); i++)
        printf("%s Loading,%s Computing,%s Outputing,", Ops[i]->GetOPTMame(), Ops[i]->GetOPTMame(), Ops[i]->GetOPTMame());
    printf("\n");
    for (size_t i = 0; i < ExecutionCount; i++)
        for (size_t j = 0; j < Ops.size() * 3; j++)
        {
            printf("%f,",ResRR[j* ExecutionCount + i]);
            if (j + 1 == Ops.size() * 3)
            {
                printf("\n");
            }
            
        }
    
    free(A);
    free(B);
    free(C);

    free(OutCPU);
    free(OutCuda);
    free(OutTensor);
    free(OutCudaGM);
    free(OutTensorS);

    free(Res);
    free(ResRR);

    
}

int main(int argc, char const *argv[])
{
    printf("Performing compairasons between CPU Cuda and Tensor implementation,\n");

    std::vector<size_t> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192*2};

    for (size_t size : sizes)
    {
        testCycle(size, 1025, 0);
    }
    

    return 0;
}
