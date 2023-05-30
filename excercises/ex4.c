// Given an array of 1000 random numbers with values between [1,100],
// count how many values are within the bins [1,10], [11,20]... [91,100]
// to compile: clang -fopenmp -O2 ex4.c -o ex4 -L/opt/homebrew/lib -lomp
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define ARRAY_SIZE 100000

int main(){

    int arr[ARRAY_SIZE];
    int count[10] = {0}; // Initialize all elements to 0
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 100 + 1; // Generate a random value between 1 to 100 and assign it to the array
    }

    int sum = 0;
    double start_time = omp_get_wtime();

#pragma omp parallel for 
    for (int i = 0; i < ARRAY_SIZE; i++){
        int bin = (arr[i]-1) / 10;
        
        #pragma atomic
        {count[bin] ++;}
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    for (int i = 0; i < 10; i++){
        printf("bin: [%d-%d], count: %d\n", i*10+1, i*10+10, count[i]);
    }
    printf("Execution time: %f seconds\n", elapsed_time);

    

}