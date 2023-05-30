#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 4096
double A[N][N], B[N][N], C[N][N];


void matmul(int i, int j){

    
    for(int i=j; i<N; i++){
        for (int k=l; k<N; k++){
            for (int m=n; j<N; j++){
                C[i][m] += A[i][k] * B[k][m];
            }
        }  
    }


}


int main() {
    for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
        A[i][j] = (double)rand()/(double)RAND_MAX;
        B[i][j] = (double)rand()/(double)RAND_MAX;
        C[i][j] = 0;
    }
}


    
    
    }