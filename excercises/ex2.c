
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


double f(double x) {
    return(4.0 / (1.0 + x*x));
    }

double CalcPi(int n) {
        
    const double fH = 1.0 / (double) n;
    double fSum = 0.0;
    double fX;
    int i;

    #pragma omp parallel for private(fX,i) reduction(+:fSum)
    for(i = 0; i < n; i++){
        fX = fH* ((double)i + 0.5);
        fSum += f(fX);
        }
    return fH* fSum;
}

int main() {
    float s = CalcPi(234);
    
    printf("%f\n",s);
    
    
    }
