#ifndef BENCHMARKING_H
#define BENCHMARKING_H


#define USE_BENCHMARKING
#ifdef USE_BENCHMARKING
#include <chrono>
#include <stdio.h>

#define BENCH(code)\
{\
    auto t_begin = std::chrono::high_resolution_clock::now();\
    code\
    auto t_end = std::chrono::high_resolution_clock::now();\
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_begin).count();\
    printf("%f,", elapsed * 1e-6);\
}

#define BENCH_STORE(code, container)\
{\
    auto t_begin = std::chrono::high_resolution_clock::now();\
    code\
    auto t_end = std::chrono::high_resolution_clock::now();\
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_begin).count();\
    container = elapsed * 1e-6;\
}


#else
#define BENCH(code) code

#define BENCH_STORE(code, container) code
#endif
#endif