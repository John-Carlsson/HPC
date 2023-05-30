#include <omp.h>
#include <arm_neon.h>
#include <stdlib.h>

void matmul(float* A, float* B, float* C, int M, int N, int K, int tile_size) {
    const int num_tiles_m = (M + tile_size - 1) / tile_size;
    const int num_tiles_n = (N + tile_size - 1) / tile_size;
    const int num_tiles_k = (K + tile_size - 1) / tile_size;

    // Allocate padded matrices
    float* A_padded = (float*) malloc(num_tiles_m * tile_size * K * sizeof(float));
    float* B_padded = (float*) malloc(num_tiles_n * tile_size * K * sizeof(float));
    float* C_padded = (float*) malloc(num_tiles_m * tile_size * num_tiles_n * tile_size * sizeof(float));

    // Copy data to padded matrices
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            A_padded[i*K + k] = A[i*K + k];
        }
        for (int k = K; k < num_tiles_k*tile_size; ++k) {
            A_padded[i*K + k] = 0.0f;
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            B_padded[j*K + k] = B[j*K + k];
        }
        for (int k = K; k < num_tiles_k*tile_size; ++k) {
            B_padded[j*K + k] = 0.0f;
        }
    }
    for (int i = 0; i < num_tiles_m * tile_size; ++i) {
        for (int j = 0; j < num_tiles_n * tile_size; ++j) {
            C_padded[i*num_tiles_n*tile_size + j] = 0.0f;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int tile_m = 0; tile_m < num_tiles_m; ++tile_m) {
        for (int tile_n = 0; tile_n < num_tiles_n; ++tile_n) {
            int m_start = tile_m * tile_size;
            int m_end = (tile_m + 1) * tile_size;
            if (m_end > M) m_end = M;

            int n_start = tile_n * tile_size;
            int n_end = (tile_n + 1) * tile_size;
            if (n_end > N) n_end = N;

            for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
                int k_start = tile_k * tile_size;
                int k_end = (tile_k + 1) * tile_size;
                if (k_end > K) k_end = K;

                for (int i = m_start; i < m_end; ++i) {
                    for (int j = n_start; j < n_end; ++j) {
                        float32x4_t acc = vdupq_n_f32(0.0f);
                        for (int k = k_start; k < k_end; k += 4) {
                            float32x
