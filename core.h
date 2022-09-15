#ifndef RNA_CORE_H
#define RNA_CORE_H

#ifdef WARPRNA_ENABLE_GPU
#include <cuda_runtime.h>
#endif

typedef enum {
    RNA_STATUS_SUCCESS = 0,
    RNA_STATUS_WARP_FAILED = 1,
    RNA_STATUS_GRADS_BLANK_FAILED = 2,
    RNA_STATUS_GRADS_LABEL_FAILED = 3,
    RNA_STATUS_COSTS_FAILED = 4
} rnaStatus_t;

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

#ifdef WARPRNA_ENABLE_GPU
rnaStatus_t run_warp_rna(
    cudaStream_t stream, unsigned int *counts, float *alphas, float *betas,
    const int *labels, const float *log_probs, float *grads, float *costs,
    const int *xn, const int *yn, int N, int T, int S, int U, int V, int blank);
#endif

#ifdef WARPRNA_ENABLE_CPU
rnaStatus_t run_warp_rna_cpu(
    unsigned int *counts, float *alphas, float *betas,
    const int *labels, const float *log_probs, float *grads, float *costs,
    const int *xn, const int *yn, int N, int T, int S, int U, int V, int blank);
#endif

#ifdef __cplusplus
}
#endif

#endif //RNA_CORE_H
