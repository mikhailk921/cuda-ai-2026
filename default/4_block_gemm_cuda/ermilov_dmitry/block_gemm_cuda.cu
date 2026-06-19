#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void BlockGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int m = 0; m < n; m += TILE) {

        As[ty][tx] = A[row * n + (m + tx)];
        Bs[ty][tx] = B[(m + ty) * n + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);

    float *vidA = nullptr;
    float *vidB = nullptr;
    float *vidC = nullptr;

    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMalloc(&vidA, bytes);
    cudaMalloc(&vidB, bytes);
    cudaMalloc(&vidC, bytes);

    cudaMemcpy(vidA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vidB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);

    BlockGemmKernel<<<grid, block>>>(vidA, vidB, vidC, n);

    cudaMemcpy(c.data(), vidC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(vidA);
    cudaFree(vidB);
    cudaFree(vidC);

    return c;
}