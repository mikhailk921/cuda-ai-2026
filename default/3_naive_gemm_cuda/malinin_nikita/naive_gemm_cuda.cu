#include "naive_gemm_cuda.h"

#include <cuda/cmath>

__global__ void NaiveGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j < n && i < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; ++k) {
            sum += a[j * n + k] * b[k * n + i];
        }

        c[j * n + i] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
    {
        const int mxSize = a.size();
        const int bytesSize = n * n * sizeof(float);

        float* gpuBufferA = nullptr;
        float* gpuBufferB = nullptr;
        float* gpuBufferC = nullptr;

        cudaMalloc(&gpuBufferA, bytesSize);
        cudaMalloc(&gpuBufferB, bytesSize);
        cudaMalloc(&gpuBufferC, bytesSize);

        cudaMemcpy(gpuBufferA, a.data(), bytesSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpuBufferB, b.data(), bytesSize, cudaMemcpyHostToDevice);
        cudaMemset(gpuBufferC, 0, bytesSize);

        constexpr int threads = 16;
        int blocks = cuda::ceil_div(n, threads);
        dim3 threadsDim(threads, threads);
        dim3 blocksDim(blocks, blocks);

        NaiveGemmCUDAImpl<<<blocksDim, threadsDim>>>(gpuBufferA, gpuBufferB, gpuBufferC, n);

        std::vector<float> c(mxSize);

        cudaDeviceSynchronize();
        cudaMemcpy(c.data(), gpuBufferC, bytesSize, cudaMemcpyDeviceToHost);

        cudaFree(gpuBufferA);
        cudaFree(gpuBufferB);
        cudaFree(gpuBufferC);

        return c;
    }
