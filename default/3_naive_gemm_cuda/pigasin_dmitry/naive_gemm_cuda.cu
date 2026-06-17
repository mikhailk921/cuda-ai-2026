#include "naive_gemm_cuda.h"

#include <thread>


__global__ void kernel(const float* a, const float* b, float* c, size_t n) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) {
        return;
    }

    float sum = 0.f;
    for (size_t k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t numElem = a.size();

    std::vector<float> c;
    std::thread t([&](){c.resize(numElem);});

    float *gpuA, *gpuB, *gpuC;
    const size_t numBytes = numElem * sizeof(float);
    cudaMalloc(&gpuA, numBytes);
    cudaMalloc(&gpuB, numBytes);
    cudaMalloc(&gpuC, numBytes);

    dim3 blockSize(16, 16);
    dim3 numBlocks(n / 16, n / 16);

    cudaMemcpy(gpuA, a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b.data(), numBytes, cudaMemcpyHostToDevice);
    kernel<<<numBlocks, blockSize>>>(gpuA, gpuB, gpuC, n);
    t.join();
    cudaMemcpy(c.data(), gpuC, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);

    return c;
}
