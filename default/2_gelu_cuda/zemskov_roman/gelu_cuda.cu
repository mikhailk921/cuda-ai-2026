#include "gelu_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include <memory>
#include <vector>

__global__ void geluKernel(float * data, int size)
{
    int tredIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // const float s_2sqrt2_Pi = 1.595769f;
    if (tredIdx < size)
    {
        const float x = data[tredIdx];
        const float x2 = x * x;
        const float expx_plus1 = cuda::std::expf(1.595769f * x * (1.0f + 0.044715f * x2)) + 1.0f;
        data[tredIdx] = x - x / expx_plus1;
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    // Place your implementation here
    const size_t vecLen = input.size();
    const size_t size = vecLen * sizeof(float);

    float * devInput = nullptr;
    // float * devOut = nullptr;
    cudaMalloc(&devInput, size);

    // cudaMemcpyAsync(devInput, input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(devInput, input.data(), size, cudaMemcpyHostToDevice);

    constexpr int numThreads = 256;
    int numBlocks = cuda::ceil_div(vecLen, numThreads);
    geluKernel<<<numBlocks, numThreads>>>(devInput, static_cast<int>(size));

    std::vector<float> output(vecLen);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), devInput, size, cudaMemcpyDeviceToHost);

    // float * outPtr;
    // cudaMallocHost(&outPtr, size);

    // cudaDeviceSynchronize();
    // cudaMemcpyAsync(outPtr, devInput, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(outPtr, devInput, size, cudaMemcpyDeviceToHost);

    cudaFree(devInput);

    // std::vector<float> output(outPtr, outPtr + vecLen);
    // cudaFreeHost(outPtr);
    return output;
}