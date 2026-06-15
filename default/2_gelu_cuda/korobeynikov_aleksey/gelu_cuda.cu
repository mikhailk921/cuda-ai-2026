#include "gelu_cuda.h"
#include <cuda_runtime.h>


constexpr float SQRT_CONSTANT = 0.7978845608028f;

__device__  __forceinline__ float tanh_with_exp(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));;
}

__global__ void gelu_kernel(const float* input, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float x = input[i];
        float tanh_val = tanh_with_exp(SQRT_CONSTANT * (x + 0.044715f * x*x*x));
        result[i] = 0.5f * x * (1.0f + tanh_val);
    }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();

    float* gpu_result = nullptr;
    float* gpu_input = nullptr;
    cudaMalloc(&gpu_result, n * sizeof(float));
    cudaMalloc(&gpu_input, n * sizeof(float));

    cudaMemcpy(gpu_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    gelu_kernel<<<blocks, threads_per_block>>>(gpu_input, gpu_result, n);


    std::vector<float> cpu_result(n);
    cudaMemcpy(cpu_result.data(), gpu_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_result);

    return cpu_result;
}
