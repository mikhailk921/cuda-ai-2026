#include "gelu_cuda.h"

#include <cuda/cmath>

__global__ void CUDAGeluKernel(float* dev_ptr, int num_elems) {
    constexpr float innerCoeff = M_2_SQRTPI * M_SQRT1_2;
    constexpr float cubeCoeff = innerCoeff * 0.044715;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elems) {
        float x = dev_ptr[i];
        float cube = x * x * x;
        dev_ptr[i] = 0.5 * x * (1 + cuda::std::tanh(innerCoeff * x + cubeCoeff * cube));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int input_num_elems = input.size();
    const int input_size = input_num_elems * sizeof(float);
    const float* input_ptr = input.data();
    float* dev_ptr = nullptr;

    cudaMalloc(&dev_ptr, input_size);
    cudaMemcpy(dev_ptr, input_ptr, input_size, cudaMemcpyHostToDevice);

    constexpr int block_size = 256;
    const int num_blocks = cuda::ceil_div(input_num_elems, block_size);

    CUDAGeluKernel<<<num_blocks, block_size>>>(dev_ptr, input_num_elems);
    cudaDeviceSynchronize();

    std::vector<float> result(input_num_elems);
    cudaMemcpy(result.data(), dev_ptr, input_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);
    
    return result;
}
