#include "naive_gemm_cuda.h"

#include <cuda/cmath>

__global__ void CUDAGemmKernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n && j < n) {
        for (int k = 0; k < n; ++k) {
            c[j + i * n] += a[k + i * n] * b[j + k * n];
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int num_elems = a.size();
    const int size = num_elems * sizeof(float);

    float* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, 3 * size);
    float* a_dev = dev_ptr;
    float* b_dev = dev_ptr + num_elems;
    float* c_dev = dev_ptr + 2 * num_elems;

    cudaMemcpy(a_dev, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(c_dev, 0, size);

    constexpr int block_size = 16;
    int num_blocks = cuda::ceil_div(n, block_size);
    CUDAGemmKernel<<<{num_blocks, num_blocks}, {block_size, block_size}>>>(a_dev, b_dev, c_dev, n);
    cudaDeviceSynchronize();

    std::vector<float> result(num_elems);
    cudaMemcpy(result.data(), c_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);

    return result;
}
