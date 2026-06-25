#include <algorithm>
#include <cuda/cmath>
#include "softmax_cuda.h"

constexpr int BLOCK_SIZE = 32;

__global__ void SoftmaxCUDAImpl(const float* input, float* output, int row_size) {
    __shared__ float sdata[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockLim = blockDim.x;

    float localMax = -INFINITY;

    const float* rowIn = input + row * row_size;
    float* rowOut = output + row * row_size;

    for (int i = tid; i < row_size; i += blockLim) {
        localMax = cuda::std::fmax(localMax, rowIn[i]);
    }
    sdata[tid] = localMax;
    __syncthreads();

    for (int s = blockLim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = cuda::std::fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float rowMax = sdata[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int i = tid; i < row_size; i += blockLim) {
        float exp_val = cuda::std::expf(rowIn[i] - rowMax);
        rowOut[i] = exp_val;
        localSum += exp_val;
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockLim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float rowSum = sdata[0];

    for (int i = tid; i < row_size; i += blockLim) {
        rowOut[i] /= rowSum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int size = input.size();
    int rowSize = size / row_count;
    int bytesSize = size * sizeof(float);

    float* gpuBufferIn = nullptr;
    float* gpuBufferOut = nullptr;

    cudaMalloc(&gpuBufferIn, bytesSize);
    cudaMalloc(&gpuBufferOut, bytesSize);

    cudaMemcpy(gpuBufferIn, input.data(), bytesSize, cudaMemcpyHostToDevice);

    SoftmaxCUDAImpl<<<row_count, BLOCK_SIZE>>>(gpuBufferIn, gpuBufferOut, rowSize);

    std::vector<float> output(size);
    cudaMemcpy(output.data(), gpuBufferOut, bytesSize, cudaMemcpyDeviceToHost);

    cudaFree(gpuBufferIn);
    cudaFree(gpuBufferOut);

    return output;
}