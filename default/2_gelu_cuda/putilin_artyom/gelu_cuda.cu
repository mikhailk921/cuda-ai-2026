#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>


__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int N)
{
    constexpr float s_Const1 = 1.595769121605731; // 2*sqrt(2/pi)
    constexpr float s_Const2 = 0.044715f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        float tanhArg = s_Const1 * (x[i] + s_Const2*x[i]*x[i]*x[i]);
        float expVal  = __expf(tanhArg);
        float tanhVal = (expVal - 1.0f)/(expVal + 1.0f);

        y[i] = 0.5f * x[i] * (1.0f + tanhVal);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    size_t N = static_cast<int>(input.size());
    std::vector<float> geluVal(N);

    size_t threadsPerBlock = 256;
    size_t memSize = N * sizeof(float);
    size_t blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    const float* in_ptr = input.data();
    float* out_ptr = geluVal.data();

    float* dInput = nullptr;
    float* dOutput = nullptr;

    cudaMalloc(&dInput, memSize);
    cudaMalloc(&dOutput, memSize);

    cudaMemcpy(dInput, in_ptr, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, out_ptr, memSize, cudaMemcpyHostToDevice);

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(dInput, dOutput, N);

    cudaDeviceSynchronize();

    cudaMemcpy(out_ptr, dOutput, memSize, cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);

    return geluVal;
}
