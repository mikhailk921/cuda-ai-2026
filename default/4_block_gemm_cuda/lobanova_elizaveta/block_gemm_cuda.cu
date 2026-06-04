#include "block_gemm_cuda.h"

#include <cuda/cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>

#define BLOCK_SIZE 16

__global__ void BlockGemm(const float *a, const float *b, float *c, int n) {
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = threadIdx.y + blockIdx.y * blockDim.y;
    int globalCol = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    __shared__ float blockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE * BLOCK_SIZE];
    
    for (int block = 0; block < gridDim.x; ++block) {
        blockA[localRow * BLOCK_SIZE + localCol] = a[globalRow * n + block * BLOCK_SIZE + localCol];
        blockB[localRow * BLOCK_SIZE + localCol] = b[(block * BLOCK_SIZE + localRow) * n + globalCol];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[localRow * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + localCol];
        }
        __syncthreads();
    }
    c[globalRow * n + globalCol] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const float* aData = a.data();
    const float* bData = b.data();
    const int dataSize = n * n;

    float* devA = nullptr;
    cudaMalloc(&devA, dataSize * sizeof(float));
    float* devB = nullptr;
    cudaMalloc(&devB, dataSize * sizeof(float));
    float* devC = nullptr;
    cudaMalloc(&devC, dataSize * sizeof(float));

    cudaMemcpy(devA, aData, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, bData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockCount(n / BLOCK_SIZE, n / BLOCK_SIZE);
    BlockGemm<<<blockCount, threadsPerBlock>>>(devA, devB, devC, n);

    std::vector<float> c(dataSize);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), devC, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return c;
}

static std::vector<float> NaiveGemmCUDARef(const std::vector<float>& a,
                                           const std::vector<float>& b,
                                           int n) {
    std::vector<float> c(a.size(), 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[i* n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    return c;
}

#if 0
int main() {
    constexpr size_t n = 1024;
    constexpr float minVal = 0.0f;
    constexpr float maxVal = 1.0f;

    std::vector<float> a(n * n);
    std::generate(a.begin(), a.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    std::vector<float> b(n * n);
    std::generate(b.begin(), b.end(), [](){
        return minVal + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (maxVal - minVal);
    });

    auto cRef = NaiveGemmCUDARef(a, b, n);
    auto c = BlockGemmCUDA(a, b, n);
    float error = 0.0f;
    for (size_t i = 0; i < n * n; ++i) {
        error = std::max(std::fabs(c[i] - cRef[i]), error);
    }
    std::cout << "Absolute max error: " << error << std::endl;

    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        BlockGemmCUDA(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time << " seconds" << std::endl;
}
#endif