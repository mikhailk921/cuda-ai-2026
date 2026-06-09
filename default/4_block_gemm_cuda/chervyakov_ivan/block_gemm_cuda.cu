#include "block_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void blockGemmKernelExtShMem(const float *a, const float *b, float *c, int n, int blockSize)
{
    extern __shared__ float shared_mem[];
    float* As = shared_mem; 
    float* Bs = &shared_mem[blockSize * blockSize];

    int irow = blockIdx.y * blockSize + threadIdx.y;
    int icol = blockIdx.x * blockSize + threadIdx.x;

    float zero = 0.0f;
    float sum = 0.0f;
    int numTiles = (n + blockSize - 1) / blockSize;

    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * blockSize + threadIdx.x;
        int bRow = t * blockSize + threadIdx.y;

        As[threadIdx.y * blockSize + threadIdx.x] = (irow < n && aCol < n) ? a[irow * n + aCol] : zero;
        Bs[threadIdx.y * blockSize + threadIdx.x] = (bRow < n && icol < n) ? b[bRow * n + icol] : zero;

        __syncthreads();

        for (int k = 0; k < blockSize; ++k)
        {
            sum += As[threadIdx.y * blockSize + k] * Bs[k * blockSize + threadIdx.x];
        }

        __syncthreads();
    }

    if (irow < n && icol < n)
    {
        c[irow * n + icol] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    constexpr int blockSize = 16;
    size_t sharedMemBytes = 2 * blockSize * blockSize * sizeof(float);


    size_t N = n * n;
    size_t size = N * sizeof(float);
    std::vector<float> c(N);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    float *host_a = const_cast<float *>(a.data());
    float *host_b = const_cast<float *>(b.data());
    float *host_c = const_cast<float *>(c.data());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_a)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_b)), size, cudaHostRegisterDefault);
    cudaHostRegister(const_cast<void *>(static_cast<const void *>(host_c)), size, cudaHostRegisterDefault);

    cudaMemcpyAsync(dev_a, host_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, host_b, size, cudaMemcpyHostToDevice, stream);

    dim3 threads(blockSize, blockSize);
    int blocksNum = cuda::ceil_div(n, blockSize);
    dim3 blocks(blocksNum, blocksNum);
    blockGemmKernelExtShMem<<<blocks, threads, sharedMemBytes, stream>>>(dev_a, dev_b, dev_c, n, blockSize);

    cudaMemcpyAsync(host_c, dev_c, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}