#include "softmax_cuda.h"

#include <float.h>
#include <math.h>
#include <iostream>
#include <vector>

template <int BLOCK_SIZE = 32>
__global__ void softmax_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];

    float local_max = -FLT_MAX;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input[row * D + i]);
    }

    float* red = shmem + BLOCK_SIZE;
    red[tid] = local_max;
    __syncthreads();

    __shared__ float row_max;
    if(tid == 0)
    {
        row_max = -FLT_MAX;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            row_max = fmaxf(red[i], row_max);
        }
    }

    float sum = 0.0;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        sum += expf(input[row * D + i] - row_max);
    }

    float* loc_sum = shmem + BLOCK_SIZE;
    loc_sum[tid] = sum;
    __syncthreads();

    __shared__ float sum_row;
    if(tid == 0)
    {
        sum_row = 0.0;
        for(int i = 0; i < BLOCK_SIZE; ++i)
        {
            sum_row += loc_sum[i];
        }
    }
    __syncthreads();

    for (int i = tid; i < D; i += BLOCK_SIZE) {
        output[row * D + i] = expf(input[row * D + i] - row_max) / sum_row;
    }
}

void softmax(const float* d_input, float* d_output, int N, int D, cudaStream_t stream = 0)
{
    const int blockSize = 32;
    int sharedMemSize = (D + blockSize) * sizeof(float);

    dim3 grid(N);
    dim3 block(blockSize);

    softmax_kernel<blockSize><<<grid, block, sharedMemSize, stream>>>(d_input, d_output, N, D);
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    const int N = row_count;
    const int D = static_cast<int>(input.size()) / row_count;

    std::vector<float> h_output(N*D);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * D * sizeof(float));
    cudaMalloc(&d_out, N * D * sizeof(float));
    cudaMemcpy(d_in, input.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    softmax(d_in, d_out, N, D);

    cudaMemcpy(h_output.data(), d_out, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return h_output;
}

// int main()
// {
//     int N = 1024;  // batch size
//     int D = 1024;   // features per sample

//     std::vector<float> h_input(N*D, 0.0), h_output(N*D, 0.0);

//     for(int i = 0; i < N; i++)
//         for(int j = 0; j < N; j++)
//         {
//             h_input[i*N+j] = i+j+2;
//              //std::cout << i*N+j << ", "<< h_input[i*N+j] << std::endl;
//         }

//     h_output = SoftmaxCUDA(h_input, N);

//     for(int i = 0; i < N; i++)
//     {
//         for(int j = 0; j < N; j++)
//         {
//             std::cout << h_output[i*N+j] << std::endl;
//         }
//     }

//     return 0;
// }
