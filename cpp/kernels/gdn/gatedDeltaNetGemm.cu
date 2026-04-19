/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gatedDeltaNetGemm.h"

#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace plugins
{

//! Simple GEMM kernel: C = A @ B.T where C[M,N], A[M,K], B[N,K] (B is transposed)
//! Each thread computes one output element
__global__ void gatedDeltaGemmKernel(
    __half* C, __half const* A, __half const* B, int32_t M, int32_t N, int32_t K)
{
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int32_t k = 0; k < K; ++k)
        {
            sum += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

//! Apply sigmoid activation in-place: sigmoid(x) = 1 / (1 + exp(-x))
__global__ void sigmoidKernel(__half* data, int32_t numElements)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

//! Apply silu activation in-place: silu(x) = x / (1 + exp(-x))
__global__ void siluKernel(__half* data, int32_t numElements)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

//! Transpose: [B, M, N] -> [B, N, M]
__global__ void transposeKernel(
    __half* dst, __half const* src, int32_t B, int32_t M, int32_t N)
{
    int32_t b = blockIdx.x;
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < B && i < M && j < N)
    {
        dst[b * N * M + j * M + i] = src[b * M * N + i * N + j];
    }
}

void gatedDeltaGemm(
    __half* C, __half const* A, __half const* B, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    gatedDeltaGemmKernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);
}

void sigmoidActivation(__half* data, int32_t numElements, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((numElements + 255) / 256);
    sigmoidKernel<<<grid, block, 0, stream>>>(data, numElements);
}

void siluActivation(__half* data, int32_t numElements, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((numElements + 255) / 256);
    siluKernel<<<grid, block, 0, stream>>>(data, numElements);
}

void transposeTensor(
    __half* dst, __half const* src, int32_t B, int32_t M, int32_t N, cudaStream_t stream)
{
    dim3 block(8, 8, 4);
    dim3 grid(B, (M + 7) / 8, (N + 7) / 8);
    transposeKernel<<<grid, block, 0, stream>>>(dst, src, B, M, N);
}

//! RMSNorm kernel with gating: output = z * (x * weight * bias / (||x|| + eps))
//! x: [batch, seq, num_heads, head_dim]
//! z: [batch, seq, num_heads, head_dim] (gating factor, silu already applied)
//! weight: [head_dim], bias: [head_dim]
//! output: [batch, seq, num_heads, head_dim]
__global__ void rmsNormGatingKernel(
    __half const* __restrict__ x,
    __half const* __restrict__ z,
    __half const* __restrict__ weight,
    __half const* __restrict__ bias,
    __half* __restrict__ output,
    int32_t batch, int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    int32_t b = blockIdx.x / (seqLen * numHeads);
    int32_t s = (blockIdx.x / numHeads) % seqLen;
    int32_t h = blockIdx.x % numHeads;

    int32_t headOffset = (b * seqLen * numHeads + s * numHeads + h) * headDim;

    // Shared memory for sum of squares reduction
    extern __shared__ float shared[];
    float* sumSquares = shared;
    float* rms = &shared[blockDim.x];

    // Each thread computes sum of squares for its portion of headDim
    float threadSum = 0.0f;
    for (int32_t i = threadIdx.x; i < headDim; i += blockDim.x)
    {
        float val = __half2float(x[headOffset + i]);
        threadSum += val * val;
    }
    sumSquares[threadIdx.x] = threadSum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int32_t s2 = blockDim.x / 2; s2 > 0; s2 /= 2)
    {
        if (threadIdx.x < s2)
        {
            sumSquares[threadIdx.x] += sumSquares[threadIdx.x + s2];
        }
        __syncthreads();
    }

    // Thread 0 computes RMS and stores it
    if (threadIdx.x == 0)
    {
        rms[0] = rsqrtf(sumSquares[0] / headDim + 1e-6f);
    }
    __syncthreads();

    float rmsVal = rms[0];

    // Each thread computes its portion of the output
    for (int32_t i = threadIdx.x; i < headDim; i += blockDim.x)
    {
        float val = __half2float(x[headOffset + i]) * rmsVal;
        float w = __half2float(weight[i]);
        float b = __half2float(bias[i]);
        float normVal = val * w * b;
        float zVal = __half2float(z[headOffset + i]);
        output[headOffset + i] = __float2half(zVal * normVal);
    }
}

void rmsNormGating(
    __half const* x, __half const* z, __half const* weight, __half const* bias,
    __half* output, int32_t batch, int32_t seqLen, int32_t numHeads, int32_t headDim, cudaStream_t stream)
{
    int32_t gridDim = batch * seqLen * numHeads;
    int32_t blockDim = 256; // threads per head
    size_t sharedMemSize = 2 * blockDim * sizeof(float); // sumSquares + rms
    rmsNormGatingKernel<<<gridDim, blockDim, sharedMemSize, stream>>>(x, z, weight, bias, output, batch, seqLen, numHeads, headDim);
}

} // namespace plugins
} // namespace trt_edgellm
