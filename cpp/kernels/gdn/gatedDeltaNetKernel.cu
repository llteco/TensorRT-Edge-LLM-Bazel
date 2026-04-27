/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gatedDeltaNetKernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>

namespace trt_edgellm
{
namespace kernels
{

namespace
{
constexpr float EPS = 1e-6f;

__device__ float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

__device__ half silu_h(half x)
{
    return __float2half(silu_f(__half2float(x)));
}

// ---------------------------------------------------------------------------
// Float-to-Half cast helper (internal kernel)
// ---------------------------------------------------------------------------

__global__ void castFloatToHalfKernel(float const* in, half* out, int32_t n)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = __float2half(in[idx]);
    }
}

} // anonymous namespace

void castFloatToHalf(void const* in, void* out, int32_t numel, cudaStream_t stream)
{
    int32_t blockSize = 256;
    int32_t gridSize = (numel + blockSize - 1) / blockSize;
    castFloatToHalfKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<float const*>(in), static_cast<half*>(out), numel);
}

namespace
{

// ---------------------------------------------------------------------------
// L2 Norm
// ---------------------------------------------------------------------------

template <typename T, int32_t BLOCK_SIZE>
__global__ void l2NormKernel(
    T const* __restrict__ input,
    T* __restrict__ output,
    int32_t batchSeq,
    int32_t numHeads,
    int32_t headDim)
{
    __shared__ float sharedNorm;

    int32_t bs = blockIdx.x / numHeads;
    int32_t h = blockIdx.x % numHeads;
    int32_t tid = threadIdx.x;

    if (bs >= batchSeq || h >= numHeads)
        return;

    int32_t offset = (bs * numHeads + h) * headDim;

    float sumSquares = 0.0f;
    for (int32_t i = tid; i < headDim; i += BLOCK_SIZE)
    {
        float val = static_cast<float>(input[offset + i]);
        sumSquares += val * val;
    }

    for (int32_t mask = BLOCK_SIZE / 2; mask > 0; mask /= 2)
    {
        sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, mask);
    }

    if (tid == 0)
    {
        sharedNorm = sqrtf(sumSquares + EPS);
    }
    __syncthreads();

    float normFactor = sharedNorm;
    for (int32_t i = tid; i < headDim; i += BLOCK_SIZE)
    {
        output[offset + i] = static_cast<T>(static_cast<float>(input[offset + i]) / normFactor);
    }
}

// ---------------------------------------------------------------------------
// Scale QK by 1/sqrt(head_dim)
// ---------------------------------------------------------------------------


__global__ void scaleQKByHeadDimKernel(
    half* q,
    int32_t batchSeq,
    int32_t numHeads,
    int32_t headDim)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total = batchSeq * numHeads * headDim;
    if (idx >= total)
        return;

    float scale = 1.0f / sqrtf(static_cast<float>(headDim));
    q[idx] = __float2half(__half2float(q[idx]) * scale);
}

// ---------------------------------------------------------------------------
// Recurrent Gated Delta Step (matrix state)
// ---------------------------------------------------------------------------
// State S: [head_k_dim, head_v_dim] row-major: S[i*VDim + j]
//
// S = S * exp(g)
// kv_mem[j] = sum_i S[i,j] * k[i]
// delta[j] = (v[j] - kv_mem[j]) * beta
// S[i,j] += k[i] * delta[j]
// out[j] = sum_i S[i,j] * q[i]
// ---------------------------------------------------------------------------

__global__ void recurrentGatedDeltaStepKernel(
    half const* __restrict__ query,     // [batch, numVHeads, headKDim]
    half const* __restrict__ key,       // [batch, numVHeads, headKDim]
    half const* __restrict__ value,     // [batch, numVHeads, headVDim]
    half const* __restrict__ g,         // [batch, numVHeads]
    half const* __restrict__ beta,      // [batch, numVHeads]
    half* __restrict__ state,           // [batch, numVHeads, headKDim, headVDim] (workspace)
    half* __restrict__ output,          // [batch, numVHeads, headVDim]
    int32_t batch,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim)
{
    int32_t bh = blockIdx.x;
    int32_t b = bh / numVHeads;
    int32_t h = bh % numVHeads;

    if (bh >= batch * numVHeads)
        return;

    int32_t qkOffset = bh * headKDim;
    int32_t vOffset = bh * headVDim;
    int32_t gbOffset = bh;
    int32_t stateOffset = bh * headKDim * headVDim;
    int32_t stateSize = headKDim * headVDim;

    float gVal = expf(__half2float(g[gbOffset]));
    float betaVal = __half2float(beta[gbOffset]);

    // Step 1: decay state in-place (global memory)
    for (int32_t i = threadIdx.x; i < stateSize; i += blockDim.x)
    {
        float s = __half2float(state[stateOffset + i]);
        state[stateOffset + i] = __float2half(s * gVal);
    }
    __syncthreads();

    // Step 2: each thread handles one or more v-dim columns
    for (int32_t j = threadIdx.x; j < headVDim; j += blockDim.x)
    {
        // kv_mem[j] = sum_i state[i,j] * k[i]
        float kvMem = 0.0f;
        for (int32_t i = 0; i < headKDim; ++i)
        {
            kvMem += __half2float(state[stateOffset + i * headVDim + j]) * __half2float(key[qkOffset + i]);
        }

        float vVal = __half2float(value[vOffset + j]);
        float delta = (vVal - kvMem) * betaVal;

        // Update state and compute output in one pass
        float outVal = 0.0f;
        for (int32_t i = 0; i < headKDim; ++i)
        {
            int32_t sIdx = stateOffset + i * headVDim + j;
            float sVal = __half2float(state[sIdx]);
            sVal += __half2float(key[qkOffset + i]) * delta;
            state[sIdx] = __float2half(sVal);
            outVal += sVal * __half2float(query[qkOffset + i]);
        }
        output[vOffset + j] = __float2half(outVal);
    }
}

// ---------------------------------------------------------------------------
// Serial Loop over recurrent gated delta rule (prefill)
// ---------------------------------------------------------------------------

__global__ void gatedDeltaSerialLoopKernel(
    half const* __restrict__ query,     // [batch, seqLen, numVHeads, headKDim]
    half const* __restrict__ key,       // [batch, seqLen, numVHeads, headKDim]
    half const* __restrict__ value,     // [batch, seqLen, numVHeads, headVDim]
    half const* __restrict__ g,         // [batch, seqLen, numVHeads]
    half const* __restrict__ beta,      // [batch, seqLen, numVHeads]
    half* __restrict__ state,           // [batch, numVHeads, headKDim, headVDim] (workspace)
    half* __restrict__ output,          // [batch, seqLen, numVHeads, headVDim]
    int32_t batch,
    int32_t seqLen,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim)
{
    int32_t bh = blockIdx.x;
    int32_t b = bh / numVHeads;
    int32_t h = bh % numVHeads;

    if (bh >= batch * numVHeads)
        return;

    int32_t stateSize = headKDim * headVDim;
    int32_t stateOffset = bh * stateSize;

    for (int32_t s = 0; s < seqLen; ++s)
    {
        int32_t qkOffset = ((b * seqLen + s) * numVHeads + h) * headKDim;
        int32_t vOffset = ((b * seqLen + s) * numVHeads + h) * headVDim;
        int32_t gbOffset = (b * seqLen + s) * numVHeads + h;

        float gVal = expf(__half2float(g[gbOffset]));
        float betaVal = __half2float(beta[gbOffset]);

        // Step 1: decay state in-place
        for (int32_t i = threadIdx.x; i < stateSize; i += blockDim.x)
        {
            float sv = __half2float(state[stateOffset + i]);
            state[stateOffset + i] = __float2half(sv * gVal);
        }
        __syncthreads();

        // Step 2: compute output for this timestep
        for (int32_t j = threadIdx.x; j < headVDim; j += blockDim.x)
        {
            float kvMem = 0.0f;
            for (int32_t i = 0; i < headKDim; ++i)
            {
                kvMem += __half2float(state[stateOffset + i * headVDim + j]) * __half2float(key[qkOffset + i]);
            }

            float vVal = __half2float(value[vOffset + j]);
            float delta = (vVal - kvMem) * betaVal;

            float outVal = 0.0f;
            for (int32_t i = 0; i < headKDim; ++i)
            {
                int32_t sIdx = stateOffset + i * headVDim + j;
                float sVal = __half2float(state[sIdx]);
                sVal += __half2float(key[qkOffset + i]) * delta;
                state[sIdx] = __float2half(sVal);
                outVal += sVal * __half2float(query[qkOffset + i]);
            }
            output[vOffset + j] = __float2half(outVal);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d Prefill
// ---------------------------------------------------------------------------

__global__ void causalConv1dPrefillKernel(
    half const* __restrict__ x,         // [batch, convDim, seqLen]
    half const* __restrict__ weight,    // [convDim, kernelSize]
    half const* __restrict__ bias,      // [convDim]
    half* __restrict__ output,          // [batch, convDim, seqLen]
    half* __restrict__ convStateOut,    // [batch, convDim, kernelSize]
    int32_t batch,
    int32_t convDim,
    int32_t seqLen,
    int32_t kernelSize,
    int32_t activation)
{
    int32_t b = blockIdx.x;
    int32_t d = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= batch || d >= convDim)
        return;

    int32_t base = b * convDim * seqLen + d * seqLen;

    for (int32_t s = 0; s < seqLen; ++s)
    {
        float acc = (bias != nullptr) ? __half2float(bias[d]) : 0.0f;
        for (int32_t k = 0; k < kernelSize; ++k)
        {
            int32_t inputPos = s - kernelSize + 1 + k;
            if (inputPos >= 0 && inputPos < seqLen)
            {
                acc += __half2float(x[base + inputPos]) * __half2float(weight[d * kernelSize + k]);
            }
        }
        if (activation == 0)
        {
            acc = silu_f(acc);
        }
        output[base + s] = __float2half(acc);
    }

    // Capture conv state: last kernelSize inputs, zero-padded
    int32_t stateBase = b * convDim * kernelSize + d * kernelSize;
    int32_t pad = kernelSize - seqLen;
    if (pad > 0)
    {
        for (int32_t k = 0; k < pad; ++k)
        {
            convStateOut[stateBase + k] = __float2half(0.0f);
        }
    }
    int32_t copyStart = (pad > 0) ? pad : 0;
    int32_t srcStart = (pad > 0) ? 0 : seqLen - kernelSize;
    for (int32_t k = copyStart; k < kernelSize; ++k)
    {
        convStateOut[stateBase + k] = x[base + srcStart + (k - copyStart)];
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d Decode
// ---------------------------------------------------------------------------

__global__ void causalConv1dDecodeKernel(
    half const* __restrict__ x,         // [batch, convDim, 1]
    half const* __restrict__ convState, // [batch, convDim, kernelSize]
    half const* __restrict__ weight,    // [convDim, kernelSize]
    half const* __restrict__ bias,      // [convDim]
    half* __restrict__ output,          // [batch, convDim, 1]
    half* __restrict__ convStateOut,    // [batch, convDim, kernelSize]
    int32_t batch,
    int32_t convDim,
    int32_t kernelSize,
    int32_t activation)
{
    int32_t b = blockIdx.x;
    int32_t d = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= batch || d >= convDim)
        return;

    int32_t stateBase = b * convDim * kernelSize + d * kernelSize;

    // Shift and insert first (matching Qwen3.5 torch_causal_conv1d_update)
    for (int32_t k = 0; k < kernelSize - 1; ++k)
    {
        convStateOut[stateBase + k] = convState[stateBase + k + 1];
    }
    convStateOut[stateBase + kernelSize - 1] = x[b * convDim + d];

    // Compute output using the updated state
    float acc = (bias != nullptr) ? __half2float(bias[d]) : 0.0f;
    for (int32_t k = 0; k < kernelSize; ++k)
    {
        acc += __half2float(convStateOut[stateBase + k]) * __half2float(weight[d * kernelSize + k]);
    }
    if (activation == 0)
    {
        acc = silu_f(acc);
    }
    output[b * convDim + d] = __float2half(acc);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void computeGatedDeltaQKNorm(
    void const* q,
    void const* k,
    void* q_normed,
    void* k_normed,
    int32_t batchSeq,
    int32_t numHeads,
    int32_t headDim,
    cudaStream_t cudaStream)
{
    dim3 block(32);
    dim3 grid(batchSeq * numHeads);
    l2NormKernel<half, 32><<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(q),
        static_cast<half*>(q_normed),
        batchSeq, numHeads, headDim);
    l2NormKernel<half, 32><<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(k),
        static_cast<half*>(k_normed),
        batchSeq, numHeads, headDim);
}

void scaleQK(
    void* q,
    int32_t batchSeq,
    int32_t numHeads,
    int32_t headDim,
    cudaStream_t cudaStream)
{
    int32_t total = batchSeq * numHeads * headDim;
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total + 255) / 256);
    void* args[] = {&q, &batchSeq, &numHeads, &headDim};
    cudaLaunchKernel((const void*) scaleQKByHeadDimKernel, numBlocks, threadsPerBlock, args, 0, cudaStream);
}

void recurrentGatedDeltaStep(
    void const* query,
    void const* key,
    void const* value,
    void const* g,
    void const* beta,
    void const* initialState,
    void* output,
    void* finalState,
    int32_t batch,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    cudaStream_t cudaStream)
{
    // Initialize finalState as workspace: copy initialState or zero-fill
    int32_t stateElems = batch * numVHeads * headKDim * headVDim;
    int32_t stateBytes = stateElems * sizeof(half);
    if (initialState != nullptr)
    {
        cudaMemcpyAsync(finalState, initialState, stateBytes, cudaMemcpyDeviceToDevice, cudaStream);
    }
    else
    {
        cudaMemsetAsync(finalState, 0, stateBytes, cudaStream);
    }

    dim3 grid(batch * numVHeads);
    dim3 block(128);
    recurrentGatedDeltaStepKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(query),
        static_cast<half const*>(key),
        static_cast<half const*>(value),
        static_cast<half const*>(g),
        static_cast<half const*>(beta),
        static_cast<half*>(finalState),
        static_cast<half*>(output),
        batch, numVHeads, headVDim, headKDim);
}

void gatedDeltaNetSerialLoop(
    void const* query,
    void const* key,
    void const* value,
    void const* g,
    void const* beta,
    void const* initialState,
    void* output,
    void* finalState,
    int32_t batch,
    int32_t seqLen,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    cudaStream_t cudaStream)
{
    // Initialize finalState as workspace: copy initialState or zero-fill
    int32_t stateElems = batch * numVHeads * headKDim * headVDim;
    int32_t stateBytes = stateElems * sizeof(half);
    if (initialState != nullptr)
    {
        cudaMemcpyAsync(finalState, initialState, stateBytes, cudaMemcpyDeviceToDevice, cudaStream);
    }
    else
    {
        cudaMemsetAsync(finalState, 0, stateBytes, cudaStream);
    }

    dim3 grid(batch * numVHeads);
    dim3 block(128);
    gatedDeltaSerialLoopKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(query),
        static_cast<half const*>(key),
        static_cast<half const*>(value),
        static_cast<half const*>(g),
        static_cast<half const*>(beta),
        static_cast<half*>(finalState),
        static_cast<half*>(output),
        batch, seqLen, numVHeads, headVDim, headKDim);
}

void causalConv1dPrefill(
    void const* x,
    void const* weight,
    void const* bias,
    void* output,
    void* convStateOut,
    int32_t batch,
    int32_t convDim,
    int32_t seqLen,
    int32_t convKernelSize,
    int32_t activation,
    cudaStream_t cudaStream)
{
    dim3 block(128);
    dim3 grid(batch, (convDim + 127) / 128);
    causalConv1dPrefillKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(x),
        static_cast<half const*>(weight),
        static_cast<half const*>(bias),
        static_cast<half*>(output),
        static_cast<half*>(convStateOut),
        batch, convDim, seqLen, convKernelSize, activation);
}

void causalConv1dDecode(
    void const* x,
    void const* convState,
    void const* weight,
    void const* bias,
    void* output,
    void* convStateOut,
    int32_t batch,
    int32_t convDim,
    int32_t convKernelSize,
    int32_t activation,
    cudaStream_t cudaStream)
{
    dim3 block(128);
    dim3 grid(batch, (convDim + 127) / 128);
    causalConv1dDecodeKernel<<<grid, block, 0, cudaStream>>>(
        static_cast<half const*>(x),
        static_cast<half const*>(convState),
        static_cast<half const*>(weight),
        static_cast<half const*>(bias),
        static_cast<half*>(output),
        static_cast<half*>(convStateOut),
        batch, convDim, convKernelSize, activation);
}

// Legacy compatibility wrapper
void causalConv1dUpdate(
    void const* mixedQKV,
    void const* convState,
    void const* convWeight,
    void const* convBias,
    void* updatedConvState,
    void* output,
    int32_t batch,
    int32_t convDim,
    int32_t seqLen,
    int32_t convKernelSize,
    int32_t activation,
    cudaStream_t stream)
{
    if (seqLen == 1)
    {
        causalConv1dDecode(
            mixedQKV, convState, convWeight, convBias,
            output, updatedConvState, batch, convDim, convKernelSize, activation, stream);
    }
    else
    {
        causalConv1dPrefill(
            mixedQKV, convWeight, convBias,
            output, updatedConvState, batch, convDim, seqLen, convKernelSize, activation, stream);
    }
}

} // namespace kernels
} // namespace trt_edgellm
