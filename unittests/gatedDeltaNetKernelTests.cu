/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/gdn/gatedDeltaNetKernel.cuh"
#include "testUtils.h"

using namespace trt_edgellm::kernels;
using namespace trt_edgellm;
using namespace nvinfer1;

// =============================================================================
// CPU Reference Implementations
// =============================================================================

namespace
{

void computeGatedDeltaQKNormRef(
    std::vector<half> const& q, std::vector<half> const& k, std::vector<half>& qNormed, std::vector<half>& kNormed,
    int32_t batchSeq, int32_t numHeads, int32_t headDim)
{
    qNormed.resize(q.size());
    kNormed.resize(k.size());

    for (int32_t bs = 0; bs < batchSeq; ++bs)
    {
        for (int32_t h = 0; h < numHeads; ++h)
        {
            float sumSqQ = 0.0f;
            float sumSqK = 0.0f;
            for (int32_t d = 0; d < headDim; ++d)
            {
                float qv = __half2float(q[(bs * numHeads + h) * headDim + d]);
                float kv = __half2float(k[(bs * numHeads + h) * headDim + d]);
                sumSqQ += qv * qv;
                sumSqK += kv * kv;
            }
            float normQ = std::sqrt(sumSqQ + 1e-6f);
            float normK = std::sqrt(sumSqK + 1e-6f);

            for (int32_t d = 0; d < headDim; ++d)
            {
                qNormed[(bs * numHeads + h) * headDim + d]
                    = __float2half(__half2float(q[(bs * numHeads + h) * headDim + d]) / normQ);
                kNormed[(bs * numHeads + h) * headDim + d]
                    = __float2half(__half2float(k[(bs * numHeads + h) * headDim + d]) / normK);
            }
        }
    }
}

void scaleQKRef(std::vector<half>& q, int32_t /*batchSeq*/, int32_t /*numHeads*/, int32_t headDim)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    for (size_t i = 0; i < q.size(); ++i)
    {
        q[i] = __float2half(__half2float(q[i]) * scale);
    }
}

//! Recurrent gated delta rule reference (matrix state)
//! S: [batch, numVHeads, headKDim, headVDim]
void recurrentGatedDeltaStepRef(std::vector<half> const& query, std::vector<half> const& key,
    std::vector<half> const& value, std::vector<half> const& g, std::vector<half> const& beta,
    std::vector<half> const* initialState, std::vector<half>& output, std::vector<half>& finalState,
    int32_t batch, int32_t numVHeads, int32_t headVDim, int32_t headKDim)
{
    int32_t stateSize = headKDim * headVDim;
    output.resize(batch * numVHeads * headVDim);
    finalState.resize(batch * numVHeads * stateSize);

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < numVHeads; ++h)
        {
            int32_t bh = b * numVHeads + h;
            int32_t qkOffset = bh * headKDim;
            int32_t vOffset = bh * headVDim;
            int32_t gbOffset = bh;

            float gVal = std::exp(__half2float(g[gbOffset]));
            float betaVal = __half2float(beta[gbOffset]);

            std::vector<half> state(stateSize);
            if (initialState != nullptr)
            {
                memcpy(state.data(), initialState->data() + bh * stateSize, stateSize * sizeof(half));
            }
            else
            {
                memset(state.data(), 0, stateSize * sizeof(half));
            }

            // S = S * gVal (in half precision)
            for (int32_t i = 0; i < stateSize; ++i)
            {
                state[i] = __float2half(__half2float(state[i]) * gVal);
            }

            for (int32_t j = 0; j < headVDim; ++j)
            {
                float kvMem = 0.0f;
                for (int32_t i = 0; i < headKDim; ++i)
                {
                    kvMem += __half2float(state[i * headVDim + j]) * __half2float(key[qkOffset + i]);
                }
                float vVal = __half2float(value[vOffset + j]);
                float delta = (vVal - kvMem) * betaVal;

                float outVal = 0.0f;
                for (int32_t i = 0; i < headKDim; ++i)
                {
                    float sVal = __half2float(state[i * headVDim + j]);
                    sVal += __half2float(key[qkOffset + i]) * delta;
                    state[i * headVDim + j] = __float2half(sVal);
                    outVal += sVal * __half2float(query[qkOffset + i]);
                }
                output[vOffset + j] = __float2half(outVal);
            }

            memcpy(finalState.data() + bh * stateSize, state.data(), stateSize * sizeof(half));
        }
    }
}

//! Serial loop reference for prefill
void gatedDeltaNetSerialLoopRef(std::vector<half> const& query, std::vector<half> const& key,
    std::vector<half> const& value, std::vector<half> const& g, std::vector<half> const& beta,
    std::vector<half> const* initialState, std::vector<half>& output, std::vector<half>& finalState,
    int32_t batch, int32_t seqLen, int32_t numVHeads, int32_t headVDim, int32_t headKDim)
{
    int32_t stateSize = headKDim * headVDim;
    output.resize(batch * seqLen * numVHeads * headVDim);
    finalState.resize(batch * numVHeads * stateSize);

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t h = 0; h < numVHeads; ++h)
        {
            int32_t bh = b * numVHeads + h;
            int32_t stateOffset = bh * stateSize;

            std::vector<half> state(stateSize);
            if (initialState != nullptr)
            {
                memcpy(state.data(), initialState->data() + stateOffset, stateSize * sizeof(half));
            }
            else
            {
                memset(state.data(), 0, stateSize * sizeof(half));
            }

            for (int32_t s = 0; s < seqLen; ++s)
            {
                int32_t qkOffset = ((b * seqLen + s) * numVHeads + h) * headKDim;
                int32_t vOffset = ((b * seqLen + s) * numVHeads + h) * headVDim;
                int32_t gbOffset = (b * seqLen + s) * numVHeads + h;

                float gVal = std::exp(__half2float(g[gbOffset]));
                float betaVal = __half2float(beta[gbOffset]);

                for (int32_t i = 0; i < stateSize; ++i)
                {
                    state[i] = __float2half(__half2float(state[i]) * gVal);
                }

                for (int32_t j = 0; j < headVDim; ++j)
                {
                    float kvMem = 0.0f;
                    for (int32_t i = 0; i < headKDim; ++i)
                    {
                        kvMem += __half2float(state[i * headVDim + j]) * __half2float(key[qkOffset + i]);
                    }
                    float vVal = __half2float(value[vOffset + j]);
                    float delta = (vVal - kvMem) * betaVal;

                    float outVal = 0.0f;
                    for (int32_t i = 0; i < headKDim; ++i)
                    {
                        float sVal = __half2float(state[i * headVDim + j]);
                        sVal += __half2float(key[qkOffset + i]) * delta;
                        state[i * headVDim + j] = __float2half(sVal);
                        outVal += sVal * __half2float(query[qkOffset + i]);
                    }
                    output[vOffset + j] = __float2half(outVal);
                }
            }

            memcpy(finalState.data() + stateOffset, state.data(), stateSize * sizeof(half));
        }
    }
}

//! Causal conv1d prefill reference
void causalConv1dPrefillRef(std::vector<half> const& x, std::vector<half> const& weight,
    std::vector<half> const& bias, std::vector<half>& output, std::vector<half>& convStateOut,
    int32_t batch, int32_t convDim, int32_t seqLen, int32_t kernelSize, int32_t activation)
{
    output.resize(batch * convDim * seqLen);
    convStateOut.resize(batch * convDim * kernelSize);
    std::fill(convStateOut.begin(), convStateOut.end(), __float2half(0.0f));

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < convDim; ++d)
        {
            int32_t base = b * convDim * seqLen + d * seqLen;
            for (int32_t s = 0; s < seqLen; ++s)
            {
                float acc = __half2float(bias[d]);
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
                    acc = acc / (1.0f + std::exp(-acc));
                }
                output[base + s] = __float2half(acc);
            }

            // Capture state
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
    }
}

//! Causal conv1d decode reference (matches Qwen3.5 torch_causal_conv1d_update)
void causalConv1dDecodeRef(std::vector<half> const& x, std::vector<half> const& convState,
    std::vector<half> const& weight, std::vector<half> const& bias, std::vector<half>& output,
    std::vector<half>& convStateOut, int32_t batch, int32_t convDim, int32_t kernelSize, int32_t activation)
{
    output.resize(batch * convDim);
    convStateOut.resize(batch * convDim * kernelSize);

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < convDim; ++d)
        {
            int32_t stateBase = b * convDim * kernelSize + d * kernelSize;

            // Shift and insert first
            for (int32_t k = 0; k < kernelSize - 1; ++k)
            {
                convStateOut[stateBase + k] = convState[stateBase + k + 1];
            }
            convStateOut[stateBase + kernelSize - 1] = x[b * convDim + d];

            // Compute output using updated state
            float acc = __half2float(bias[d]);
            for (int32_t k = 0; k < kernelSize; ++k)
            {
                acc += __half2float(convStateOut[stateBase + k]) * __half2float(weight[d * kernelSize + k]);
            }
            if (activation == 0)
            {
                acc = acc / (1.0f + std::exp(-acc));
            }
            output[b * convDim + d] = __float2half(acc);
        }
    }
}

// =============================================================================
// Test Drivers
// =============================================================================

void runComputeQKNormTest(int32_t batchSeq, int32_t numHeads, int32_t headDim)
{
    std::vector<half> qHost(batchSeq * numHeads * headDim);
    std::vector<half> kHost(batchSeq * numHeads * headDim);
    uniformFloatInitialization<half>(qHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(kHost, -1.0f, 1.0f);

    std::vector<half> qNormedRef, kNormedRef;
    computeGatedDeltaQKNormRef(qHost, kHost, qNormedRef, kNormedRef, batchSeq, numHeads, headDim);

    auto qDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto kDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto qNormedDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto kNormedDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(qDevice.rawPointer(), qHost.data(), qHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kDevice.rawPointer(), kHost.data(), kHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    computeGatedDeltaQKNorm(qDevice.rawPointer(), kDevice.rawPointer(), qNormedDevice.rawPointer(),
        kNormedDevice.rawPointer(), batchSeq, numHeads, headDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> qNormedHost(qNormedRef.size());
    std::vector<half> kNormedHost(kNormedRef.size());
    CUDA_CHECK(cudaMemcpy(qNormedHost.data(), qNormedDevice.rawPointer(), qNormedHost.size() * sizeof(half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kNormedHost.data(), kNormedDevice.rawPointer(), kNormedHost.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t qMismatches = 0;
    for (size_t i = 0; i < qNormedRef.size(); ++i)
    {
        if (!isclose(qNormedHost[i], qNormedRef[i], rtol, atol))
        {
            if (qMismatches < 5)
            {
                std::cout << "Q norm mismatch at " << i << ": got " << __half2float(qNormedHost[i])
                          << ", expected " << __half2float(qNormedRef[i]) << std::endl;
            }
            qMismatches++;
        }
    }
    EXPECT_EQ(qMismatches, 0);

    int32_t kMismatches = 0;
    for (size_t i = 0; i < kNormedRef.size(); ++i)
    {
        if (!isclose(kNormedHost[i], kNormedRef[i], rtol, atol))
        {
            if (kMismatches < 5)
            {
                std::cout << "K norm mismatch at " << i << ": got " << __half2float(kNormedHost[i])
                          << ", expected " << __half2float(kNormedRef[i]) << std::endl;
            }
            kMismatches++;
        }
    }
    EXPECT_EQ(kMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "computeGatedDeltaQKNorm: batchSeq=" << batchSeq << ", numHeads=" << numHeads << ", headDim=" << headDim
              << " PASS" << std::endl;
}

void runScaleQKTest(int32_t batchSeq, int32_t numHeads, int32_t headDim)
{
    std::vector<half> qHost(batchSeq * numHeads * headDim);
    std::vector<half> kHost(batchSeq * numHeads * headDim);
    uniformFloatInitialization<half>(qHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(kHost, -1.0f, 1.0f);

    std::vector<half> qRef = qHost;
    std::vector<half> kRef = kHost;
    scaleQKRef(qRef, batchSeq, numHeads, headDim);

    auto qDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto kDevice = rt::Tensor({batchSeq, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(qDevice.rawPointer(), qHost.data(), qHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kDevice.rawPointer(), kHost.data(), kHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    scaleQK(qDevice.rawPointer(), batchSeq, numHeads, headDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> qGpu(qRef.size());
    std::vector<half> kGpu(kRef.size());
    CUDA_CHECK(cudaMemcpy(qGpu.data(), qDevice.rawPointer(), qGpu.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kGpu.data(), kDevice.rawPointer(), kGpu.size() * sizeof(half), cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t qMismatches = 0;
    for (size_t i = 0; i < qRef.size(); ++i)
    {
        if (!isclose(qGpu[i], qRef[i], rtol, atol))
        {
            qMismatches++;
        }
    }
    EXPECT_EQ(qMismatches, 0);

    int32_t kMismatches = 0;
    for (size_t i = 0; i < kHost.size(); ++i)
    {
        if (!isclose(kGpu[i], kHost[i], rtol, atol))
        {
            kMismatches++;
        }
    }
    EXPECT_EQ(kMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "scaleQK: batchSeq=" << batchSeq << ", numHeads=" << numHeads << ", headDim=" << headDim << " PASS"
              << std::endl;
}

void runRecurrentStepTest(int32_t batch, int32_t numVHeads, int32_t headVDim, int32_t headKDim, bool hasInitialState)
{
    std::vector<half> queryHost(batch * numVHeads * headKDim);
    std::vector<half> keyHost(batch * numVHeads * headKDim);
    std::vector<half> valueHost(batch * numVHeads * headVDim);
    std::vector<half> gHost(batch * numVHeads);
    std::vector<half> betaHost(batch * numVHeads);

    uniformFloatInitialization<half>(queryHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(keyHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(valueHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(gHost, -2.0f, 2.0f);
    uniformFloatInitialization<half>(betaHost, 0.0f, 1.0f);

    int32_t stateSize = headKDim * headVDim;
    std::vector<half> initialStateHost;
    if (hasInitialState)
    {
        initialStateHost.resize(batch * numVHeads * stateSize);
        uniformFloatInitialization<half>(initialStateHost, -0.5f, 0.5f);
    }

    std::vector<half> outputRef, finalStateRef;
    recurrentGatedDeltaStepRef(queryHost, keyHost, valueHost, gHost, betaHost,
        hasInitialState ? &initialStateHost : nullptr, outputRef, finalStateRef, batch, numVHeads, headVDim, headKDim);

    auto queryDevice = rt::Tensor({batch, numVHeads, headKDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto keyDevice = rt::Tensor({batch, numVHeads, headKDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto valueDevice = rt::Tensor({batch, numVHeads, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gDevice = rt::Tensor({batch, numVHeads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto betaDevice = rt::Tensor({batch, numVHeads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, numVHeads, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto finalStateDevice = rt::Tensor({batch, numVHeads, headKDim, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(queryDevice.rawPointer(), queryHost.data(), queryHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(keyDevice.rawPointer(), keyHost.data(), keyHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(valueDevice.rawPointer(), valueHost.data(), valueHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gDevice.rawPointer(), gHost.data(), gHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(betaDevice.rawPointer(), betaHost.data(), betaHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::unique_ptr<rt::Tensor> initialStateDevice;
    void const* initialStateDevPtr = nullptr;
    if (hasInitialState)
    {
        initialStateDevice = std::make_unique<rt::Tensor>(
            rt::Tensor({batch, numVHeads, headKDim, headVDim}, rt::DeviceType::kGPU, DataType::kHALF));
        CUDA_CHECK(cudaMemcpy(initialStateDevice->rawPointer(), initialStateHost.data(), initialStateHost.size() * sizeof(half),
            cudaMemcpyHostToDevice));
        initialStateDevPtr = initialStateDevice->rawPointer();
    }

    recurrentGatedDeltaStep(queryDevice.rawPointer(), keyDevice.rawPointer(), valueDevice.rawPointer(),
        gDevice.rawPointer(), betaDevice.rawPointer(), initialStateDevPtr, outputDevice.rawPointer(),
        finalStateDevice.rawPointer(), batch, numVHeads, headVDim, headKDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> outputFromGpu(outputRef.size());
    std::vector<half> finalStateFromGpu(finalStateRef.size());
    CUDA_CHECK(cudaMemcpy(outputFromGpu.data(), outputDevice.rawPointer(), outputFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(finalStateFromGpu.data(), finalStateDevice.rawPointer(), finalStateFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t outputMismatches = 0;
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        if (!isclose(outputFromGpu[i], outputRef[i], rtol, atol))
        {
            if (outputMismatches < 5)
            {
                std::cout << "Output mismatch at " << i << ": got " << __half2float(outputFromGpu[i])
                          << ", expected " << __half2float(outputRef[i]) << std::endl;
            }
            outputMismatches++;
        }
    }
    EXPECT_EQ(outputMismatches, 0);

    int32_t stateMismatches = 0;
    for (size_t i = 0; i < finalStateRef.size(); ++i)
    {
        if (!isclose(finalStateFromGpu[i], finalStateRef[i], rtol, atol))
        {
            if (stateMismatches < 5)
            {
                std::cout << "State mismatch at " << i << ": got " << __half2float(finalStateFromGpu[i])
                          << ", expected " << __half2float(finalStateRef[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "recurrentGatedDeltaStep: batch=" << batch << ", numVHeads=" << numVHeads << ", headVDim=" << headVDim
              << ", headKDim=" << headKDim << ", hasInitialState=" << hasInitialState << " PASS" << std::endl;
}

void runSerialLoopTest(int32_t batch, int32_t seqLen, int32_t numVHeads, int32_t headVDim, int32_t headKDim,
    bool hasInitialState)
{
    std::vector<half> queryHost(batch * seqLen * numVHeads * headKDim);
    std::vector<half> keyHost(batch * seqLen * numVHeads * headKDim);
    std::vector<half> valueHost(batch * seqLen * numVHeads * headVDim);
    std::vector<half> gHost(batch * seqLen * numVHeads);
    std::vector<half> betaHost(batch * seqLen * numVHeads);

    uniformFloatInitialization<half>(queryHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(keyHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(valueHost, -1.0f, 1.0f);
    // Use negative g values so exp(g) <= 1 (decay), matching real model behavior
    uniformFloatInitialization<half>(gHost, -2.0f, 0.0f);
    uniformFloatInitialization<half>(betaHost, 0.0f, 1.0f);

    int32_t stateSize = headKDim * headVDim;
    std::vector<half> initialStateHost;
    if (hasInitialState)
    {
        initialStateHost.resize(batch * numVHeads * stateSize);
        uniformFloatInitialization<half>(initialStateHost, -0.5f, 0.5f);
    }

    std::vector<half> outputRef, finalStateRef;
    gatedDeltaNetSerialLoopRef(queryHost, keyHost, valueHost, gHost, betaHost,
        hasInitialState ? &initialStateHost : nullptr, outputRef, finalStateRef, batch, seqLen, numVHeads, headVDim,
        headKDim);

    auto queryDevice = rt::Tensor({batch, seqLen, numVHeads, headKDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto keyDevice = rt::Tensor({batch, seqLen, numVHeads, headKDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto valueDevice = rt::Tensor({batch, seqLen, numVHeads, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto gDevice = rt::Tensor({batch, seqLen, numVHeads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto betaDevice = rt::Tensor({batch, seqLen, numVHeads}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, numVHeads, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto finalStateDevice = rt::Tensor({batch, numVHeads, headKDim, headVDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(queryDevice.rawPointer(), queryHost.data(), queryHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(keyDevice.rawPointer(), keyHost.data(), keyHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(valueDevice.rawPointer(), valueHost.data(), valueHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gDevice.rawPointer(), gHost.data(), gHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(betaDevice.rawPointer(), betaHost.data(), betaHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::unique_ptr<rt::Tensor> initialStateDevice;
    void const* initialStateDevPtr = nullptr;
    if (hasInitialState)
    {
        initialStateDevice = std::make_unique<rt::Tensor>(
            rt::Tensor({batch, numVHeads, headKDim, headVDim}, rt::DeviceType::kGPU, DataType::kHALF));
        CUDA_CHECK(cudaMemcpy(initialStateDevice->rawPointer(), initialStateHost.data(), initialStateHost.size() * sizeof(half),
            cudaMemcpyHostToDevice));
        initialStateDevPtr = initialStateDevice->rawPointer();
    }

    gatedDeltaNetSerialLoop(queryDevice.rawPointer(), keyDevice.rawPointer(), valueDevice.rawPointer(),
        gDevice.rawPointer(), betaDevice.rawPointer(), initialStateDevPtr, outputDevice.rawPointer(),
        finalStateDevice.rawPointer(), batch, seqLen, numVHeads, headVDim, headKDim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> outputFromGpu(outputRef.size());
    std::vector<half> finalStateFromGpu(finalStateRef.size());
    CUDA_CHECK(cudaMemcpy(outputFromGpu.data(), outputDevice.rawPointer(), outputFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(finalStateFromGpu.data(), finalStateDevice.rawPointer(), finalStateFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t outputMismatches = 0;
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        if (!isclose(outputFromGpu[i], outputRef[i], rtol, atol))
        {
            if (outputMismatches < 5)
            {
                std::cout << "Output mismatch at " << i << ": got " << __half2float(outputFromGpu[i])
                          << ", expected " << __half2float(outputRef[i]) << std::endl;
            }
            outputMismatches++;
        }
    }
    EXPECT_EQ(outputMismatches, 0);

    int32_t stateMismatches = 0;
    for (size_t i = 0; i < finalStateRef.size(); ++i)
    {
        if (!isclose(finalStateFromGpu[i], finalStateRef[i], rtol, atol))
        {
            if (stateMismatches < 5)
            {
                std::cout << "State mismatch at " << i << ": got " << __half2float(finalStateFromGpu[i])
                          << ", expected " << __half2float(finalStateRef[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "gatedDeltaNetSerialLoop: batch=" << batch << ", seqLen=" << seqLen << ", numVHeads=" << numVHeads
              << ", headVDim=" << headVDim << ", headKDim=" << headKDim << ", hasInitialState=" << hasInitialState
              << " PASS" << std::endl;
}

void runCausalConv1dPrefillTest(int32_t batch, int32_t convDim, int32_t seqLen, int32_t kernelSize, int32_t activation)
{
    std::vector<half> xHost(batch * convDim * seqLen);
    std::vector<half> weightHost(convDim * kernelSize);
    std::vector<half> biasHost(convDim);

    uniformFloatInitialization<half>(xHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(weightHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(biasHost, -0.5f, 0.5f);

    std::vector<half> outputRef, convStateRef;
    causalConv1dPrefillRef(xHost, weightHost, biasHost, outputRef, convStateRef, batch, convDim, seqLen, kernelSize,
        activation);

    auto xDevice = rt::Tensor({batch, convDim, seqLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({convDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, convDim, seqLen}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateDevice = rt::Tensor({batch, convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(xDevice.rawPointer(), xHost.data(), xHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weightDevice.rawPointer(), weightHost.data(), weightHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(biasDevice.rawPointer(), biasHost.data(), biasHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    causalConv1dPrefill(xDevice.rawPointer(), weightDevice.rawPointer(), biasDevice.rawPointer(),
        outputDevice.rawPointer(), convStateDevice.rawPointer(), batch, convDim, seqLen, kernelSize, activation, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> outputFromGpu(outputRef.size());
    std::vector<half> convStateFromGpu(convStateRef.size());
    CUDA_CHECK(cudaMemcpy(outputFromGpu.data(), outputDevice.rawPointer(), outputFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(convStateFromGpu.data(), convStateDevice.rawPointer(), convStateFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t outMismatches = 0;
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        if (!isclose(outputFromGpu[i], outputRef[i], rtol, atol))
        {
            if (outMismatches < 5)
            {
                std::cout << "Prefill output mismatch at " << i << ": got " << __half2float(outputFromGpu[i])
                          << ", expected " << __half2float(outputRef[i]) << std::endl;
            }
            outMismatches++;
        }
    }
    EXPECT_EQ(outMismatches, 0);

    int32_t stateMismatches = 0;
    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        if (!isclose(convStateFromGpu[i], convStateRef[i], rtol, atol))
        {
            if (stateMismatches < 5)
            {
                std::cout << "Prefill state mismatch at " << i << ": got " << __half2float(convStateFromGpu[i])
                          << ", expected " << __half2float(convStateRef[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "causalConv1dPrefill: batch=" << batch << ", convDim=" << convDim << ", seqLen=" << seqLen
              << ", kernelSize=" << kernelSize << " PASS" << std::endl;
}

void runCausalConv1dDecodeTest(int32_t batch, int32_t convDim, int32_t kernelSize, int32_t activation)
{
    std::vector<half> xHost(batch * convDim);
    std::vector<half> convStateHost(batch * convDim * kernelSize);
    std::vector<half> weightHost(convDim * kernelSize);
    std::vector<half> biasHost(convDim);

    uniformFloatInitialization<half>(xHost, -1.0f, 1.0f);
    uniformFloatInitialization<half>(convStateHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(weightHost, -0.5f, 0.5f);
    uniformFloatInitialization<half>(biasHost, -0.5f, 0.5f);

    std::vector<half> outputRef, convStateRef;
    causalConv1dDecodeRef(xHost, convStateHost, weightHost, biasHost, outputRef, convStateRef, batch, convDim,
        kernelSize, activation);

    auto xDevice = rt::Tensor({batch, convDim, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateDevice = rt::Tensor({batch, convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({convDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, convDim, 1}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateOutDevice = rt::Tensor({batch, convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy(xDevice.rawPointer(), xHost.data(), xHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(convStateDevice.rawPointer(), convStateHost.data(), convStateHost.size() * sizeof(half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weightDevice.rawPointer(), weightHost.data(), weightHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(biasDevice.rawPointer(), biasHost.data(), biasHost.size() * sizeof(half), cudaMemcpyHostToDevice));

    causalConv1dDecode(xDevice.rawPointer(), convStateDevice.rawPointer(), weightDevice.rawPointer(),
        biasDevice.rawPointer(), outputDevice.rawPointer(), convStateOutDevice.rawPointer(), batch, convDim, kernelSize,
        activation, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> outputFromGpu(outputRef.size());
    std::vector<half> convStateFromGpu(convStateRef.size());
    CUDA_CHECK(cudaMemcpy(outputFromGpu.data(), outputDevice.rawPointer(), outputFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(convStateFromGpu.data(), convStateOutDevice.rawPointer(), convStateFromGpu.size() * sizeof(half),
        cudaMemcpyDeviceToHost));

    auto [rtol, atol] = getTolerance<half>();
    int32_t outMismatches = 0;
    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        if (!isclose(outputFromGpu[i], outputRef[i], rtol, atol))
        {
            if (outMismatches < 5)
            {
                std::cout << "Decode output mismatch at " << i << ": got " << __half2float(outputFromGpu[i])
                          << ", expected " << __half2float(outputRef[i]) << std::endl;
            }
            outMismatches++;
        }
    }
    EXPECT_EQ(outMismatches, 0);

    int32_t stateMismatches = 0;
    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        if (!isclose(convStateFromGpu[i], convStateRef[i], rtol, atol))
        {
            if (stateMismatches < 5)
            {
                std::cout << "Decode state mismatch at " << i << ": got " << __half2float(convStateFromGpu[i])
                          << ", expected " << __half2float(convStateRef[i]) << std::endl;
            }
            stateMismatches++;
        }
    }
    EXPECT_EQ(stateMismatches, 0);

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "causalConv1dDecode: batch=" << batch << ", convDim=" << convDim << ", kernelSize=" << kernelSize
              << " PASS" << std::endl;
}

} // anonymous namespace

// =============================================================================
// Test Cases: QKNorm
// =============================================================================

TEST(GatedDeltaNetQKNorm, Batch1_Heads8_Dim64)
{
    runComputeQKNormTest(1, 8, 64);
}

TEST(GatedDeltaNetQKNorm, Batch2_Heads16_Dim128)
{
    runComputeQKNormTest(2, 16, 128);
}

TEST(GatedDeltaNetQKNorm, Qwen35_Heads16_Dim128)
{
    runComputeQKNormTest(1, 16, 128);
}

// =============================================================================
// Test Cases: scaleQK
// =============================================================================

TEST(GatedDeltaNetScaleQK, Batch1_Heads8_Dim64)
{
    runScaleQKTest(1, 8, 64);
}

TEST(GatedDeltaNetScaleQK, Qwen35_Heads16_Dim128)
{
    runScaleQKTest(1, 16, 128);
}

// =============================================================================
// Test Cases: recurrentGatedDeltaStep
// =============================================================================

TEST(GatedDeltaNetRecurrentStep, Qwen35_Basic)
{
    runRecurrentStepTest(1, 32, 128, 128, false);
}

TEST(GatedDeltaNetRecurrentStep, Qwen35_WithInitialState)
{
    runRecurrentStepTest(1, 32, 128, 128, true);
}

TEST(GatedDeltaNetRecurrentStep, Batch2_VHeads16_Dim64)
{
    runRecurrentStepTest(2, 16, 64, 64, false);
}

TEST(GatedDeltaNetRecurrentStep, Batch1_VHeads8_Dim128)
{
    runRecurrentStepTest(1, 8, 128, 128, false);
}

TEST(GatedDeltaNetRecurrentStep, Batch1_VHeads8_KDim64_VDim128)
{
    runRecurrentStepTest(1, 8, 128, 64, false);
}

// =============================================================================
// Test Cases: gatedDeltaNetSerialLoop
// =============================================================================

TEST(GatedDeltaNetSerialLoop, Qwen35_Seq4)
{
    runSerialLoopTest(1, 4, 32, 128, 128, false);
}

TEST(GatedDeltaNetSerialLoop, Qwen35_Seq8_WithInitialState)
{
    runSerialLoopTest(1, 8, 32, 128, 128, true);
}

TEST(GatedDeltaNetSerialLoop, Seq16_Batch2)
{
    runSerialLoopTest(2, 16, 16, 64, 64, false);
}

TEST(GatedDeltaNetSerialLoop, Seq8_VHeads8_KDim64_VDim128)
{
    runSerialLoopTest(1, 8, 8, 128, 64, false);
}

// =============================================================================
// Test Cases: causalConv1dPrefill
// =============================================================================

TEST(GatedDeltaNetCausalConv1dPrefill, Qwen35_ConvDim_Seq1)
{
    runCausalConv1dPrefillTest(1, 6144, 1, 4, 0);
}

TEST(GatedDeltaNetCausalConv1dPrefill, Seq16_ConvDim4096_Kernel4)
{
    runCausalConv1dPrefillTest(1, 4096, 16, 4, 0);
}

TEST(GatedDeltaNetCausalConv1dPrefill, Batch4_ConvDim2048_Kernel3)
{
    runCausalConv1dPrefillTest(4, 2048, 8, 3, 0);
}

TEST(GatedDeltaNetCausalConv1dPrefill, ConvDim1024_Seq4_Kernel2)
{
    runCausalConv1dPrefillTest(2, 1024, 4, 2, 0);
}

// =============================================================================
// Test Cases: causalConv1dDecode
// =============================================================================

TEST(GatedDeltaNetCausalConv1dDecode, Qwen35_ConvDim)
{
    runCausalConv1dDecodeTest(1, 6144, 4, 0);
}

TEST(GatedDeltaNetCausalConv1dDecode, Batch4_ConvDim2048_Kernel3)
{
    runCausalConv1dDecodeTest(4, 2048, 3, 0);
}

TEST(GatedDeltaNetCausalConv1dDecode, ConvDim1024_Kernel2)
{
    runCausalConv1dDecodeTest(2, 1024, 2, 0);
}
