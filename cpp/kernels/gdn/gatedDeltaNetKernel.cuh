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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernels
{

//! Compute L2 normalized query and key for GDN
//! q_normed = q / (||q|| + eps), same for k
//! This is applied inside the recurrent gated delta rule
void computeGatedDeltaQKNorm(
    void const* q,      // [batch, num_heads, head_dim] or [batch, seq, num_heads, head_dim]
    void const* k,      // [batch, num_heads, head_dim] or [batch, seq, num_heads, head_dim]
    void* q_normed,     // same shape as q
    void* k_normed,     // same shape as k
    int32_t batch,
    int32_t numHeads,
    int32_t headDim,
    cudaStream_t stream);

//! Single-step recurrent gated delta rule
//! Processes one token at a time and updates recurrent state
//! DT = softplus(dt_proj(x) + dt_bias)
//! A = -exp(A_log)
//! g = -exp(A_log) * softplus(a + dt_bias)
//! state_new = state * exp(A * DT) + v * DT * g
//! y = sum_i(state_new_i * k_i) * beta + v * (1 - beta)
void recurrentGatedDeltaStep(
    void const* query,          // [batch, num_v_heads, head_v_dim]
    void const* key,            // [batch, num_k_heads, head_k_dim]
    void const* value,          // [batch, num_v_heads, head_v_dim]
    void const* g,              // [batch, num_v_heads] - gating factor
    void const* beta,           // [batch, num_v_heads] - sigmoid gate
    void const* initialState,   // [batch, num_v_heads, head_v_dim] or None
    void* output,               // [batch, num_v_heads, head_v_dim]
    void* finalState,           // [batch, num_v_heads, head_v_dim]
    int32_t batch,
    int32_t numKVHeads,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    int32_t vRepeatFactor,     // num_v_heads / num_k_heads
    bool hasInitialState,
    cudaStream_t stream);

//! Serial loop over the recurrent gated delta rule
//! For prefill with seq_len > 1, loops over each step serially
void gatedDeltaNetSerialLoop(
    void const* query,          // [batch, seq_len, num_v_heads, head_v_dim]
    void const* key,            // [batch, seq_len, num_k_heads, head_k_dim]
    void const* value,          // [batch, seq_len, num_v_heads, head_v_dim]
    void const* g,              // [batch, seq_len, num_v_heads]
    void const* beta,           // [batch, seq_len, num_v_heads]
    void const* initialState,   // [batch, num_v_heads, head_v_dim] or nullptr
    void* output,               // [batch, seq_len, num_v_heads, head_v_dim]
    void* finalState,           // [batch, num_v_heads, head_v_dim]
    int32_t batch,
    int32_t seqLen,
    int32_t numKVHeads,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    int32_t vRepeatFactor,
    bool hasInitialState,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream);

//! Scale query and key by 1/sqrt(head_dim)
void scaleQK(
    void* q,
    int32_t batchSeq,
    int32_t numHeads,
    int32_t headDim,
    cudaStream_t stream);

//! Single-step recurrent gated delta rule (decode path)
//! State S is a matrix [batch, num_v_heads, head_k_dim, head_v_dim]
void recurrentGatedDeltaStep(
    void const* query,          // [batch, num_v_heads, head_k_dim]
    void const* key,            // [batch, num_v_heads, head_k_dim]
    void const* value,          // [batch, num_v_heads, head_v_dim]
    void const* g,              // [batch, num_v_heads]
    void const* beta,           // [batch, num_v_heads]
    void const* initialState,   // [batch, num_v_heads, head_k_dim, head_v_dim] or nullptr
    void* output,               // [batch, num_v_heads, head_v_dim]
    void* finalState,           // [batch, num_v_heads, head_k_dim, head_v_dim]
    int32_t batch,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    cudaStream_t stream);

//! Serial loop over the recurrent gated delta rule for prefill
void gatedDeltaNetSerialLoop(
    void const* query,          // [batch, seq_len, num_v_heads, head_k_dim]
    void const* key,            // [batch, seq_len, num_v_heads, head_k_dim]
    void const* value,          // [batch, seq_len, num_v_heads, head_v_dim]
    void const* g,              // [batch, seq_len, num_v_heads]
    void const* beta,           // [batch, seq_len, num_v_heads]
    void const* initialState,   // [batch, num_v_heads, head_k_dim, head_v_dim] or nullptr
    void* output,               // [batch, seq_len, num_v_heads, head_v_dim]
    void* finalState,           // [batch, num_v_heads, head_k_dim, head_v_dim]
    int32_t batch,
    int32_t seqLen,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    cudaStream_t stream);

//! Causal conv1d prefill path
void causalConv1dPrefill(
    void const* x,              // [batch, conv_dim, seq_len]
    void const* weight,         // [conv_dim, conv_kernel_size]
    void const* bias,           // [conv_dim]
    void* output,               // [batch, conv_dim, seq_len]
    void* convStateOut,         // [batch, conv_dim, conv_kernel_size]
    int32_t batch,
    int32_t convDim,
    int32_t seqLen,
    int32_t convKernelSize,
    int32_t activation,
    cudaStream_t stream);

//! Causal conv1d decode path (seq_len == 1)
void causalConv1dDecode(
    void const* x,              // [batch, conv_dim, 1]
    void const* convState,      // [batch, conv_dim, conv_kernel_size]
    void const* weight,         // [conv_dim, conv_kernel_size]
    void const* bias,           // [conv_dim]
    void* output,               // [batch, conv_dim, 1]
    void* convStateOut,         // [batch, conv_dim, conv_kernel_size]
    int32_t batch,
    int32_t convDim,
    int32_t convKernelSize,
    int32_t activation,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Backward-compatible overloads for gatedDeltaNetPlugin.cpp (DO NOT MODIFY)
// ---------------------------------------------------------------------------

//! Legacy causal conv1d update (prefill + decode unified)
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
    cudaStream_t stream);

//! Legacy recurrent gated delta step overload
inline void recurrentGatedDeltaStep(
    void const* query,
    void const* key,
    void const* value,
    void const* g,
    void const* beta,
    void const* initialState,
    void* output,
    void* finalState,
    int32_t batch,
    int32_t /*numKVHeads*/,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    int32_t /*vRepeatFactor*/,
    bool /*hasInitialState*/,
    cudaStream_t stream)
{
    recurrentGatedDeltaStep(
        query, key, value, g, beta, initialState,
        output, finalState, batch, numVHeads, headVDim, headKDim, stream);
}

//! Legacy serial loop overload
inline void gatedDeltaNetSerialLoop(
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
    int32_t /*numKVHeads*/,
    int32_t numVHeads,
    int32_t headVDim,
    int32_t headKDim,
    int32_t /*vRepeatFactor*/,
    bool /*hasInitialState*/,
    void* /*workspace*/,
    size_t /*workspaceSize*/,
    cudaStream_t stream)
{
    gatedDeltaNetSerialLoop(
        query, key, value, g, beta, initialState,
        output, finalState, batch, seqLen, numVHeads, headVDim, headKDim, stream);
}

} // namespace kernels
} // namespace trt_edgellm
