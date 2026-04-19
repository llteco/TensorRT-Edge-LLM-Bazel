/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flexible context flash-attention kernel that supports arbitrary head sizes
 * (including 256) and multiple data types.  Performance is not the primary
 * goal – correctness and flexibility are.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace trt_edgellm
{
namespace kernels
{

/*!
 * @brief Launch a flexible flash-attention kernel for the context (prefill) phase.
 *
 * Supports head sizes that are not covered by the pre-compiled FMHA_v2 cubins.
 *
 * Layout: BSHD (batch, seq, head, dim)
 *   Q: [batch_size, seq_len, num_q_heads, head_size]
 *   K: [batch_size, seq_len, num_kv_heads, head_size]
 *   V: [batch_size, seq_len, num_kv_heads, head_size]
 *   O: [batch_size, seq_len, num_q_heads, head_size]
 *
 * @tparam T  Data type (currently half / __nv_bfloat16 are supported).
 *
 * @param q           Device pointer to Q.
 * @param k           Device pointer to K.
 * @param v           Device pointer to V.
 * @param o           Device pointer to output O.
 * @param batch_size  Batch size.
 * @param seq_len     Sequence length (Q and KV share the same length for prefill).
 * @param num_q_heads Number of query heads.
 * @param num_kv_heads Number of key/value heads (GQA).
 * @param head_size   Head dimension (D).
 * @param scale       Softmax scale (typically 1/sqrt(D)).
 * @param causal      Whether to apply a causal mask.
 * @param stream      CUDA stream.
 */
template <typename T>
void launchFlexibleContextAttention(T const* q, T const* k, T const* v, T* o,
    int32_t batch_size, int32_t seq_len, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_size, float scale, bool causal, cudaStream_t stream);

/*!
 * @brief Query whether the flexible kernel can implement a given configuration.
 *
 * The flexible kernel is intended as a fallback for configurations that the
 * highly-optimised FMHA_v2 / CuTe DSL paths do not support (e.g. head_size=256).
 */
bool canImplementFlexibleContextAttention(
    int32_t head_size, int32_t sm_version, cudaDataType data_type) noexcept;

} // namespace kernels
} // namespace trt_edgellm
