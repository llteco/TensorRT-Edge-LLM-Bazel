/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runner for the flexible context flash-attention kernel.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace trt_edgellm
{
namespace kernels
{

/*!
 * @brief Host-side runner for the flexible context FMHA kernel.
 *
 * This runner is intentionally simple: it receives the same Q/K/V/O pointers
 * that the highly-optimised ContextFMHARunner uses, but launches the flexible
 * kernel instead of looking up a pre-compiled cubin.
 */
class FlexibleContextFMHARunner
{
public:
    FlexibleContextFMHARunner(
        int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize);

    //! @brief Run the flexible attention kernel.
    //! @param q           Q tensor pointer (device).
    //! @param k           K tensor pointer (device).
    //! @param v           V tensor pointer (device).
    //! @param o           Output tensor pointer (device).
    //! @param scale       Softmax scale (typically 1/sqrt(headSize)).
    //! @param causal      Whether to apply causal masking.
    //! @param stream      CUDA stream.
    void run(void const* q, void const* k, void const* v, void* o,
        float scale, bool causal, cudaStream_t stream) const;

private:
    int32_t mBatchSize;
    int32_t mSeqLen;
    int32_t mNumQHeads;
    int32_t mNumKVHeads;
    int32_t mHeadSize;
};

} // namespace kernels
} // namespace trt_edgellm
