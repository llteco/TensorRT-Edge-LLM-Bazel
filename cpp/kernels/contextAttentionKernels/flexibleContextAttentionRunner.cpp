/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "flexibleContextAttentionRunner.h"
#include "flexibleContextAttention.h"
#include "common/checkMacros.h"
#include <cuda_fp16.h>
#include <cmath>

namespace trt_edgellm
{
namespace kernels
{

FlexibleContextFMHARunner::FlexibleContextFMHARunner(
    int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize)
    : mBatchSize(batchSize)
    , mSeqLen(seqLen)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
{
    check::check(numQHeads % numKVHeads == 0, "GQA requires numQHeads %% numKVHeads == 0");
}

void FlexibleContextFMHARunner::run(void const* q, void const* k, void const* v, void* o,
    float scale, bool causal, cudaStream_t stream) const
{
    // For now we only support FP16.  BF16 path is compiled but not wired here
    // until the plugin passes the correct type tag.
    launchFlexibleContextAttention(static_cast<half const*>(q), static_cast<half const*>(k),
        static_cast<half const*>(v), static_cast<half*>(o), mBatchSize, mSeqLen, mNumQHeads, mNumKVHeads,
        mHeadSize, scale, causal, stream);
}

} // namespace kernels
} // namespace trt_edgellm
