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
namespace plugins
{

//! GEMM kernel for GatedDeltaNet projections
//! C = A @ B.T where C[M,N], A[M,K], B[N,K]
void gatedDeltaGemm(
    __half* C, __half const* A, __half const* B, int32_t M, int32_t N, int32_t K, cudaStream_t stream);

//! Sigmoid activation in-place
void sigmoidActivation(__half* data, int32_t numElements, cudaStream_t stream);

//! Silu activation in-place
void siluActivation(__half* data, int32_t numElements, cudaStream_t stream);

//! Transpose: [B, M, N] -> [B, N, M]
void transposeTensor(
    __half* dst, __half const* src, int32_t B, int32_t M, int32_t N, cudaStream_t stream);

//! RMSNorm with gating: output = z * RMSNorm(x)
//! RMSNorm(x) = x * weight * bias / (||x|| + eps)
//! z is assumed to already have silu activation applied
void rmsNormGating(
    __half const* x, __half const* z, __half const* weight, __half const* bias,
    __half* output, int32_t batch, int32_t seqLen, int32_t numHeads, int32_t headDim, cudaStream_t stream);

} // namespace plugins
} // namespace trt_edgellm
