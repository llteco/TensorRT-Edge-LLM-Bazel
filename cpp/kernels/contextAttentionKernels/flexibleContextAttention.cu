/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flexible context flash-attention kernel.
 *
 * Algorithm: classic Flash-Attention (online softmax) with tile-based loops.
 * Each block processes one head and one tile of queries (Br rows).
 * Shared memory holds tiles of Q, K, V and the accumulating O tile.
 *
 * This is intentionally a simple / reference-quality implementation.
 * It trades peak performance for flexibility (arbitrary head size, easy
 * data-type extension, small code size).
 */

#include "flexibleContextAttention.h"
#include "common/checkMacros.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

namespace trt_edgellm
{
namespace kernels
{

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_float(half x)
{
    return __half2float(x);
}

__device__ __forceinline__ float to_float(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float to_float(float x)
{
    return x;
}

template <typename T>
struct FloatToT
{
    __device__ static inline T apply(float x);
};

template <>
struct FloatToT<half>
{
    __device__ static inline half apply(float x)
    {
        return __float2half_rn(x);
    }
};

template <>
struct FloatToT<__nv_bfloat16>
{
    __device__ static inline __nv_bfloat16 apply(float x)
    {
        return __float2bfloat16_rn(x);
    }
};

template <>
struct FloatToT<float>
{
    __device__ static inline float apply(float x)
    {
        return x;
    }
};

// ---------------------------------------------------------------------------
// Tile-size selection (host side)
// ---------------------------------------------------------------------------

struct TileConfig
{
    int br; //!< rows of Q processed per block
    int bc; //!< cols of K/V processed per inner-loop tile
    int threads;
};

static TileConfig chooseTileConfig(int head_size)
{
    TileConfig cfg;
    cfg.threads = 128;

    if (head_size <= 64)
    {
        cfg.br = 64;
        cfg.bc = 64;
    }
    else if (head_size <= 128)
    {
        cfg.br = 32;
        cfg.bc = 64;
    }
    else
    {
        cfg.br = 16;
        cfg.bc = 32;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Shared memory layout (all offsets in bytes, calculated at runtime):
//   [0                         ) -> Q_tile  [br * D]     (T)
//   [br*D*sizeof(T)            ) -> K_tile  [bc * D]     (T)
//   [(br+bc)*D*sizeof(T)       ) -> V_tile  [bc * D]     (T)
//   [(br+2*bc)*D*sizeof(T)     ) -> S_tile  [br * bc]    (float)
//   [(br+2*bc)*D*sizeof(T)+br*bc*sizeof(float)) -> O_acc [br * D] (float)

template <typename T>
__global__ void flexibleFlashAttentionKernel(T const* __restrict__ Q, T const* __restrict__ K,
    T const* __restrict__ V, T* __restrict__ O, int32_t const batch_size, int32_t const seq_len,
    int32_t const num_q_heads, int32_t const num_kv_heads, int32_t const head_size, float const scale,
    bool const causal, int32_t const br, int32_t const bc)
{
    int32_t const b = blockIdx.z;
    int32_t const hq = blockIdx.y;
    int32_t const q_tile = blockIdx.x;

    if (b >= batch_size)
        return;

    int32_t const hv = (num_q_heads == num_kv_heads) ? hq : (hq * num_kv_heads / num_q_heads);

    int64_t const q_stride_s = int64_t(num_q_heads) * head_size;
    int64_t const kv_stride_s = int64_t(num_kv_heads) * head_size;

    int64_t const q_base = (int64_t(b) * seq_len * num_q_heads + hq) * head_size;
    int64_t const kv_base = (int64_t(b) * seq_len * num_kv_heads + hv) * head_size;

    int32_t const q_start = q_tile * br;
    int32_t const q_end = min(q_start + br, seq_len);
    int32_t const num_q_rows = q_end - q_start;

    extern __shared__ char smem_raw[];
    T* smem_Q = reinterpret_cast<T*>(smem_raw);
    T* smem_K = reinterpret_cast<T*>(&smem_Q[br * head_size]);
    T* smem_V = reinterpret_cast<T*>(&smem_K[bc * head_size]);
    float* smem_S = reinterpret_cast<float*>(&smem_V[bc * head_size]);
    float* smem_O = reinterpret_cast<float*>(&smem_S[br * bc]);

    // Zero-initialise the O accumulator in shared memory.
    for (int idx = threadIdx.x; idx < num_q_rows * head_size; idx += blockDim.x)
    {
        smem_O[idx] = 0.0f;
    }
    __syncthreads();

    // Per-row running softmax statistics (kept in registers).
    float row_m[64];
    float row_l[64];
    for (int i = 0; i < num_q_rows; ++i)
    {
        row_m[i] = -std::numeric_limits<float>::infinity();
        row_l[i] = 0.0f;
    }

    // Load Q tile once (it does not change across KV tiles).
    for (int idx = threadIdx.x; idx < num_q_rows * head_size; idx += blockDim.x)
    {
        int32_t r = idx / head_size;
        int32_t d = idx % head_size;
        int64_t q_off = q_base + int64_t(q_start + r) * q_stride_s + d;
        smem_Q[r * head_size + d] = Q[q_off];
    }
    __syncthreads();

    int32_t const num_kv_tiles = (seq_len + bc - 1) / bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile)
    {
        int32_t const kv_start = kv_tile * bc;
        int32_t const kv_end = min(kv_start + bc, seq_len);
        int32_t const kv_len = kv_end - kv_start;

        // Load K tile.
        for (int idx = threadIdx.x; idx < kv_len * head_size; idx += blockDim.x)
        {
            int32_t r = idx / head_size;
            int32_t d = idx % head_size;
            int64_t k_off = kv_base + int64_t(kv_start + r) * kv_stride_s + d;
            smem_K[r * head_size + d] = K[k_off];
        }
        // Load V tile.
        for (int idx = threadIdx.x; idx < kv_len * head_size; idx += blockDim.x)
        {
            int32_t r = idx / head_size;
            int32_t d = idx % head_size;
            int64_t v_off = kv_base + int64_t(kv_start + r) * kv_stride_s + d;
            smem_V[r * head_size + d] = V[v_off];
        }
        __syncthreads();

        // Compute S = Q_tile @ K_tile^T  (num_q_rows x kv_len)
        for (int local_q = 0; local_q < num_q_rows; ++local_q)
        {
            int32_t global_q = q_start + local_q;
            for (int k = threadIdx.x; k < kv_len; k += blockDim.x)
            {
                float s = 0.0f;
                for (int d = 0; d < head_size; ++d)
                {
                    s += to_float(smem_Q[local_q * head_size + d]) * to_float(smem_K[k * head_size + d]);
                }
                s *= scale;

                if (causal && (kv_start + k) > global_q)
                {
                    s = -std::numeric_limits<float>::infinity();
                }
                smem_S[local_q * bc + k] = s;
            }
        }
        __syncthreads();

        // Online softmax + O update for each query row.
        for (int local_q = 0; local_q < num_q_rows; ++local_q)
        {
            // 1) Row max.
            float row_max = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < kv_len; ++k)
            {
                row_max = fmaxf(row_max, smem_S[local_q * bc + k]);
            }

            // 2) Exp and sum.
            float row_sum = 0.0f;
            for (int k = 0; k < kv_len; ++k)
            {
                float exp_s = expf(smem_S[local_q * bc + k] - row_max);
                smem_S[local_q * bc + k] = exp_s;
                row_sum += exp_s;
            }

            // 3) Update running stats.
            float new_m = fmaxf(row_m[local_q], row_max);
            float alpha = expf(row_m[local_q] - new_m);
            float beta = expf(row_max - new_m);
            row_l[local_q] = alpha * row_l[local_q] + beta * row_sum;

            // 4) Update O accumulator in shared memory.
            for (int d = threadIdx.x; d < head_size; d += blockDim.x)
            {
                float pv = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    pv += smem_S[local_q * bc + k] * to_float(smem_V[k * head_size + d]);
                }
                int o_idx = local_q * head_size + d;
                smem_O[o_idx] = alpha * smem_O[o_idx] + beta * pv;
            }
            row_m[local_q] = new_m;
            __syncthreads();
        }
    }

    // Finalise and write O.
    for (int local_q = 0; local_q < num_q_rows; ++local_q)
    {
        int32_t global_q = q_start + local_q;
        float inv_l = 1.0f / row_l[local_q];
        for (int d = threadIdx.x; d < head_size; d += blockDim.x)
        {
            int64_t o_off = q_base + int64_t(global_q) * q_stride_s + d;
            int o_idx = local_q * head_size + d;
            O[o_off] = FloatToT<T>::apply(smem_O[o_idx] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

template <typename T>
void launchFlexibleContextAttention(T const* q, T const* k, T const* v, T* o,
    int32_t batch_size, int32_t seq_len, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_size, float scale, bool causal, cudaStream_t stream)
{
    if (batch_size <= 0 || seq_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_size <= 0)
    {
        return;
    }

    check::check(num_q_heads % num_kv_heads == 0, "GQA requires num_q_heads %% num_kv_heads == 0");

    TileConfig cfg = chooseTileConfig(head_size);

    int32_t const num_q_tiles = (seq_len + cfg.br - 1) / cfg.br;
    dim3 grid(num_q_tiles, num_q_heads, batch_size);
    dim3 block(cfg.threads);

    size_t const smem_size = (cfg.br + 2 * cfg.bc) * head_size * sizeof(T)
        + cfg.br * cfg.bc * sizeof(float)
        + cfg.br * head_size * sizeof(float);

    flexibleFlashAttentionKernel<T><<<grid, block, smem_size, stream>>>(
        q, k, v, o, batch_size, seq_len, num_q_heads, num_kv_heads, head_size, scale, causal, cfg.br, cfg.bc);
}

// Explicit instantiations

template void launchFlexibleContextAttention<half>(half const* q, half const* k, half const* v, half* o,
    int32_t batch_size, int32_t seq_len, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_size, float scale, bool causal, cudaStream_t stream);

template void launchFlexibleContextAttention<__nv_bfloat16>(__nv_bfloat16 const* q, __nv_bfloat16 const* k,
    __nv_bfloat16 const* v, __nv_bfloat16* o, int32_t batch_size, int32_t seq_len, int32_t num_q_heads,
    int32_t num_kv_heads, int32_t head_size, float scale, bool causal, cudaStream_t stream);

// ---------------------------------------------------------------------------
// canImplement
// ---------------------------------------------------------------------------

bool canImplementFlexibleContextAttention(
    int32_t head_size, int32_t sm_version, cudaDataType data_type) noexcept
{
    (void) sm_version;
    if (head_size <= 0 || (head_size % 8) != 0)
    {
        return false;
    }
    if (data_type != CUDA_R_16F && data_type != CUDA_R_16BF)
    {
        return false;
    }
    return true;
}

} // namespace kernels
} // namespace trt_edgellm
