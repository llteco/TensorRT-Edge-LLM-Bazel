#!/usr/bin/env python3
"""
Generate test vectors for GDN CausalConv1D and GatedDeltaRule kernels.

Reference implementations are imported directly from the transformers Qwen3.5
modeling code to guarantee numerical alignment.

Outputs safetensors files to unittests/resources/.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def generate_causal_conv1d_cases(resources_dir: str) -> None:
    """Generate CausalConv1D test vectors using Qwen3.5 reference."""
    cases = [
        {
            "name": "gdn_causal_conv1d_1b_16d_1s_4k_decode",
            "batch": 1,
            "conv_dim": 16,
            "seq_len": 1,
            "kernel_size": 4,
        },
        {
            "name": "gdn_causal_conv1d_1b_32d_4s_4k_prefill",
            "batch": 1,
            "conv_dim": 32,
            "seq_len": 4,
            "kernel_size": 4,
        },
        {
            "name": "gdn_causal_conv1d_2b_64d_16s_3k_prefill",
            "batch": 2,
            "conv_dim": 64,
            "seq_len": 16,
            "kernel_size": 3,
        },
    ]

    for case in cases:
        B, D, S, K = (
            case["batch"],
            case["conv_dim"],
            case["seq_len"],
            case["kernel_size"],
        )
        rng = np.random.default_rng(seed=42)
        x = rng.uniform(-0.5, 0.5, size=(B, D, S)).astype(np.float16)
        weight = rng.uniform(-0.3, 0.3, size=(D, K)).astype(np.float16)
        bias = rng.uniform(-0.1, 0.1, size=(D,)).astype(np.float16)

        if S == 1:
            # Decode path: Qwen3.5 torch_causal_conv1d_update
            conv_state = np.zeros((B, D, K), dtype=np.float16)
            x_t = torch.from_numpy(x)
            conv_state_t = torch.from_numpy(conv_state)
            weight_t = torch.from_numpy(weight)
            bias_t = torch.from_numpy(bias)
            out_t = torch_causal_conv1d_update(
                x_t, conv_state_t, weight_t, bias_t, activation="silu"
            )
            output = out_t.numpy().astype(np.float16)
            conv_state_out = conv_state_t.numpy().astype(np.float16)
        else:
            # Prefill path: Qwen3.5 fallback conv1d
            x_t = torch.from_numpy(x)
            w_t = torch.from_numpy(weight).unsqueeze(1)  # [D, 1, K]
            b_t = torch.from_numpy(bias)
            out_t = F.conv1d(x_t, w_t, b_t, padding=K - 1, groups=D)
            out_t = F.silu(out_t[:, :, :S])
            output = out_t.numpy().astype(np.float16)
            # Capture conv state: last kernel_size inputs, zero-padded
            conv_state_out = np.zeros((B, D, K), dtype=np.float16)
            if S >= K:
                conv_state_out[:, :, :] = x[:, :, S - K : S]
            else:
                pad = K - S
                conv_state_out[:, :, pad:] = x[:, :, :S]

        tensors = {
            "x": x,
            "weight": weight,
            "bias": bias,
            "output": output,
            "conv_state_out": conv_state_out,
        }
        path = os.path.join(resources_dir, f"{case['name']}.safetensors")
        save_file(tensors, path)
        print(f"Generated: {path}")


def generate_gated_delta_rule_cases(resources_dir: str) -> None:
    """Generate GatedDeltaRule test vectors using Qwen3.5 reference."""
    cases = [
        {
            "name": "gated_delta_rule_1b_1s_8h_64k_64v_decode",
            "batch": 1,
            "seq_len": 1,
            "num_v_heads": 8,
            "head_k_dim": 64,
            "head_v_dim": 64,
            "has_initial_state": False,
        },
        {
            "name": "gated_delta_rule_1b_1s_8h_64k_128v_decode_asym",
            "batch": 1,
            "seq_len": 1,
            "num_v_heads": 8,
            "head_k_dim": 64,
            "head_v_dim": 128,
            "has_initial_state": False,
        },
        {
            "name": "gated_delta_rule_1b_4s_16h_128k_128v_prefill",
            "batch": 1,
            "seq_len": 4,
            "num_v_heads": 16,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "has_initial_state": False,
        },
        {
            "name": "gated_delta_rule_1b_8s_16h_128k_128v_prefill_with_state",
            "batch": 1,
            "seq_len": 8,
            "num_v_heads": 16,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "has_initial_state": True,
        },
        {
            "name": "gated_delta_rule_2b_16s_8h_64k_64v_prefill_batch2",
            "batch": 2,
            "seq_len": 16,
            "num_v_heads": 8,
            "head_k_dim": 64,
            "head_v_dim": 64,
            "has_initial_state": False,
        },
    ]

    for case in cases:
        B, S, H, Kdim, Vdim = (
            case["batch"],
            case["seq_len"],
            case["num_v_heads"],
            case["head_k_dim"],
            case["head_v_dim"],
        )
        rng = np.random.default_rng(seed=1234)
        q = rng.uniform(-1.0, 1.0, size=(B, S, H, Kdim)).astype(np.float16)
        k = rng.uniform(-0.5, 0.5, size=(B, S, H, Kdim)).astype(np.float16)
        v = rng.uniform(-1.0, 1.0, size=(B, S, H, Vdim)).astype(np.float16)
        g = rng.uniform(-2.0, 0.0, size=(B, S, H)).astype(np.float16)
        beta = rng.uniform(0.0, 1.0, size=(B, S, H)).astype(np.float16)

        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        g_t = torch.from_numpy(g)
        beta_t = torch.from_numpy(beta)

        initial_state = None
        initial_state_t = None
        if case["has_initial_state"]:
            initial_state = rng.uniform(-0.5, 0.5, size=(B, H, Kdim, Vdim)).astype(
                np.float16
            )
            initial_state_t = torch.from_numpy(initial_state)

        if S == 1:
            out_t, final_state_t = torch_recurrent_gated_delta_rule(
                q_t,
                k_t,
                v_t,
                g_t,
                beta_t,
                initial_state=initial_state_t,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            out_t, final_state_t = torch_chunk_gated_delta_rule(
                q_t,
                k_t,
                v_t,
                g_t,
                beta_t,
                chunk_size=64,
                initial_state=initial_state_t,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        assert final_state_t is not None
        output = out_t.numpy().astype(np.float16)
        final_state = final_state_t.numpy().astype(np.float16)

        tensors = {
            "query": q,
            "key": k,
            "value": v,
            "g": g,
            "beta": beta,
            "output": output,
            "final_state": final_state,
        }
        if initial_state is not None:
            tensors["initial_state"] = initial_state

        path = os.path.join(resources_dir, f"{case['name']}.safetensors")
        save_file(tensors, path)
        print(f"Generated: {path}")


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, "..", "resources")
    os.makedirs(resources_dir, exist_ok=True)

    generate_causal_conv1d_cases(resources_dir)
    generate_gated_delta_rule_cases(resources_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
