# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gated Delta Net Plugin Operations for TensorRT ONNX Export

Provides dummy PyTorch custom ops and ONNX symbolic functions for:
  - trt::gated_delta_net_causal_conv1d: depthwise causal 1D conv with SiLU
  - trt::gated_delta_rule: gated delta rule linear attention

These map to the C++ TensorRT plugins:
  - GatedDeltaNetCausalConv1d
  - GatedDeltaRule
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from ...common import ONNX_OPSET_VERSION

# ---------------------------------------------------------------------------
# Dynamo exporter support (PyTorch 2.10+)
# ---------------------------------------------------------------------------

try:
    from onnxscript import ir
    from torch.onnx._internal.exporter import _core

    def _emit_custom_node(domain: str, op_type: str, inputs, attrs, num_outputs: int):
        """Emit a custom-domain ONNX node using the active dynamo tracer."""
        tracer = _core.current_tracer
        if tracer is None:
            raise RuntimeError(
                "No ONNX tracer is currently active. "
                "This function should only be called during ONNX export."
            )
        outputs = [ir.Value() for _ in range(num_outputs)]
        node = ir.Node(
            domain,
            op_type,
            inputs=inputs,
            attributes=attrs,
            outputs=outputs,
        )
        tracer.nodes.append(node)
        return outputs[0] if num_outputs == 1 else tuple(outputs)

    def _onnx_attr(name: str, value):
        """Build an onnxscript ir.Attr for INT attributes."""
        return ir.Attr(name, ir.AttributeType.INT, value)

    def _symbolic_gated_delta_net_causal_conv1d_dynamo(
        x, weight, bias, conv_state, kernel_size=4, activation=0
    ):
        """Dynamo-compatible ONNX symbolic for gated_delta_net_causal_conv1d."""
        attrs = {
            "kernel_size": _onnx_attr("kernel_size", kernel_size),
            "activation": _onnx_attr("activation", activation),
        }
        return _emit_custom_node(
            "trt", "GatedDeltaNetCausalConv1d", [x, weight, bias, conv_state], attrs, 2
        )

    def _symbolic_gated_delta_rule_dynamo(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        num_v_heads=1,
        head_v_dim=64,
        head_k_dim=64,
        use_qk_l2norm=1,
    ):
        """Dynamo-compatible ONNX symbolic for gated_delta_rule."""
        inputs = [query, key, value, g, beta]
        if initial_state is not None:
            inputs.append(initial_state)
        attrs = {
            "num_v_heads": _onnx_attr("num_v_heads", num_v_heads),
            "head_v_dim": _onnx_attr("head_v_dim", head_v_dim),
            "head_k_dim": _onnx_attr("head_k_dim", head_k_dim),
            "use_qk_l2norm": _onnx_attr("use_qk_l2norm", use_qk_l2norm),
        }
        return _emit_custom_node("trt", "GatedDeltaRule", inputs, attrs, 2)

    _DYNAMO_SYMBOLICS_AVAILABLE = True
except Exception:
    _DYNAMO_SYMBOLICS_AVAILABLE = False
    _symbolic_gated_delta_net_causal_conv1d_dynamo = None  # type: ignore[misc,assignment]
    _symbolic_gated_delta_rule_dynamo = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# ONNX OpSchemas
# ---------------------------------------------------------------------------

gated_delta_net_causal_conv1d_schema = OpSchema(
    name="GatedDeltaNetCausalConv1d",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom Gated Delta Net causal 1D depthwise convolution plugin with SiLU.",
    inputs=[
        OpSchema.FormalParameter(
            name="x", description="Input tensor [B, conv_dim, S]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="weight",
            description="Conv weight [conv_dim, kernel_size]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="bias", description="Conv bias [conv_dim]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="conv_state",
            description="Conv state [B, conv_dim, kernel_size]",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output", description="Conv output [B, conv_dim, S]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="conv_state_out",
            description="Updated conv state [B, conv_dim, kernel_size]",
            type_str="T",
        ),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(
            name="kernel_size",
            type=OpSchema.AttrType.INT,
            description="Convolution kernel size",
            required=True,
        ),
        OpSchema.Attribute(
            name="activation",
            type=OpSchema.AttrType.INT,
            description="Activation type (0=SiLU)",
            required=True,
        ),
    ],
)

gated_delta_rule_schema = OpSchema(
    name="GatedDeltaRule",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom Gated Delta Rule linear attention plugin.",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Query [B, S, num_v_heads, head_k_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key", description="Key [B, S, num_v_heads, head_k_dim]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value [B, S, num_v_heads, head_v_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="g", description="Decay [B, S, num_v_heads]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="beta", description="Beta gate [B, S, num_v_heads]", type_str="T"
        ),
        OpSchema.FormalParameter(
            name="initial_state",
            description="Initial recurrent state [B, num_v_heads, head_k_dim, head_v_dim]",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Attention output [B, S, num_v_heads, head_v_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="final_state",
            description="Final recurrent state [B, num_v_heads, head_k_dim, head_v_dim]",
            type_str="T",
        ),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_v_heads",
            type=OpSchema.AttrType.INT,
            description="Number of value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_v_dim",
            type=OpSchema.AttrType.INT,
            description="Value head dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_k_dim",
            type=OpSchema.AttrType.INT,
            description="Key head dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="use_qk_l2norm",
            type=OpSchema.AttrType.INT,
            description="Apply QK L2Norm (1=yes, 0=no)",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Symbolic functions
# ---------------------------------------------------------------------------


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i")
def symbolic_gated_delta_net_causal_conv1d(
    g, x, weight, bias, conv_state, kernel_size, activation
):
    """Map trt::gated_delta_net_causal_conv1d to ONNX custom op."""
    output, conv_state_out = g.op(
        "trt::GatedDeltaNetCausalConv1d",
        x,
        weight,
        bias,
        conv_state,
        kernel_size_i=kernel_size,
        activation_i=activation,
        outputs=2,
    )
    output.setType(x.type())
    conv_state_out.setType(conv_state.type())
    return output, conv_state_out


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "i", "i", "i", "i")
def symbolic_gated_delta_rule(
    g,
    query,
    key,
    value,
    g_tensor,
    beta,
    initial_state,
    num_v_heads,
    head_v_dim,
    head_k_dim,
    use_qk_l2norm,
):
    """Map trt::gated_delta_rule to ONNX custom op."""
    inputs = [query, key, value, g_tensor, beta]
    if initial_state is not None and initial_state.type().kind() != "NoneType":
        inputs.append(initial_state)

    output, final_state = g.op(
        "trt::GatedDeltaRule",
        *inputs,
        num_v_heads_i=num_v_heads,
        head_v_dim_i=head_v_dim,
        head_k_dim_i=head_k_dim,
        use_qk_l2norm_i=use_qk_l2norm,
        outputs=2,
    )
    output.setType(value.type())
    final_state.setType(value.type())
    return output, final_state


# ---------------------------------------------------------------------------
# Dummy PyTorch custom ops (for tracing only)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::gated_delta_net_causal_conv1d", mutates_args=())
def gated_delta_net_causal_conv1d_plugin(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    kernel_size: int,
    activation: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dummy causal conv1d for ONNX tracing. Not used at runtime."""
    return torch.zeros_like(x), conv_state.clone()


@gated_delta_net_causal_conv1d_plugin.register_fake
def _gated_delta_net_causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    kernel_size: int,
    activation: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake impl for torch.export / dynamo tracing."""
    return torch.empty_like(x), torch.empty_like(conv_state)


@gated_delta_net_causal_conv1d_plugin.register_kernel("cpu")
def _gated_delta_net_causal_conv1d_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    kernel_size: int,
    activation: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real CPU implementation for numerical testing."""
    if x.shape[-1] == 1 and torch_causal_conv1d_update is not None:
        # Decode path: shift-insert + conv
        out = torch_causal_conv1d_update(x, conv_state, weight, bias, activation="silu")
        return out, conv_state.clone()
    else:
        # Prefill path: standard causal conv1d
        _, hidden_size, seq_len = x.shape
        padded = torch.nn.functional.pad(x, (kernel_size - 1, 0))
        w = weight.unsqueeze(1).to(x.dtype)  # [hidden_size, 1, kernel_size]
        b = bias.to(x.dtype) if bias is not None else None
        out = torch.nn.functional.conv1d(padded, w, b, padding=0, groups=hidden_size)
        out = out[:, :, :seq_len]
        if activation == 0:
            out = torch.nn.functional.silu(out)
        # conv_state_out: last kernel_size elements, zero-padded if needed
        conv_state_out = torch.zeros_like(conv_state)
        if seq_len >= kernel_size:
            conv_state_out[:, :, :] = x[:, :, seq_len - kernel_size : seq_len]
        else:
            pad = kernel_size - seq_len
            conv_state_out[:, :, pad:] = x[:, :, :seq_len]
        return out, conv_state_out


@torch.library.custom_op("trt::gated_delta_rule", mutates_args=())
def gated_delta_rule_plugin(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    num_v_heads: int = 1,
    head_v_dim: int = 64,
    head_k_dim: int = 64,
    use_qk_l2norm: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dummy gated delta rule for ONNX tracing. Not used at runtime."""
    batch_size, seq_len, _, _ = value.shape
    output = torch.zeros(
        batch_size,
        seq_len,
        num_v_heads,
        head_v_dim,
        dtype=value.dtype,
        device=value.device,
    )
    if initial_state is not None:
        final_state = initial_state.clone()
    else:
        final_state = torch.zeros(
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            dtype=value.dtype,
            device=value.device,
        )
    return output, final_state


@gated_delta_rule_plugin.register_fake
def _gated_delta_rule_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    num_v_heads: int = 1,
    head_v_dim: int = 64,
    head_k_dim: int = 64,
    use_qk_l2norm: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake impl for torch.export / dynamo tracing."""
    batch_size, seq_len, _, _ = value.shape
    output = torch.empty(
        batch_size,
        seq_len,
        num_v_heads,
        head_v_dim,
        dtype=value.dtype,
        device=value.device,
    )
    if initial_state is not None:
        final_state = torch.empty_like(initial_state)
    else:
        final_state = torch.empty(
            batch_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            dtype=value.dtype,
            device=value.device,
        )
    return output, final_state


@gated_delta_rule_plugin.register_kernel("cpu")
def _gated_delta_rule_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    num_v_heads: int = 1,
    head_v_dim: int = 64,
    head_k_dim: int = 64,
    use_qk_l2norm: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real CPU implementation for numerical testing."""
    if query.shape[1] == 1 and torch_recurrent_gated_delta_rule is not None:
        out, final_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm),
        )
    elif torch_chunk_gated_delta_rule is not None:
        out, final_state = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm),
        )
    else:
        raise RuntimeError("Gated delta rule reference implementation not available")
    return out, final_state


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_gdn_plugin_onnx_symbolic_functions() -> None:
    """Register ONNX symbolic functions for Gated Delta Net plugins (TorchScript path)."""
    register_custom_op_symbolic(
        "trt::gated_delta_net_causal_conv1d",
        symbolic_gated_delta_net_causal_conv1d,
        ONNX_OPSET_VERSION,
    )
    register_custom_op_symbolic(
        "trt::gated_delta_rule",
        symbolic_gated_delta_rule,
        ONNX_OPSET_VERSION,
    )


def get_gdn_plugin_dynamo_translation_table() -> Dict[Callable, Callable]:
    """Return a ``custom_translation_table`` for torch.onnx.export(..., dynamo=True).

    Usage::

        from tensorrt_edgellm.llm_models.layers.gdn_plugin import (
            get_gdn_plugin_dynamo_translation_table,
        )

        torch.onnx.export(
            model,
            args,
            f,
            dynamo=True,
            custom_translation_table=get_gdn_plugin_dynamo_translation_table(),
        )
    """
    if not _DYNAMO_SYMBOLICS_AVAILABLE:
        raise RuntimeError(
            "Dynamo symbolic functions are not available. "
            "Ensure you are using PyTorch >= 2.10 with onnx-script support."
        )
    return {
        torch.ops.trt.gated_delta_net_causal_conv1d.default: _symbolic_gated_delta_net_causal_conv1d_dynamo,
        torch.ops.trt.gated_delta_rule.default: _symbolic_gated_delta_rule_dynamo,
    }
