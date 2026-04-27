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
Test single-layer Qwen3.5 GatedDeltaNet ONNX export with custom TRT plugins.

Validates:
  1. Numerical parity between original Qwen3_5GatedDeltaNet and EdgeLLMGatedDeltaNetLayer
  2. torchscript ONNX export produces trt_edgellm custom op nodes
  3. dynamo ONNX export produces trt_edgellm custom op nodes
"""

import os
import tempfile
import unittest
from typing import Tuple

import onnx
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from tensorrt_edgellm.llm_models.layers.gdn_plugin import (
    get_gdn_plugin_dynamo_translation_table,
    register_gdn_plugin_onnx_symbolic_functions,
)

# Dynamo exporter may not support custom_translation_table on older PyTorch
try:
    _DYNAMO_TABLE = get_gdn_plugin_dynamo_translation_table()
    _DYNAMO_CUSTOM_OP_SUPPORTED = True
except Exception:
    _DYNAMO_TABLE = {}
    _DYNAMO_CUSTOM_OP_SUPPORTED = False
from tensorrt_edgellm.llm_models.layers.layers import EdgeLLMGatedDeltaNetLayer

# Register ONNX symbolic functions once at import time
register_gdn_plugin_onnx_symbolic_functions()


class TestGatedDeltaNetLayer(unittest.TestCase):
    """Numerical parity and ONNX export tests for EdgeLLMGatedDeltaNetLayer."""

    @classmethod
    def setUpClass(cls):
        """Build a minimal Qwen3.5 config with one linear_attention layer."""
        cls.config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            layer_types=["linear_attention"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )
        cls.device = torch.device("cpu")
        cls.dtype = torch.float16

    def _create_mixer(self) -> Qwen3_5GatedDeltaNet:
        """Instantiate original Qwen3.5 GatedDeltaNet mixer."""
        mixer = Qwen3_5GatedDeltaNet(self.config, layer_idx=0)
        mixer = mixer.to(self.device).to(self.dtype).eval()
        return mixer

    def _create_wrapper(self, mixer: Qwen3_5GatedDeltaNet) -> EdgeLLMGatedDeltaNetLayer:
        """Build EdgeLLMGatedDeltaNetLayer from original mixer."""
        wrapper = EdgeLLMGatedDeltaNetLayer(mixer)
        wrapper = wrapper.to(self.device).to(self.dtype).eval()
        return wrapper

    def _make_states(
        self, mixer: Qwen3_5GatedDeltaNet, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized conv_state and recurrent_state."""
        conv_state = torch.zeros(
            batch_size,
            mixer.conv_dim,
            mixer.conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )
        recurrent_state = torch.zeros(
            batch_size,
            mixer.num_v_heads,
            mixer.head_k_dim,
            mixer.head_v_dim,
            device=self.device,
            dtype=self.dtype,
        )
        return conv_state, recurrent_state

    def _run_original_prefill(
        self,
        mixer: Qwen3_5GatedDeltaNet,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, DynamicCache]:
        """Run original mixer in prefill mode (no cache)."""
        cache = DynamicCache(config=self.config)
        output = mixer(
            hidden_states=hidden_states,
            cache_params=cache,
            attention_mask=None,
        )
        return output, cache

    def _run_original_decode(
        self,
        mixer: Qwen3_5GatedDeltaNet,
        hidden_states: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        """Run original mixer in decode mode (seq_len=1 with cache)."""
        output = mixer(
            hidden_states=hidden_states,
            cache_params=cache,
            attention_mask=None,
        )
        return output

    def _assert_close(self, a: torch.Tensor, b: torch.Tensor, msg: str):
        """Assert two tensors are close within FP16 tolerance."""
        rtol, atol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(a, b, rtol=rtol, atol=atol),
            f"{msg}: max diff = {(a - b).abs().max().item():.6f}",
        )

    # ------------------------------------------------------------------
    # Numerical parity tests
    # ------------------------------------------------------------------

    def test_prefill_numerical_parity(self):
        """Prefill (seq_len=4): wrapper vs original."""
        batch_size, seq_len = 2, 4
        mixer = self._create_mixer()
        wrapper = self._create_wrapper(mixer)

        hidden_states = torch.randn(
            batch_size,
            seq_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        # Original
        out_orig, cache = self._run_original_prefill(mixer, hidden_states)

        # Wrapper with zero states
        conv_state, recurrent_state = self._make_states(mixer, batch_size)
        out_wrap, conv_out, recurrent_out = wrapper(
            hidden_states,
            conv_state,
            recurrent_state,
        )

        self._assert_close(out_wrap, out_orig, "Prefill output mismatch")

    def test_decode_numerical_parity(self):
        """Decode (seq_len=1): wrapper vs original with prefill cache."""
        batch_size = 2
        mixer = self._create_mixer()
        wrapper = self._create_wrapper(mixer)

        # Prefill to populate cache
        prefill_len = 4
        hidden_prefill = torch.randn(
            batch_size,
            prefill_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        _, cache = self._run_original_prefill(mixer, hidden_prefill)

        # Extract states BEFORE running decode (original modifies cache in-place)
        conv_state = cache.layers[0].conv_states.clone().to(self.dtype)
        recurrent_state = cache.layers[0].recurrent_states.clone().to(self.dtype)

        # Decode step
        hidden_decode = torch.randn(
            batch_size,
            1,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        out_orig = self._run_original_decode(mixer, hidden_decode, cache)

        # Wrapper with the same pre-decode states
        out_wrap, conv_out, recurrent_out = wrapper(
            hidden_decode,
            conv_state,
            recurrent_state,
        )

        self._assert_close(out_wrap, out_orig, "Decode output mismatch")

    def test_prefill_then_decode_state_consistency(self):
        """Run wrapper prefill then decode; verify state chaining works."""
        batch_size, prefill_len = 2, 4
        mixer = self._create_mixer()
        wrapper = self._create_wrapper(mixer)

        # Prefill
        hidden_prefill = torch.randn(
            batch_size,
            prefill_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        conv_state, recurrent_state = self._make_states(mixer, batch_size)
        _, conv_state, recurrent_state = wrapper(
            hidden_prefill,
            conv_state,
            recurrent_state,
        )

        # Decode using returned states
        hidden_decode = torch.randn(
            batch_size,
            1,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        out_wrap, conv_state2, recurrent_state2 = wrapper(
            hidden_decode,
            conv_state,
            recurrent_state,
        )

        # Just verify shapes and no NaNs
        self.assertEqual(out_wrap.shape, (batch_size, 1, self.config.hidden_size))
        self.assertTrue(torch.isfinite(out_wrap).all())

    # ------------------------------------------------------------------
    # ONNX export tests
    # ------------------------------------------------------------------

    def _export_and_verify(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        input_names: list,
        output_names: list,
        dynamo: bool = False,
    ) -> str:
        """Export to ONNX and return path. Verifies custom ops exist in graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            with torch.inference_mode():
                if dynamo:
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=23,
                        dynamo=True,
                        custom_translation_table=_DYNAMO_TABLE,
                    )
                else:
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=23,
                        custom_opsets={"trt_edgellm": 23},
                        dynamo=False,
                    )

            onnx_model = onnx.load(onnx_path)
            op_types = [node.op_type for node in onnx_model.graph.node]

            self.assertIn(
                "GatedDeltaNetCausalConv1d",
                op_types,
                "ONNX graph missing GatedDeltaNetCausalConv1d custom op",
            )
            self.assertIn(
                "GatedDeltaRule",
                op_types,
                "ONNX graph missing GatedDeltaRule custom op",
            )
            return onnx_path

    def test_onnx_export_torchscript(self):
        """ONNX export via torchscript path."""
        batch_size, seq_len = 1, 4
        mixer = self._create_mixer()
        wrapper = self._create_wrapper(mixer)

        hidden_states = torch.randn(
            batch_size,
            seq_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        conv_state, recurrent_state = self._make_states(mixer, batch_size)

        class ExportWrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, hidden_states, conv_state, recurrent_state):
                return self.layer(hidden_states, conv_state, recurrent_state)

        model = ExportWrapper(wrapper).eval()
        self._export_and_verify(
            model,
            (hidden_states, conv_state, recurrent_state),
            input_names=["hidden_states", "conv_state", "recurrent_state"],
            output_names=["output", "conv_state_out", "recurrent_state_out"],
            dynamo=False,
        )

    @unittest.skipUnless(
        _DYNAMO_CUSTOM_OP_SUPPORTED,
        "Dynamo custom-op translation table not available on this PyTorch version",
    )
    def test_onnx_export_dynamo(self):
        """ONNX export via dynamo path (primary)."""
        batch_size, seq_len = 1, 4
        mixer = self._create_mixer()
        wrapper = self._create_wrapper(mixer)

        hidden_states = torch.randn(
            batch_size,
            seq_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        conv_state, recurrent_state = self._make_states(mixer, batch_size)

        class ExportWrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, hidden_states, conv_state, recurrent_state):
                return self.layer(hidden_states, conv_state, recurrent_state)

        model = ExportWrapper(wrapper).eval()
        self._export_and_verify(
            model,
            (hidden_states, conv_state, recurrent_state),
            input_names=["hidden_states", "conv_state", "recurrent_state"],
            output_names=["output", "conv_state_out", "recurrent_state_out"],
            dynamo=True,
        )
