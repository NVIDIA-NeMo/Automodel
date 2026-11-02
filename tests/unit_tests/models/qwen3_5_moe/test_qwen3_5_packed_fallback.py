# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU coverage for Qwen's refusal to silently drop packed boundaries."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_qwen
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.qwen3_5_moe import cp_linear_attn

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def fallback_module(monkeypatch: pytest.MonkeyPatch) -> cp_linear_attn.CPAwareGatedDeltaNet:
    """Construct a tiny native module with both optional kernels unavailable."""
    # Hide the optional packages at construction rather than treating a
    # non-None decorated HF fallback as proof that its kernel is installed.
    monkeypatch.setitem(sys.modules, "causal_conv1d", None)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", None)
    # HF's decorator resolves dependencies at import time. Expose its actual
    # torch reference so CPU controls do not call a previously cached CUDA op.
    monkeypatch.setattr(
        cp_linear_attn, "torch_chunk_gated_delta_rule", inspect.unwrap(hf_qwen.torch_chunk_gated_delta_rule)
    )
    monkeypatch.setattr(cp_linear_attn, "causal_conv1d_fn", inspect.unwrap(hf_qwen.causal_conv1d_fn))
    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["linear_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        use_cache=False,
    )
    torch.manual_seed(123)
    return cp_linear_attn.CPAwareGatedDeltaNet(config, layer_idx=0)


@pytest.mark.parametrize(
    "missing_kernel,distribution",
    [("convolution", "causal-conv1d"), ("recurrence", "flash-linear-attention")],
)
@pytest.mark.parametrize("boundaries", [[0, 8], [0, 3, 8]])
def test_missing_kernel_rejects_boundaries_before_computation(
    fallback_module: cp_linear_attn.CPAwareGatedDeltaNet,
    missing_kernel: str,
    distribution: str,
    boundaries: list[int],
) -> None:
    """Reject boundary metadata before work starts when either kernel is missing."""
    if missing_kernel == "convolution":
        fallback_module.chunk_gated_delta_rule = MagicMock()
    else:
        fallback_module.causal_conv1d_fn = MagicMock()
    # A rejected forward must not perform projections or mutate a cache.
    projection = MagicMock(side_effect=AssertionError("packed fallback reached a projection"))
    fallback_module.in_proj_qkv.forward = projection
    hidden_states = torch.randn(1, 8, 16)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    with pytest.raises(ValueError, match=f"packed sequences.*{distribution}"):
        fallback_module(
            hidden_states,
            attention_mask=torch.ones(1, 8, dtype=torch.int32),
            cu_seqlens=cu_seqlens,
            indices=torch.arange(8),
        )
    projection.assert_not_called()


@pytest.mark.parametrize("with_padding", [False, True])
def test_unpacked_torch_fallback_forward_backward(
    fallback_module: cp_linear_attn.CPAwareGatedDeltaNet, with_padding: bool
) -> None:
    """Keep ordinary and padded unpacked CPU forward/backward operational."""
    hidden_states = torch.randn(2, 8, 16, requires_grad=True)
    attention_mask = None
    if with_padding:
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    output = fallback_module(hidden_states, attention_mask=attention_mask)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    output.backward(torch.randn_like(output))
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.abs().sum() > 0
    assert torch.isfinite(fallback_module.conv1d.weight.grad).all()
    assert fallback_module.conv1d.weight.grad.abs().sum() > 0
    assert torch.isfinite(fallback_module.in_proj_qkv.weight.grad).all()


def test_available_kernel_dispatch_preserves_boundaries(
    fallback_module: cp_linear_attn.CPAwareGatedDeltaNet,
) -> None:
    """Pass document boundaries through when kernel callables are installed."""
    # This checks dispatch only; CUDA kernel coverage lives in test_cp_linear_attn.py.
    convolution = MagicMock(wraps=inspect.unwrap(hf_qwen.causal_conv1d_fn))
    recurrence = MagicMock(wraps=inspect.unwrap(hf_qwen.torch_chunk_gated_delta_rule))
    fallback_module.causal_conv1d_fn = convolution
    fallback_module.chunk_gated_delta_rule = recurrence
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2]], dtype=torch.int32)
    output = fallback_module(
        torch.randn(1, 8, 16),
        attention_mask=mask,
        cu_seqlens=cu_seqlens,
        indices=torch.arange(8),
    )
    assert output.shape == (1, 8, 16)
    torch.testing.assert_close(convolution.call_args.kwargs["seq_idx"], mask)
    assert recurrence.call_args.kwargs["cu_seqlens"] is cu_seqlens
