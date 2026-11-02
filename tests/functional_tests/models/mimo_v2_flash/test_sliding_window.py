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

"""MiMo attention-module parity, including the TE inclusive left-window contract."""

import json
from collections.abc import Callable

import pytest
import torch
from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import MiMoV2FlashConfig as HFConfig
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import MiMoV2FlashAttention as HFAttention

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashAttention


def _compare(actual: torch.Tensor, expected: torch.Tensor, atol: float) -> dict[str, float]:
    """Check BF16 module results against an FP32 reference.

    Args:
        actual: Tensor of arbitrary shape containing an output or gradient.
        expected: Reference tensor with the same shape and axis order.
        atol: Maximum absolute error allowed for the compared tensor.

    Returns:
        Maximum absolute error and relative L2 error over all elements.
    """
    actual = actual.float()
    expected = expected.float()
    delta = actual - expected
    metrics = {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
    }
    assert torch.isfinite(actual).all()
    assert metrics["max_abs"] < atol, metrics
    assert metrics["relative_l2"] < 0.02, metrics
    return metrics


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("is_swa", [False, True])
def test_te_window_attention_matches_reference(
    layout: str, is_swa: bool, record_property: Callable[[str, str], None]
) -> None:
    torch.manual_seed(20261013)
    config = MiMoV2FlashConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        v_head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        swa_v_head_dim=64,
        partial_rotary_factor=0.5,
        hybrid_layer_pattern=[int(is_swa)],
        sliding_window=128,
        sliding_window_size=128,
        attention_value_scale=0.707,
        torch_dtype="bfloat16",
    )
    assert config.sliding_window == 129
    backend = BackendConfig(linear="torch", attn="te", rms_norm="torch", rope_fusion=False)
    attention = MiMoV2FlashAttention(config, backend, is_swa, 0).cuda()
    reference_config = HFConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        v_head_dim=64,
        layer_types=["sliding_attention" if is_swa else "full_attention"],
        sliding_window=129,
        attention_value_scale=0.707,
        attn_implementation="eager",
    )
    reference = HFAttention(reference_config, 0).cuda().float()
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weight = getattr(reference, name).weight
            weight.normal_(std=0.02)
            # Both modules see identical BF16-representable weights.
            weight.copy_(weight.to(torch.bfloat16).float())
            getattr(attention, name).weight.copy_(weight)
        if is_swa:
            reference.sinks.zero_()
            reference.sinks.requires_grad_(False)
            attention.attn_module.softmax_offset.zero_()
            assert attention.attn_module.softmax_offset.dtype == torch.float32

    lengths = [257] if layout == "bshd" else [129, 257]
    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    total = sum(lengths)
    hidden = torch.randn(1, total, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().float().requires_grad_(True)
    inverse_frequency = 10000.0 ** (-torch.arange(0, 32, 2, device="cuda").float() / 32)
    angles = positions.float()[:, None] * inverse_frequency[None, :]
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(0)
    rope = (angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16))

    # Independent official/SGLang semantics: 128 previous keys plus query,
    # within each document. No HF or AM mask helper is the oracle.
    reference_mask = torch.full((1, 1, total, total), float("-inf"), device="cuda")
    offset = 0
    for length in lengths:
        for query in range(length):
            left = max(0, query - 128) if is_swa else 0
            reference_mask[:, :, offset + query, offset + left : offset + query + 1] = 0
        offset += length
    expected_counts = positions.clamp_max(128) + 1 if is_swa else positions + 1
    torch.testing.assert_close((reference_mask[0, 0] == 0).sum(-1), expected_counts)

    if layout == "thd":
        cumulative = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
        actual = attention(
            hidden,
            position_embeddings=rope,
            qkv_format="thd",
            cu_seqlens=cumulative,
            cu_seqlens_padded=cumulative,
            max_seqlen=max(lengths),
        )[0]
    else:
        actual = attention(hidden, position_embeddings=rope)[0]
    expected = reference(reference_hidden, position_embeddings=rope, attention_mask=reference_mask)[0]
    metrics = {
        "output_before_boundary": _compare(actual[:, positions < 128], expected[:, positions < 128], 0.002),
        "output_at_after_boundary": _compare(actual[:, positions >= 128], expected[:, positions >= 128], 0.002),
    }
    upstream_gradient = torch.randn_like(actual)
    actual.backward(upstream_gradient)
    expected.backward(upstream_gradient.float())
    metrics["input_gradient"] = _compare(hidden.grad, reference_hidden.grad, 0.002)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        metrics[name + "_gradient"] = _compare(
            getattr(attention, name).weight.grad, getattr(reference, name).weight.grad, 0.02
        )
    record_property("layout", layout)
    record_property("is_swa", str(is_swa))
    record_property("lengths", json.dumps(lengths))
    record_property("metrics", json.dumps(metrics))
