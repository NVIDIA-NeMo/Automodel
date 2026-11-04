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

"""Focused tests for Qwen3.8-Flash-Next's model-specific GDN output gate."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextGatedDeltaNet,
    Qwen3_8_FlashNextRMSNormGated,
)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_gated_delta_net_constructs_without_explicit_layer_types(device: str) -> None:
    config = Qwen3_8_FlashNextTextConfig(
        hidden_size=16,
        num_hidden_layers=4,
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )

    # Exercise the real Transformers constructor, not a mocked base class.
    with torch.device(device):
        layer = Qwen3_8_FlashNextGatedDeltaNet(config, layer_idx=0)

    assert layer.layer_idx == 0
    assert layer.in_proj_qkv.weight.shape == (48, 16)
    assert layer.in_proj_qkv.weight.device.type == device


def test_qwen3_8_flash_next_gdn_norm_uses_sigmoid_gate() -> None:
    norm = Qwen3_8_FlashNextRMSNormGated(hidden_size=4, eps=1e-6, activation="sigmoid", dtype=torch.bfloat16)
    norm.weight.detach().copy_(torch.tensor([0.75, 1.0, 1.25, 1.5], dtype=torch.bfloat16))
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 0.25, -0.5, -0.25]], dtype=torch.bfloat16)
    gate = torch.tensor([[-3.0, -1.0, 1.0, 3.0], [2.0, -2.0, 0.5, -0.5]], dtype=torch.bfloat16)

    actual = norm(hidden_states, gate)
    normalized = hidden_states.float() * torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (norm.weight.float() * normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    silu_result = (norm.weight.float() * normalized * F.silu(gate.float())).to(torch.bfloat16)
    assert not torch.allclose(actual, silu_result)


def test_qwen3_8_flash_next_gdn_norm_reset_uses_multiplicative_identity() -> None:
    norm = Qwen3_8_FlashNextRMSNormGated(hidden_size=8)
    norm.weight.detach().zero_()
    norm.reset_parameters()

    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight), rtol=0, atol=0)
