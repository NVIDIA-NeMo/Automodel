# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Bidirectional context and media-prefix isolation in the Omni encoder."""

import pytest
import torch
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

from nemo_automodel.components.models.d1.encoder import D1OmniEncoder


def tiny_config() -> Lfm2Config:
    return Lfm2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        norm_eps=1e-5,
        conv_L_cache=3,
        conv_bias=False,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=16,
        rope_theta=1e6,
    )


def test_text_is_bidirectional_but_media_cannot_read_text():
    torch.manual_seed(7)
    encoder = D1OmniEncoder(tiny_config()).eval()
    values = torch.randn(1, 6, 64)
    changed = values.clone()
    changed[:, 4:] += torch.randn(1, 2, 64) * 2
    valid = torch.ones(1, 6, dtype=torch.bool)
    media_length = torch.tensor([2])
    before = encoder(values, valid, media_length)
    after = encoder(changed, valid, media_length)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert not torch.allclose(before[:, 2], after[:, 2])
    text_values = values.detach().requires_grad_()
    text_output = encoder(text_values, valid, torch.tensor([0]))
    text_output[:, 0, :8].sum().backward()
    assert torch.isfinite(text_values.grad).all()
    assert text_values.grad[:, -1].abs().sum() > 0


def test_padding_cannot_change_real_encoder_states():
    encoder = D1OmniEncoder(tiny_config()).eval()
    values = torch.randn(1, 4, 64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    prefix = torch.tensor([1])
    original = encoder(values, valid, prefix)
    padded = torch.cat([values, torch.randn(1, 3, 64) * 100], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    actual = encoder(padded, padded_valid, prefix)[:, :4]
    torch.testing.assert_close(original, actual, atol=2e-6, rtol=2e-5)


def test_dtype_moves_preserve_exact_float32_rotary_frequencies():
    encoder = D1OmniEncoder(tiny_config())
    expected = encoder.rotary_emb.inv_freq.clone()
    encoder.to(torch.bfloat16)
    assert encoder.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(encoder.rotary_emb.inv_freq, expected, atol=0, rtol=0)
    encoder.to(torch.float32)
    torch.testing.assert_close(encoder.rotary_emb.inv_freq, expected, atol=0, rtol=0)


def test_encoder_keeps_a_private_typed_configuration() -> None:
    config = tiny_config()
    config._attn_implementation = "eager"
    encoder = D1OmniEncoder(config)
    assert isinstance(encoder.config, Lfm2Config)
    assert encoder.config is not config
    assert encoder.config._attn_implementation == "sdpa"
    assert config._attn_implementation == "eager"


@pytest.mark.parametrize("rope_type", ["linear", "yarn"])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_omni_scaled_rotary_preserves_hf_factors_after_dtype_moves(rope_type: str, device: str) -> None:
    config = tiny_config()
    config.rope_parameters = {
        "rope_type": rope_type,
        "rope_theta": 1000000.0,
        "factor": 4.0,
        **({"attention_factor": 1.5} if rope_type == "yarn" else {}),
    }
    encoder = D1OmniEncoder(config)
    # Omni's publisher builds frequencies on CPU before moving them to the execution device.
    reference = Lfm2RotaryEmbedding(config, device="cpu").to(device)
    positions = torch.tensor([[0, 17, 4095]], device=device)
    for dtype in (torch.float32, torch.bfloat16, torch.float32):
        encoder.to(device=device, dtype=dtype)
        rotary = encoder.rotary_emb
        assert rotary.inv_freq.dtype == torch.float32
        assert rotary.attention_scaling == reference.attention_scaling
        torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
        torch.testing.assert_close(rotary.original_inv_freq, reference.inv_freq, rtol=0, atol=0)
        values = torch.empty(1, 3, config.hidden_size, device=device, dtype=dtype)
        actual = rotary(values, positions)
        expected = reference(values, positions)
        for factors, expected_factors in zip(actual, expected, strict=True):
            torch.testing.assert_close(factors, expected_factors, rtol=0, atol=0)
