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

"""Tests for the gated-LoRA multi-token-prediction layout."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules, lora_token_gate
from nemo_automodel.components.attention.gated_mtp_mask import create_gated_mtp_sdpa_mask
from nemo_automodel.components.speculative.gated_mtp import GatedMTPConfig, GatedMTPSampler

_VOCAB = 64
_MASK_IDS = [61, 62, 63]


def _build_lora_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).eval()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*_proj"], dim=4, alpha=8, use_triton=False))
    # Non-zero lora_B so that the adapter changes the mask tokens' hidden states.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    return model


def _layout_forward(model, layout, input_ids):
    """Run the dense gated-MTP forward.

    Args:
        input_ids: Tensor of shape [batch, sequence].

    Returns:
        The :class:`GatedMTPOutput` of the forward.
    """
    inputs = layout.prepare_inputs(input_ids)
    attention_mask = create_gated_mtp_sdpa_mask(layout.num_depths, torch.ones_like(input_ids), dtype=torch.float32)
    with lora_token_gate(model, inputs.lora_gate):
        out = model(
            input_ids=inputs.input_ids,
            position_ids=inputs.position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    return layout.split_output(out.logits, out.hidden_states[-1])


def test_prepare_inputs_appends_one_mask_token_per_depth_and_anchor():
    layout = GatedMTPConfig(mask_token_ids=[7, 8]).build()
    input_ids = torch.tensor([[1, 2, 3]])
    position_ids = torch.tensor([[5, 6, 7]])

    inputs = layout.prepare_inputs(input_ids, position_ids)

    # depth-major: [clean | depth 1 for every anchor | depth 2 for every anchor]
    torch.testing.assert_close(inputs.input_ids, torch.tensor([[1, 2, 3, 7, 7, 7, 8, 8, 8]]))
    torch.testing.assert_close(inputs.position_ids, torch.tensor([[5, 6, 7, 6, 7, 8, 7, 8, 9]]))
    torch.testing.assert_close(inputs.lora_gate, torch.tensor([[False] * 3 + [True] * 6]))


def test_split_output_returns_clean_copy_and_per_depth_mask_states():
    layout = GatedMTPConfig(mask_token_ids=[7, 8]).build()
    hidden_states = torch.arange(2 * 9 * 4, dtype=torch.float32).view(2, 9, 4)
    logits = torch.randn(2, 9, 11)

    out = layout.split_output(logits, hidden_states)

    torch.testing.assert_close(out.hidden_states, hidden_states[:, :3])
    torch.testing.assert_close(out.logits, logits[:, :3])
    assert len(out.mtp_per_depth_h) == 2
    torch.testing.assert_close(out.mtp_per_depth_h[0], hidden_states[:, 3:6])
    torch.testing.assert_close(out.mtp_per_depth_h[1], hidden_states[:, 6:9])
    assert out.mtp_per_depth_logits is None
    # the losses flatten these with ``view``, which needs contiguous tensors for batch > 1
    assert all(t.is_contiguous() for t in [out.logits, out.hidden_states, *out.mtp_per_depth_h])
    # logits of a trailing subset (``logits_to_keep``) are passed through
    assert layout.split_output(logits[:, -1:], hidden_states).logits.shape == (2, 1, 11)
    with pytest.raises(ValueError, match="not a gated-MTP layout"):
        layout.split_output(None, hidden_states[:, :8])


@pytest.mark.parametrize("kwargs", [{"mask_token_ids": []}, {"mask_token_ids": [1], "scaling_factor": 0.0}])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError, match="gated_mtp"):
        GatedMTPConfig(**kwargs).build()


def test_clean_copy_is_the_base_model():
    model = _build_lora_model()
    layout = GatedMTPConfig(mask_token_ids=_MASK_IDS).build()
    input_ids = torch.randint(0, 60, (2, 6))

    with torch.no_grad():
        out = _layout_forward(model, layout, input_ids)
        with lora_token_gate(model, torch.zeros_like(input_ids, dtype=torch.bool)):
            base = model(input_ids=input_ids, output_hidden_states=True)

    torch.testing.assert_close(out.hidden_states, base.hidden_states[-1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.logits, base.logits, rtol=1e-5, atol=1e-5)


def test_mtp_depths_match_appending_the_mask_tokens_after_each_anchor():
    # Reference: for every anchor t and depth d, a plain causal forward over x_0..x_t followed by the first d mask
    # tokens, with the adapter on the mask tokens only. The dense layout must reproduce it for all (t, d) at once.
    model = _build_lora_model()
    layout = GatedMTPConfig(mask_token_ids=_MASK_IDS).build()
    seq_len = 5
    input_ids = torch.randint(0, 60, (1, seq_len))

    with torch.no_grad():
        out = _layout_forward(model, layout, input_ids)
        for t in range(seq_len):
            for d in range(1, layout.num_depths + 1):
                ids = torch.cat([input_ids[:, : t + 1], torch.tensor([_MASK_IDS[:d]])], dim=1)
                position_ids = torch.cat([torch.arange(t + 1), t + torch.arange(1, d + 1)]).unsqueeze(0)
                gate = torch.tensor([[False] * (t + 1) + [True] * d])
                with lora_token_gate(model, gate):
                    expected = model(input_ids=ids, position_ids=position_ids, output_hidden_states=True)
                torch.testing.assert_close(
                    out.mtp_per_depth_h[d - 1][:, t], expected.hidden_states[-1][:, -1], rtol=1e-5, atol=1e-5
                )

        # The adapter is what turns the mask tokens into drafts: without it, the depths differ.
        with lora_token_gate(model, torch.zeros(1, seq_len * (1 + layout.num_depths), dtype=torch.bool)):
            inputs = layout.prepare_inputs(input_ids)
            attention_mask = create_gated_mtp_sdpa_mask(
                layout.num_depths, torch.ones_like(input_ids), dtype=torch.float32
            )
            ungated = model(
                input_ids=inputs.input_ids,
                position_ids=inputs.position_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
    assert not torch.allclose(out.mtp_per_depth_h[0], ungated[:, seq_len : 2 * seq_len])


def test_sampler_is_a_no_op_at_init_and_learns_after_one_step():
    """The sampler's zero-initialized second layer must make it an exact identity at
    construction, and an ordinary gradient step must break that symmetry."""
    sampler = GatedMTPSampler(hidden_size=8, num_depths=3)
    h = torch.randn(2, 5, 8)
    prev_emb = torch.randn(2, 5, 8)

    out = sampler(h, prev_emb)
    assert out.shape == h.shape
    torch.testing.assert_close(out, h)

    before = {name: p.detach().clone() for name, p in sampler.named_parameters()}
    out.sum().backward()
    opt = torch.optim.SGD(sampler.parameters(), lr=0.1)
    opt.step()
    after_out = sampler(h, prev_emb)

    assert not torch.allclose(after_out, h)
    assert any(not torch.allclose(before[name], p) for name, p in sampler.named_parameters())
