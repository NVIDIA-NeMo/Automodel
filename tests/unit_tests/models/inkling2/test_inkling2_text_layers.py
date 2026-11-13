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

"""Inkling 2 layers against per-token loop references written from the model equations."""

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config, Inkling2TextConfig
from nemo_automodel.components.models.inkling2.layers import Inkling2MoE, inkling2_softcapped_swiglu
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.text import (
    Inkling2Attention,
    Inkling2BlockAttentionResidual,
    Inkling2DynamicShortConvolution,
    Inkling2ShortConvolution,
    _l2_clip,
)

from .parity_check_inkling2 import TINY_TEXT_CONFIG, _randomize

pytestmark = pytest.mark.cpu_only

_BACKEND = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_short_convolution_matches_a_causal_loop_with_padding() -> None:
    batch, sequence, channels, kernel = 2, 7, 5, 4
    conv = Inkling2ShortConvolution(channels, kernel)
    with torch.no_grad():
        conv.conv1d.weight.copy_(torch.randn(channels, 1, kernel, generator=_generator()))
    x = torch.randn(batch, sequence, channels, generator=_generator(1))
    mask = torch.ones(batch, sequence, dtype=torch.long)
    mask[1, :2] = 0

    expected = torch.empty_like(x)
    masked = x * mask[:, :, None]
    taps = conv.conv1d.weight[:, 0, :]  # [channels, kernel]; tap kernel - 1 weights the current token
    for t in range(sequence):
        total = torch.zeros(batch, channels)
        for j in range(kernel):
            source = t - (kernel - 1) + j
            if source >= 0:
                total += taps[:, j] * masked[:, source]
        expected[:, t] = (total + x[:, t]) / math.sqrt(kernel + 1)

    torch.testing.assert_close(conv(x, mask), expected, rtol=1e-6, atol=1e-6)


def test_dynamic_short_convolution_matches_a_per_head_loop() -> None:
    batch, sequence, heads, head_dim, hidden, kernel = 2, 6, 2, 3, 4, 3
    channels = heads * head_dim
    conv = Inkling2DynamicShortConvolution(channels, hidden, heads, kernel, _BACKEND, dtype=torch.float32)
    with torch.no_grad():
        conv.static_weight.copy_(torch.randn(kernel, channels, generator=_generator()))
        conv.weight_proj.weight.copy_(torch.randn(heads * kernel, hidden, generator=_generator(1)))
    x = torch.randn(batch, sequence, channels, generator=_generator(2))
    dynamic_input = torch.randn(batch, sequence, hidden, generator=_generator(3))
    mask = torch.ones(batch, sequence, dtype=torch.long)
    mask[0, 0] = 0

    dynamic = (dynamic_input @ conv.weight_proj.weight.T).view(batch, sequence, heads, kernel)
    masked = x * mask[:, :, None]
    expected = torch.empty_like(x)
    for t in range(sequence):
        for c in range(channels):
            head = c // head_dim
            total = torch.zeros(batch)
            for lag in range(kernel):
                if t - lag >= 0:
                    tap = conv.static_weight[lag, c] + dynamic[:, t, head, lag]
                    total += tap * masked[:, t - lag, c]
            expected[:, t, c] = (total + x[:, t, c]) / math.sqrt(2 * kernel + 1)

    torch.testing.assert_close(conv(x, dynamic_input, mask), expected, rtol=1e-6, atol=1e-6)


def test_block_attention_residual_is_a_softmax_over_normalized_depth() -> None:
    hidden = 8
    module = Inkling2BlockAttentionResidual(hidden, num_queries=3)
    block_states = torch.randn(2, 5, 2, hidden, generator=_generator()).to(torch.bfloat16)
    running = torch.randn(2, 5, hidden, generator=_generator(1))

    # Zero queries weight every depth slot equally.
    outputs = module(block_states, running)
    stacked = torch.cat([block_states.float(), running[:, :, None]], dim=-2)
    normalized = stacked * torch.rsqrt(stacked.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert len(outputs) == 3 and all(out.dtype == torch.float32 for out in outputs)
    torch.testing.assert_close(outputs[0], normalized.mean(-2), rtol=1e-6, atol=1e-6)

    with torch.no_grad():
        module.query.copy_(torch.randn(3, hidden, generator=_generator(2)))
    outputs = module(block_states, running)
    for index, query in enumerate(module.query):
        weights = torch.softmax((normalized @ query) / math.sqrt(hidden), dim=-1)
        torch.testing.assert_close(outputs[index], (weights[..., None] * normalized).sum(-2), rtol=1e-5, atol=1e-6)


def test_l2_clip_caps_long_vectors_and_keeps_short_ones() -> None:
    vectors = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    clipped = _l2_clip(vectors, threshold=1.0)
    torch.testing.assert_close(clipped[0].norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(clipped[1], vectors[1])


def _attention_inputs(batch: int = 2, heads: int = 4, kv_heads: int = 2, sequence: int = 6, head_dim: int = 8):
    generator = _generator(5)
    query = torch.randn(batch, heads, sequence, head_dim, generator=generator)
    key = torch.randn(batch, kv_heads, sequence, head_dim, generator=generator)
    value = torch.randn(batch, kv_heads, sequence, head_dim, generator=generator)
    bias = torch.randn(batch, heads, sequence, sequence, generator=generator)
    mask = torch.ones(sequence, sequence, dtype=torch.bool).tril()[None, None]
    return query, key, value, bias, mask


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_attention_sink_absorbs_probability_mass(attn: str) -> None:
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG)
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    attention = Inkling2Attention(config, 0, backend, dtype=torch.float32)
    attend = attention._sdpa_attention if attn == "sdpa" else attention._eager_attention
    query, key, value, bias, mask = _attention_inputs()
    key_rep, value_rep = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
    scores = (query @ key_rep.transpose(-1, -2)) * attention.scaling + bias
    scores = scores.masked_fill(~mask, float("-inf"))

    # A very negative sink leaves ordinary softmax attention.
    no_sink = attend(query, key, value, bias, torch.full((2, 4, 6, 1), -1e4), mask, 0.0)
    torch.testing.assert_close(no_sink, torch.softmax(scores, -1) @ value_rep, rtol=1e-5, atol=1e-5)
    # A finite sink s rescales the output by sum(exp(scores)) / (sum(exp(scores)) + exp(s)).
    sinks = torch.randn(2, 4, 6, 1, generator=_generator(6))
    with_sink = attend(query, key, value, bias, sinks, mask, 0.0)
    log_mass = torch.logsumexp(scores, -1, keepdim=True)
    torch.testing.assert_close(with_sink, no_sink * torch.sigmoid(log_mass - sinks), rtol=1e-5, atol=1e-5)


def test_softcapped_swiglu_reads_interleaved_gate_and_up() -> None:
    gate = torch.randn(3, 4, generator=_generator()) * 10
    up = torch.randn(3, 4, generator=_generator(1)) * 50
    interleaved = torch.stack([gate, up], dim=-1).reshape(3, 8)
    weights = torch.rand(3, 1, generator=_generator(2))

    out = inkling2_softcapped_swiglu(interleaved, weights, gate_softcap=4.0, up_softcap=25.0)
    expected = torch.tanh(gate / 4) * 4 * torch.sigmoid(gate) * torch.tanh(up / 25) * 25 * weights
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    assert out.abs().max() <= 4 * 25


def test_moe_matches_a_per_token_reference_on_the_released_layout() -> None:
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG, torch_dtype="float32")
    moe = Inkling2MoE(config, _BACKEND)
    _randomize(moe, seed=4)
    x = torch.randn(10, config.hidden_size, generator=_generator(7))
    padding_mask = torch.zeros(10, dtype=torch.bool)
    padding_mask[3] = True

    out = moe(x, padding_mask)

    w13 = moe.experts.gate_and_up_projs.detach().transpose(-1, -2)  # released [E, 2I, H], gate rows interleaved
    w2 = moe.experts.down_projs.detach().transpose(-1, -2)  # released [E, H, I]
    scores = torch.sigmoid(x @ moe.gate.weight.T)
    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        if padding_mask[token]:
            continue
        chosen = torch.topk(scores[token] + moe.gate.e_score_correction_bias, config.num_experts_per_tok).indices
        weights = scores[token, chosen] / scores[token, chosen].sum() * config.route_scale
        for weight, expert in zip(weights, chosen):
            gate, up = w13[expert, 0::2] @ x[token], w13[expert, 1::2] @ x[token]
            capped_gate = torch.tanh(gate / config.mlp_gate_softcap) * config.mlp_gate_softcap
            capped_up = torch.tanh(up / config.mlp_up_softcap) * config.mlp_up_softcap
            expected[token] += weight * (w2[expert] @ (capped_gate * torch.sigmoid(gate) * capped_up))

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    assert torch.all(out[3] == 0)


def test_fp32_input_casts_around_fp32_modules_keep_model_dtype_numerics() -> None:
    # FSDP units for the FP32 modules cast their floating inputs to FP32 before forward.
    config = Inkling2Config(text_config=TINY_TEXT_CONFIG, vision_config=None, torch_dtype="bfloat16")
    model = Inkling2ForConditionalGeneration(config, backend=_BACKEND)
    _randomize(model, seed=1)
    input_ids = torch.randint(2, 250, (2, 12), generator=_generator())
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    def cast_inputs(module: torch.nn.Module, args: tuple) -> tuple:
        return tuple(arg.float() if torch.is_tensor(arg) and arg.is_floating_point() else arg for arg in args)

    for name, module in model.named_modules():
        if isinstance(module, (Inkling2ShortConvolution, Inkling2BlockAttentionResidual)) or name.endswith("mlp.gate"):
            module.register_forward_pre_hook(cast_inputs)
    with torch.no_grad():
        cast = model(input_ids=input_ids).logits
    torch.testing.assert_close(cast, expected, rtol=0, atol=0)


def test_router_state_stays_fp32_in_a_bf16_moe() -> None:
    # The router weight and correction bias are FP32 even in a BF16 model; selection runs in FP32.
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG, torch_dtype="bfloat16")
    moe = Inkling2MoE(config, _BACKEND)
    assert moe.gate.weight.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32
    assert moe.gate.gate_precision == torch.float32
    assert moe.experts.gate_and_up_projs.dtype == torch.bfloat16
    x = torch.randn(4, config.hidden_size).to(torch.bfloat16)
    assert moe(x).dtype == torch.bfloat16


def test_router_selects_on_fp32_scores_with_bf16_activations() -> None:
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG)
    moe = Inkling2MoE(config, _BACKEND)
    _randomize(moe, seed=9)
    x = torch.randn(6, config.hidden_size, generator=_generator(3)).to(torch.bfloat16)
    weights, indices, _ = moe.gate(x, torch.ones(6, dtype=torch.bool), None)
    scores = torch.sigmoid(F.linear(x.float(), moe.gate.weight))
    expected = torch.topk(scores + moe.gate.e_score_correction_bias, config.num_experts_per_tok).indices
    assert weights.dtype == torch.float32
    assert torch.equal(indices.sort(-1).values, expected.sort(-1).values)
    torch.testing.assert_close(weights.sum(-1), torch.full((6,), config.route_scale), rtol=1e-6, atol=1e-6)
