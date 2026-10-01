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

"""Unit tests for the native HunyuanImage-3.0 layers and model.

The attention test checks the native layer against a from-scratch reference of the release semantics: fused QKV
grouped per KV head, RoPE before the per-head QK RMSNorm, and a text-causal / image-bidirectional mask. The image-side
modules come from the checkpoint's remote code, so the model tests replace them with small stand-ins.
"""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.flow_matching.adapters.hunyuan_image3 import text_causal_image_bidirectional_mask
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hunyuan_image3.config import HunyuanImage3Config, per_layer
from nemo_automodel.components.models.hunyuan_image3.layers import (
    HunyuanImage3Attention,
    HunyuanImage3Block,
    HunyuanImage3MLP,
    HunyuanImage3RMSNorm,
)
from nemo_automodel.components.models.hunyuan_image3.model import HunyuanImage3ForCausalMM, ModelClass
from nemo_automodel.components.models.hunyuan_image3.rope import build_2d_positions, build_2d_rope_cos_sin
from nemo_automodel.components.moe.layers import MoE

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
LATENT_CHANNELS = 4
VOCAB = 64


@pytest.fixture
def config():
    return HunyuanImage3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        attention_head_dim=HEAD_DIM,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_experts=4,
        moe_topk=2,
        patch_embed_hidden_dim=8,
        vae={"latent_channels": LATENT_CHANNELS},
        remote_code_dir="/unused",
        torch_dtype="float32",
    )


@pytest.fixture
def backend():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.proj(t.float().reshape(-1, 1) / 1000)


class _PatchEmbed(nn.Module):
    """Stand-in for the release UNetDown: ``[B, C, h, w]`` latents -> ``[B, h*w, hidden]`` tokens."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(LATENT_CHANNELS, hidden_size)

    def forward(self, x, temb):
        _, _, h, w = x.shape
        return self.proj(x.flatten(2).transpose(1, 2)) + temb[:, None], h, w


class _FinalLayer(nn.Module):
    """Stand-in for the release UNetUp: ``[B, h*w, hidden]`` -> ``[B, C, h, w]``."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, LATENT_CHANNELS)

    def forward(self, x, temb, h, w):
        out = self.proj(x + temb[:, None])
        return out.transpose(1, 2).reshape(x.shape[0], LATENT_CHANNELS, h, w)


def _stub_image_modules(config, include_extra):
    hidden = config.hidden_size
    return {
        "timestep_emb": _TimestepEmbedder(hidden),
        "patch_embed": _PatchEmbed(hidden),
        "time_embed": _TimestepEmbedder(hidden),
        "final_layer": _FinalLayer(hidden),
        "time_embed_2": _TimestepEmbedder(hidden),
    }


@pytest.fixture
def model(config, backend):
    torch.manual_seed(0)
    with patch("nemo_automodel.components.models.hunyuan_image3.model.build_image_modules", _stub_image_modules):
        m = HunyuanImage3ForCausalMM(config, backend=backend)
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return m.eval()


def _reference_attention(attn: HunyuanImage3Attention, x, positions, mask):
    """Release semantics written out per head."""
    bsz, seq_len, _ = x.shape
    groups = HEADS // KV_HEADS
    qkv = x @ attn.qkv_proj.weight.T  # [B, S, KV * (groups + 2) * D], laid out [q * groups, k, v] per KV head
    cos, sin = build_2d_rope_cos_sin(positions, HEAD_DIM, 10000.0)

    def rope(t):
        t1, t2 = t[..., : HEAD_DIM // 2], t[..., HEAD_DIM // 2 :]
        return t * cos + torch.cat([-t2, t1], dim=-1) * sin

    def rms(t, weight):
        return weight * t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)

    heads = []
    for h in range(HEADS):
        kv, g = divmod(h, groups)
        base = kv * (groups + 2) * HEAD_DIM
        q = qkv[..., base + g * HEAD_DIM : base + (g + 1) * HEAD_DIM]
        k = qkv[..., base + groups * HEAD_DIM : base + (groups + 1) * HEAD_DIM]
        v = qkv[..., base + (groups + 1) * HEAD_DIM : base + (groups + 2) * HEAD_DIM]
        q = rms(rope(q), attn.query_layernorm.weight)
        k = rms(rope(k), attn.key_layernorm.weight)
        scores = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        scores = scores.masked_fill(~mask, float("-inf"))
        heads.append(scores.softmax(-1) @ v)
    return torch.cat(heads, dim=-1) @ attn.o_proj.weight.T


class TestLayers:
    def test_per_layer(self):
        assert per_layer(3, 5) == 3
        assert per_layer([1, 2, 3], 1) == 2

    def test_rmsnorm(self):
        norm = HunyuanImage3RMSNorm(8, 1e-5)
        nn.init.normal_(norm.weight)
        x = torch.randn(2, 8)
        torch.testing.assert_close(norm(x), norm.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))

    def test_mlp_activation_is_on_second_half(self, backend):
        mlp = HunyuanImage3MLP(8, 4, backend).float()
        x = torch.randn(3, 8)
        up_w, gate_w = mlp.gate_and_up_proj.weight.chunk(2, dim=0)
        expected = (x @ up_w.T * F.silu(x @ gate_w.T)) @ mlp.down_proj.weight.T
        torch.testing.assert_close(mlp(x), expected)

    @pytest.mark.parametrize("image_block", [None, (2, 2, 3)])
    def test_attention_matches_reference(self, config, backend, image_block):
        torch.manual_seed(0)
        attn = HunyuanImage3Attention(config, backend).float()
        nn.init.normal_(attn.query_layernorm.weight, mean=1.0, std=0.1)
        nn.init.normal_(attn.key_layernorm.weight, mean=1.0, std=0.1)
        seq_len = 10
        x = torch.randn(2, seq_len, HIDDEN)
        if image_block is None:
            positions = build_2d_positions(seq_len, [])
            native_mask = None
            ref_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        else:
            start, h, w = image_block
            positions = build_2d_positions(seq_len, [image_block])
            native_mask = text_causal_image_bidirectional_mask(seq_len, start, h * w, x.device)
            ref_mask = native_mask[0, 0]
            assert ref_mask[start, start + h * w - 1] and not ref_mask[start - 1, start]
        positions = positions[None].expand(2, -1, -1)
        cos, sin = build_2d_rope_cos_sin(positions, HEAD_DIM, config.rope_theta)
        torch.testing.assert_close(
            attn(x, cos, sin, native_mask), _reference_attention(attn, x, positions, ref_mask), rtol=1e-4, atol=1e-5
        )

    def test_block_adds_shared_expert_to_routed_experts(self, model):
        block = model.model.layers["0"]
        assert isinstance(block, HunyuanImage3Block) and isinstance(block.mlp, MoE)
        assert block.shared_mlp.down_proj.in_features == 16
        x = torch.randn(1, 5, HIDDEN)
        cos, sin = build_2d_rope_cos_sin(torch.zeros(1, 5, 2), HEAD_DIM, 10000.0)
        with torch.no_grad():
            with patch.object(block.self_attn, "forward", lambda x, cos, sin, mask: torch.zeros_like(x)):
                out = block(x, cos=cos, sin=sin)
            h = block.post_attention_layernorm(x)
            expected = x + block.shared_mlp(h) + block.mlp(h, None)
        torch.testing.assert_close(out, expected)


def _gen_image_inputs(token_h=2, token_w=3, text=4):
    seq_len = text + 1 + token_h * token_w + 2  # prompt, timestep slot, image block, trailing tokens
    image_start = text + 1
    input_ids = torch.randint(0, VOCAB, (1, seq_len))
    image_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    image_mask[0, image_start : image_start + token_h * token_w] = True
    return {
        "input_ids": input_ids,
        "rope_positions": build_2d_positions(seq_len, [(image_start, token_h, token_w)])[None],
        "attention_mask": text_causal_image_bidirectional_mask(seq_len, image_start, token_h * token_w, "cpu"),
        "mode": "gen_image",
        "images": torch.randn(1, LATENT_CHANNELS, token_h, token_w),
        "timestep": torch.tensor([500.0]),
        "image_mask": image_mask,
        "timestep_scatter_index": torch.tensor([[text]]),
    }


class TestModel:
    def test_model_class_and_capabilities(self):
        assert ModelClass is HunyuanImage3ForCausalMM
        caps = HunyuanImage3ForCausalMM.ModelCapabilities()
        assert caps.supports_ep and not (caps.supports_tp or caps.supports_cp or caps.supports_pp)

    def test_gen_text_logits_are_causal(self, model):
        ids = torch.randint(0, VOCAB, (2, 7))
        positions = build_2d_positions(7, [])[None].expand(2, -1, -1)
        with torch.no_grad():
            logits = model(ids, rope_positions=positions).logits
            changed = ids.clone()
            changed[:, -1] = (changed[:, -1] + 1) % VOCAB
            logits_changed = model(changed, rope_positions=positions).logits
        assert logits.shape == (2, 7, VOCAB) and logits.dtype == torch.float32
        torch.testing.assert_close(logits[:, :-1], logits_changed[:, :-1])
        assert not torch.allclose(logits[:, -1], logits_changed[:, -1])

    def test_gen_image_predicts_latent_shaped_velocity(self, model):
        inputs = _gen_image_inputs()
        with torch.no_grad():
            pred = model(**inputs).diffusion_prediction
        assert pred.shape == inputs["images"].shape

    def test_gen_image_depends_on_noisy_latents_and_timestep(self, model):
        inputs = _gen_image_inputs()
        with torch.no_grad():
            base = model(**inputs).diffusion_prediction
            other_t = model(**{**inputs, "timestep": torch.tensor([100.0])}).diffusion_prediction
            other_x = model(**{**inputs, "images": torch.randn_like(inputs["images"])}).diffusion_prediction
        assert not torch.allclose(base, other_t)
        assert not torch.allclose(base, other_x)

    def test_gen_image_ignores_text_after_the_image(self, model):
        inputs = _gen_image_inputs()
        changed = inputs["input_ids"].clone()
        changed[0, -1] = (changed[0, -1] + 1) % VOCAB
        with torch.no_grad():
            base = model(**inputs).diffusion_prediction
            after = model(**{**inputs, "input_ids": changed}).diffusion_prediction
        torch.testing.assert_close(base, after)

    def test_gen_image_requires_image_inputs(self, model):
        inputs = _gen_image_inputs()
        inputs.pop("timestep")
        with pytest.raises(ValueError, match="timestep"):
            model(**inputs)

    def test_unknown_mode(self, model):
        with pytest.raises(ValueError, match="Unknown mode"):
            model(torch.zeros(1, 3, dtype=torch.long), rope_positions=torch.zeros(1, 3, 2), mode="edit")

    def test_hf_state_dict_round_trip(self, model):
        native = model.state_dict()
        hf = model.state_dict_adapter.to_hf(native)
        assert "model.wte.weight" in hf
        assert "model.layers.0.mlp.gate.wg.weight" in hf
        assert "model.layers.0.mlp.experts.3.gate_and_up_proj.weight" in hf
        assert "model.layers.0.mlp.shared_mlp.down_proj.weight" in hf
        restored = model.state_dict_adapter.from_hf(hf)
        assert restored.keys() == native.keys()
        for key, value in native.items():
            torch.testing.assert_close(restored[key], value, msg=key)
