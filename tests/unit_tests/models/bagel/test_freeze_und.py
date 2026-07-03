# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""freeze_und gradient-flow test for the BAGEL MoT decoder layer.

When ``config.freeze_und=True`` the und-path computations inside
``PackedAttentionMoT`` and ``Qwen2MoTDecoderLayer`` are detached at every
site that mirrors upstream BAGEL's qwen2_navit (3 sites in attention:
V/Q/K post-norm; 2 sites in the decoder layer: post-attn-residual and
post-MLP). The post-condition is:

  * every und-path weight (``q_proj.weight``, ``mlp.gate_proj.weight``,
    ``input_layernorm.weight``, etc.) sees zero gradient;
  * every gen-path sibling (``*_moe_gen.weight``) still receives gradient.

The test exercises this end-to-end at the layer level. It does not load
BAGEL weights; random init is fine because we only inspect
``param.grad`` — its magnitude doesn't matter, only whether it is
None / zero / nonzero.

Requires CUDA: the attention dispatch in ``PackedAttentionMoT.forward_train``
is hardcoded to ``SDPBackend.EFFICIENT_ATTENTION``, which has no CPU kernel.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _build_layer(freeze_und: bool, fused_projections: bool):
    """Construct a tiny Qwen2MoTDecoderLayer in MoT mode."""
    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import (
        Qwen2Config,
        Qwen2MoTDecoderLayer,
    )

    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        freeze_und=freeze_und,
        fused_projections=fused_projections,
    )
    # Run the whole layer in bf16 to match how it's used inside production
    # autocast (every weight is bf16, every input is bf16, every intermediate
    # tensor is bf16) — sidesteps the new_zeros/q_proj dtype-mismatch issue
    # that production avoids by virtue of every tensor already being bf16
    # by the time it enters the layer.
    layer = Qwen2MoTDecoderLayer(cfg, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    layer.train()
    return cfg, layer


def _build_packed_inputs(hidden_size: int, head_dim: int):
    """Two samples of 4 tokens each: sample 0 is und-only, sample 1 is gen-only.
    Total packed length = 8. Indexes split 4/4 across und/gen.
    """
    torch.manual_seed(0)
    n_und = 4
    n_gen = 4
    n = n_und + n_gen
    sample_lens = [n_und, n_gen]

    device = "cuda"
    # packed_sequence enters the decoder layer in bf16 in production (autocast
    # on the embedding). The layer's internal ``new_zeros(packed_sequence.shape)``
    # inherits this dtype, which must match the bf16 output of self.q_proj
    # under autocast. Pass bf16 directly here.
    packed_sequence = torch.randn(n, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=False)
    packed_und_token_indexes = torch.arange(n_und, dtype=torch.long, device=device)
    packed_gen_token_indexes = torch.arange(n_und, n, dtype=torch.long, device=device)

    # Per-sample causal masks for SDPA (one [seq, seq] mask per sample).
    # Use additive form with -inf above diagonal.
    def _causal(seq_len: int) -> torch.Tensor:
        m = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bfloat16)
        m.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1), float("-inf")
        )
        return m

    attention_mask = [_causal(n_und), _causal(n_gen)]

    # Per-token cos/sin for rotary embeddings (shape [n, head_dim]).
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(n, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    return dict(
        packed_sequence=packed_sequence,
        sample_lens=sample_lens,
        attention_mask=attention_mask,
        packed_position_embeddings=(cos, sin),
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
    )


def _classify(name: str) -> str:
    """und-path params have no ``moe_gen`` segment in the name; gen-path do."""
    return "gen" if "moe_gen" in name else "und"


def _copy_split_layer_to_fused(split, fused) -> None:
    split_params = dict(split.named_parameters())
    fused_projection_parts = {
        "self_attn.qkv_proj.weight": (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ),
        "self_attn.qkv_proj.bias": (
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ),
        "self_attn.qkv_proj_moe_gen.weight": (
            "self_attn.q_proj_moe_gen.weight",
            "self_attn.k_proj_moe_gen.weight",
            "self_attn.v_proj_moe_gen.weight",
        ),
        "self_attn.qkv_proj_moe_gen.bias": (
            "self_attn.q_proj_moe_gen.bias",
            "self_attn.k_proj_moe_gen.bias",
            "self_attn.v_proj_moe_gen.bias",
        ),
        "mlp.gate_up_proj.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "mlp_moe_gen.gate_up_proj.weight": (
            "mlp_moe_gen.gate_proj.weight",
            "mlp_moe_gen.up_proj.weight",
        ),
    }
    with torch.no_grad():
        for name, fused_param in fused.named_parameters():
            if name in fused_projection_parts:
                fused_param.copy_(torch.cat([split_params[part] for part in fused_projection_parts[name]], dim=0))
            else:
                fused_param.copy_(split_params[name])


def test_fused_mot_layer_matches_split_forward():
    split_cfg, split = _build_layer(freeze_und=False, fused_projections=False)
    _, fused = _build_layer(freeze_und=False, fused_projections=True)
    _copy_split_layer_to_fused(split, fused)
    inputs = _build_packed_inputs(split_cfg.hidden_size, split_cfg.hidden_size // split_cfg.num_attention_heads)

    split_output = split(**inputs)
    fused_output = fused(**inputs)

    torch.testing.assert_close(fused_output, split_output, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("fused_projections", [False, True])
def test_freeze_und_zeroes_und_path_grads(fused_projections):
    cfg, layer = _build_layer(freeze_und=True, fused_projections=fused_projections)
    inputs = _build_packed_inputs(cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads)
    inputs["packed_sequence"] = inputs["packed_sequence"].clone().requires_grad_(True)

    out = layer(**inputs)
    out.float().sum().backward()

    und_params, gen_params = [], []
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        (gen_params if _classify(name) == "gen" else und_params).append((name, p))

    # Sanity: we have both buckets.
    assert len(und_params) > 0, "no und-path params found"
    assert len(gen_params) > 0, "no gen-path params found"

    # Every und-path param must have zero gradient. ``grad is None`` means
    # autograd never reached the param — also acceptable (and common, since
    # every und consumer is detached before reaching loss).
    for name, p in und_params:
        if p.grad is None:
            continue
        assert p.grad.abs().sum().item() == 0.0, f"und param {name} got nonzero grad under freeze_und=True"

    # Sanity: at least one gen-path param must have a nonzero gradient —
    # otherwise the test setup never exercised the gen path.
    nonzero_gen = sum(1 for _, p in gen_params if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert nonzero_gen > 0, "no gen-path param received gradient — test setup is broken"


@pytest.mark.parametrize("fused_projections", [False, True])
def test_freeze_und_off_propagates_grads_to_und_path(fused_projections):
    """Control: with freeze_und=False, und-path params should receive gradient."""
    cfg, layer = _build_layer(freeze_und=False, fused_projections=fused_projections)
    inputs = _build_packed_inputs(cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads)
    inputs["packed_sequence"] = inputs["packed_sequence"].clone().requires_grad_(True)

    out = layer(**inputs)
    out.float().sum().backward()

    nonzero_und = 0
    total_und = 0
    for name, p in layer.named_parameters():
        if not p.requires_grad or _classify(name) != "und":
            continue
        total_und += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            nonzero_und += 1

    # At least the q/k/v proj of und path must have nonzero grad — they're
    # directly involved in computing every output token via the shared V
    # tensor and o_proj on und positions.
    assert nonzero_und > 0, f"freeze_und=False but 0/{total_und} und params received gradient"
