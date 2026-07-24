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

"""Parity tests for the config-gated BAGEL throughput optimizations.

Three optimizations are selected via ``BagelBackendConfig`` fields (default OFF,
set in YAML as ``model.backend.*``):

  * ``fused_swiglu`` — fuse ``silu(gate) * up`` into one compiled kernel.
  * ``fused_rope``   — fuse the rotary apply into one compiled kernel.
  * ``mot_grouped``  — route und/gen tokens by contiguous slices instead of
    per-layer gather/scatter, restoring original token order around the
    attention kernel.

All three are numerically equivalent to the default path (grouped routing is a
pure permutation — each token is processed by its correct expert and attention
runs over the same tokens in the same original order), so the tests assert the
optimized path matches the default path.
"""

from __future__ import annotations

import pytest
import torch

MODELING = "nemo_automodel.components.models.bagel.modeling_qwen2_packed"


def _mot_config():
    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2Config

    return Qwen2Config(
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
        hidden_act="silu",
    )


def test_fused_swiglu_matches_eager():
    """``Qwen2MLP`` with the fused SwiGLU kernel matches the eager path exactly."""
    from nemo_automodel.components.models.bagel.configuration import BagelBackendConfig
    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2MLP

    # The backend flag drives the fused path.
    assert Qwen2MLP(_mot_config(), backend=BagelBackendConfig(fused_swiglu=True))._fuse_silu_mul is True
    assert Qwen2MLP(_mot_config(), backend=BagelBackendConfig(fused_swiglu=False))._fuse_silu_mul is False

    torch.manual_seed(0)
    mlp = Qwen2MLP(_mot_config()).to(torch.float32).eval()
    x = torch.randn(16, mlp.hidden_size)

    mlp._fuse_silu_mul = False
    out_eager = mlp(x)
    mlp._fuse_silu_mul = True
    out_fused = mlp(x)

    torch.testing.assert_close(out_fused, out_eager, atol=1e-4, rtol=1e-4)


def test_fused_rope_matches_eager():
    """The fused rotary apply (``fused=True``) matches the eager path."""
    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import apply_rotary_pos_emb

    torch.manual_seed(0)
    n, heads, kv_heads, head_dim = 8, 4, 2, 16
    q = torch.randn(n, heads, head_dim)
    k = torch.randn(n, kv_heads, head_dim)
    cos = torch.randn(n, head_dim)
    sin = torch.randn(n, head_dim)

    q_eager, k_eager = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, fused=False)
    q_fused, k_fused = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, fused=True)

    torch.testing.assert_close(q_fused, q_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_fused, k_eager, atol=1e-4, rtol=1e-4)


def _interleaved_packed_inputs(hidden_size: int, head_dim: int, device: str):
    """One packed sample of 8 tokens with und/gen tokens INTERLEAVED.

    und at even positions, gen at odd — so grouped routing genuinely reorders
    them (a no-op if they were already contiguous). Returns the kwargs for
    ``Qwen2MoTDecoderLayer.forward_train`` plus the raw und/gen indexes.
    """
    torch.manual_seed(0)
    n = 8
    packed_sequence = torch.randn(n, hidden_size, dtype=torch.bfloat16, device=device)
    packed_und_token_indexes = torch.arange(0, n, 2, dtype=torch.long, device=device)  # [0,2,4,6]
    packed_gen_token_indexes = torch.arange(1, n, 2, dtype=torch.long, device=device)  # [1,3,5,7]

    # Single-sample causal mask (SDPA list path).
    causal = torch.zeros(n, n, device=device, dtype=torch.bfloat16)
    causal.masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1), float("-inf"))

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(n, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    return dict(
        packed_sequence=packed_sequence,
        sample_lens=[n],
        attention_mask=[causal],
        packed_position_embeddings=(cos, sin),
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=packed_gen_token_indexes,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoT attention uses CUDA-only EFFICIENT_ATTENTION")
def test_grouped_routing_matches_default():
    """Grouped und/gen routing (permute + slice + restore) matches the default gather/scatter path.

    Grouped routing is a pure permutation: the caller reorders the packed
    sequence so und/gen tokens are contiguous, every layer routes by slice, and
    attention is restored to original order around the kernel. Grouped mode is
    signalled solely by passing ``mot_perm``/``mot_inv`` (the model threads these
    only when ``backend.mot_grouped`` is set). Feeding the layer the permuted
    inputs + ``mot_perm``/``mot_inv`` and un-permuting the output must reproduce
    the default (interleaved gather/scatter) output.
    """
    from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2MoTDecoderLayer

    device = "cuda"
    cfg = _mot_config()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    # ``forward`` dispatches to ``forward_train`` only in training mode; parity is a
    # forward-only check so we still wrap the calls in ``torch.no_grad()`` below.
    layer = Qwen2MoTDecoderLayer(cfg, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    layer.train()

    base = _interleaved_packed_inputs(cfg.hidden_size, head_dim, device)

    # --- default path: no mot_perm/mot_inv -> gather/scatter routing ---
    with torch.no_grad():
        out_default = layer(**base)

    # --- grouped path: permute inputs, pass mot_perm/mot_inv, un-permute output ---
    perm = torch.cat([base["packed_und_token_indexes"], base["packed_gen_token_indexes"]])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    cos, sin = base["packed_position_embeddings"]

    grouped_inputs = dict(
        packed_sequence=base["packed_sequence"][perm],
        sample_lens=base["sample_lens"],
        attention_mask=base["attention_mask"],
        packed_position_embeddings=(cos[perm], sin[perm]),
        packed_und_token_indexes=base["packed_und_token_indexes"],
        packed_gen_token_indexes=base["packed_gen_token_indexes"],
        mot_perm=perm,
        mot_inv=inv,
    )
    with torch.no_grad():
        out_grouped = layer(**grouped_inputs)[inv]  # un-permute back to original order

    # Pure permutation -> near-exact; tolerance covers bf16 attention reduction order.
    torch.testing.assert_close(out_grouped, out_default, atol=1e-2, rtol=1e-2)
