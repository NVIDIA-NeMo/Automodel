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

"""Tests for the DFlash draft model and its helpers."""

from __future__ import annotations

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
)


def test_build_target_layer_ids_spread_and_count():
    # single draft layer -> middle of the target
    assert build_target_layer_ids(36, 1) == [18]
    # multiple draft layers -> monotonic, in-bounds spread
    ids = build_target_layer_ids(36, 5)
    assert len(ids) == 5
    assert ids == sorted(ids)
    assert all(0 <= i < 36 for i in ids)


def test_extract_context_feature_uses_offset_one():
    # hidden_states[0] is the embedding output; layer i's output is at index i+1.
    hs = [torch.full((1, 2, 3), float(i)) for i in range(6)]
    out = extract_context_feature(hs, [1, 3])
    assert out.shape == (1, 2, 6)
    # first 3 features come from hidden_states[2], next 3 from hidden_states[4]
    assert torch.allclose(out[..., :3], torch.full((1, 2, 3), 2.0))
    assert torch.allclose(out[..., 3:], torch.full((1, 2, 3), 4.0))


def _draft_cfg(bs=4):
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1000000,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = 8
    cfg.block_size = bs
    cfg.dflash_config = {"mask_token_id": 63, "target_layer_ids": [1, 3, 5]}
    cfg._attn_implementation = "sdpa"
    return cfg


def _rope_relerr(inv_freq):
    # Reference fp32 default-rope frequencies for head_dim=8, theta=1e6.
    ref = 1.0 / (1000000 ** (torch.arange(0, 8, 2).float() / 8))
    return ((inv_freq.float() - ref).abs() / ref).max().item()


def test_rope_inv_freq_stays_fp32_after_bf16_cast():
    """``model.to(bf16)`` must not round the RoPE frequencies.

    The serving runtime keeps an fp32 RoPE cache; if the draft trained with a
    bf16-rounded ``inv_freq`` the train/inference RoPE would diverge (worse with
    longer context) and erode acceptance. The draft pins ``inv_freq`` to fp32.
    """
    draft = Qwen3DFlashDraftModel(_draft_cfg()).to(torch.bfloat16)
    inv_freq = draft.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32
    # Recomputed fresh, so the values are exact fp32 (not a bf16 round-trip).
    assert _rope_relerr(inv_freq) < 1e-6
    # original_inv_freq (used by dynamic-rope resets) is pinned too.
    assert draft.rotary_emb.original_inv_freq.dtype == torch.float32
    # The rest of the model is still bf16 (the pin is rope-only).
    assert next(draft.layers[0].parameters()).dtype == torch.bfloat16


def test_rope_inv_freq_fp32_survives_chained_casts():
    draft = Qwen3DFlashDraftModel(_draft_cfg()).to(torch.float16).to(torch.bfloat16)
    assert draft.rotary_emb.inv_freq.dtype == torch.float32
    assert _rope_relerr(draft.rotary_emb.inv_freq) < 1e-6


def test_draft_forward_output_shape():
    H, n_layers_target, tli, bs = 32, 8, [1, 3, 5], 4
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = n_layers_target
    cfg.block_size = bs
    cfg.dflash_config = {"mask_token_id": 63, "target_layer_ids": tli}
    cfg._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(cfg)
    assert draft.target_layer_ids == tli
    assert draft.fc.in_features == len(tli) * H

    B, S, N = 2, 10, 3
    Q = N * bs
    noise = torch.randn(B, Q, H)
    target_hidden = torch.randn(B, S, len(tli) * H)
    position_ids = torch.arange(S + Q).unsqueeze(0).expand(B, -1)
    out = draft(position_ids=position_ids, attention_mask=None, noise_embedding=noise, target_hidden=target_hidden)
    assert out.shape == (B, Q, H)
    assert torch.isfinite(out).all()
