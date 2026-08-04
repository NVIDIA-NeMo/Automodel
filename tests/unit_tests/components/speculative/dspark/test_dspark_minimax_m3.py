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

"""CPU forward test for the MiniMax M3 DSpark draft + its config builder.

Builds a tiny MiniMax M3 draft from a MiniMax M3 VL text sub-config and runs one
forward, covering ``build_minimax_m3_draft_config`` and ``MiniMaxM3DSparkModel``.
"""

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.speculative.dspark.config import build_minimax_m3_draft_config
from nemo_automodel.components.speculative.dspark.draft_minimax_m3 import MiniMaxM3DSparkModel

VOCAB = 256
HIDDEN = 64
HEAD_DIM = 16
TARGET_LAYER_IDS = [1, 3]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _build_minimax_m3_draft():
    text_config = MiniMaxM3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        dense_intermediate_size=2 * HIDDEN,
        shared_intermediate_size=2 * HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        rotary_dim=HEAD_DIM // 2,
        partial_rotary_factor=0.5,
        num_mtp_modules=0,
    )
    target_config = SimpleNamespace(model_type="minimax_m3_vl", text_config=text_config)
    margs = _Args(
        num_draft_layers=2,
        target_layer_ids=TARGET_LAYER_IDS,
        block_size=4,
        mask_token_id=5,
        num_anchors=8,
        markov_rank=16,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    draft_config = build_minimax_m3_draft_config(target_config, margs)
    assert draft_config.architectures == ["MiniMaxM3DSparkModel"]
    assert draft_config.num_hidden_layers == 2
    assert draft_config.num_mtp_modules == 0
    draft_config._attn_implementation = "sdpa"
    model = MiniMaxM3DSparkModel(draft_config).to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def test_minimax_m3_draft_forward_shapes():
    model = _build_minimax_m3_draft()
    b, s, block, anchors = 2, 16, 4, 8
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        torch.manual_seed(7)
        out = model(
            input_ids=torch.randint(0, VOCAB, (b, s), generator=gen),
            target_hidden_states=torch.randn(b, s, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen),
            loss_mask=torch.ones(b, s, dtype=torch.uint8),
            target_last_hidden_states=torch.randn(b, s, HIDDEN, generator=gen),
        )
    assert out.draft_logits.shape == (b, anchors, block, VOCAB)
    assert out.confidence_pred.shape == (b, anchors, block)
    assert torch.isfinite(out.draft_logits).all()


def test_minimax_m3_rotary_device_resynced_after_to():
    """``RotaryEmbedding`` caches its frequencies via ``functools.cache`` keyed on
    the module instance, not a persistent buffer, so plain ``nn.Module._apply``
    (which only visits parameters/buffers) never touches it -- a later
    ``.to(device=...)`` would otherwise silently leave the cached frequencies on
    whatever device was current at construction. Simulate a stale recorded
    device (as if the cache had been populated on the wrong device) and confirm
    ``.to()`` resyncs it to match the model's actual parameter device.
    """
    model = _build_minimax_m3_draft()
    model.rotary_emb.device = torch.device("meta")
    model.rotary_emb._compute_concentration_and_inv_freq.cache_clear()

    model = model.to(dtype=torch.float32)

    assert model.rotary_emb.device == next(model.parameters()).device


def test_minimax_m3_draft_reuses_dense_intermediate_size_field():
    """The draft's MLP width should come from the routed-expert-scale
    ``intermediate_size`` (kept narrow), not the target-only wide
    ``dense_intermediate_size``."""
    model = _build_minimax_m3_draft()
    assert model.layers[0].mlp.gate_proj.out_features == 2 * HIDDEN


def test_minimax_m3_flex_attention_compiles_with_dynamic_shapes(monkeypatch):
    import importlib

    import nemo_automodel.components.speculative.dspark.draft_minimax_m3 as draft_minimax_m3

    captured = {}

    def compile_fn(fn, **kwargs):
        captured["fn"] = fn
        captured["kwargs"] = kwargs
        return fn

    monkeypatch.setattr(torch, "compile", compile_fn)
    try:
        importlib.reload(draft_minimax_m3)
        assert captured["fn"] is draft_minimax_m3.flex_attention
        assert captured["kwargs"] == {"mode": "max-autotune-no-cudagraphs", "dynamic": True}
    finally:
        monkeypatch.undo()
        importlib.reload(draft_minimax_m3)
