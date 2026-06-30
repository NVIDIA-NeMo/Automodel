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

"""CPU forward test for the Gemma4 DSpark draft + its config builder.

Builds a tiny Gemma4 draft from a Gemma4 text sub-config and runs one forward,
covering ``build_gemma4_draft_config`` and ``Gemma4DSparkModel``. The prototype
draft does not support per-layer input gates, so ``hidden_size_per_layer_input``
is set to 0.
"""

from types import SimpleNamespace

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from nemo_automodel.components.speculative.dspark.config import build_gemma4_draft_config
from nemo_automodel.components.speculative.dspark.draft_gemma4 import Gemma4DSparkModel

VOCAB = 256
HIDDEN = 64
TARGET_LAYER_IDS = [1, 3]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _build_gemma4_draft():
    text_config = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        hidden_size_per_layer_input=0,
    )
    target_config = SimpleNamespace(model_type="gemma4_unified", text_config=text_config)
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
    draft_config = build_gemma4_draft_config(target_config, margs)
    assert draft_config.architectures == ["Gemma4DSparkModel"]
    assert draft_config.num_hidden_layers == 2
    draft_config._attn_implementation = "sdpa"
    model = Gemma4DSparkModel(draft_config).to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def test_gemma4_draft_forward_shapes():
    model = _build_gemma4_draft()
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


def test_gemma4_rope_inv_freq_stays_fp32_after_bf16_cast():
    """``model.to(bf16)`` must keep the Gemma4 RoPE frequencies in fp32.

    A bf16-rounded ``inv_freq`` dephases with absolute position and erodes draft
    acceptance, so the draft recomputes fresh fp32 frequencies after the cast.
    """
    model = _build_gemma4_draft()  # fp32
    ref = model.rotary_emb.inv_freq.detach().clone().float()

    model = model.to(torch.bfloat16)
    inv_freq = model.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32
    # Recomputed fresh in fp32, not a bf16 round-trip (bf16 rel. error ~1e-2).
    assert torch.allclose(inv_freq, ref, rtol=1e-6, atol=1e-8)
    # The pin is rope-only: the rest of the model still casts to bf16.
    assert next(model.layers[0].parameters()).dtype == torch.bfloat16
