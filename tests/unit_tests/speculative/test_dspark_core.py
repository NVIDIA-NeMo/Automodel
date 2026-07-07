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

"""CPU forward test for DSparkTrainerModule (the recipe's per-step wrapper)."""

import torch
from transformers import Qwen3Config

from nemo_automodel.components.speculative.dspark import Qwen3DSparkModel, build_draft_config
from nemo_automodel.components.speculative.dspark.core import DSparkStepMetrics, DSparkTrainerModule

VOCAB = 256
HIDDEN = 64
TARGET_LAYER_IDS = [1, 3]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _build_draft():
    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
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
    draft_config = build_draft_config(target_config, margs)
    draft_config._attn_implementation = "sdpa"
    model = Qwen3DSparkModel(draft_config).to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def test_trainer_module_forward_returns_finite_metrics():
    trainer = DSparkTrainerModule(
        _build_draft(), loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0
    )
    b, s = 2, 16
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        torch.manual_seed(1234)
        out = trainer(
            input_ids=torch.randint(0, VOCAB, (b, s), generator=gen),
            target_hidden_states=torch.randn(b, s, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen),
            loss_mask=torch.ones(b, s, dtype=torch.uint8),
            target_last_hidden_states=torch.randn(b, s, HIDDEN, generator=gen),
        )
    assert isinstance(out, DSparkStepMetrics)
    for term in (
        out.loss,
        out.ce_loss,
        out.l1_loss,
        out.confidence_loss,
        out.accept_rate,
        out.tau,
        out.confidence_abs_error,
        out.confidence_bias,
        out.confidence_cumprod_bias,
    ):
        assert torch.isfinite(term).all()
