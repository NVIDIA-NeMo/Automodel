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

"""Functional test: train a DSpark draft against a real (tiny) Qwen3 target.

Exercises the full path the recipe will use -- run the frozen target, capture
its hidden states at the configured feature layers plus the last hidden state,
feed the DSpark draft, and confirm the objective drives the loss down. Requires
a GPU (FlexAttention has no CPU backward); uses a tiny random target so there is
no download.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from nemo_automodel.components.speculative.dspark import (
    Qwen3DSparkModel,
    build_draft_config,
    compute_dspark_loss,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention backward requires a GPU"
)

VOCAB = 1024
HIDDEN = 128
TARGET_LAYER_IDS = [1, 3, 5]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _model_args() -> _Args:
    return _Args(
        num_draft_layers=2,
        target_layer_ids=list(TARGET_LAYER_IDS),
        block_size=4,
        mask_token_id=7,
        num_anchors=16,
        markov_rank=32,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )


def test_dspark_draft_trains_against_tiny_qwen_target():
    device, dtype = "cuda", torch.float32
    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    target = AutoModelForCausalLM.from_config(target_config).to(device=device, dtype=dtype).eval()
    target.requires_grad_(False)

    draft = Qwen3DSparkModel(build_draft_config(target_config, _model_args())).to(
        device=device, dtype=dtype
    )
    draft.initialize_embeddings_and_head(
        embed_tokens=target.get_input_embeddings(),
        lm_head=target.get_output_embeddings(),
        freeze=True,
    )

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (1, 32), device=device)
    loss_mask = torch.ones(1, 32, dtype=torch.uint8, device=device)

    # Capture the target supervision exactly as the recipe will: concat of the
    # configured feature layers, plus the final hidden state for the TV / confidence loss.
    with torch.no_grad():
        out = target(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states  # length num_layers + 1; hs[0] is the embedding output
        target_hidden_states = torch.cat([hs[i + 1] for i in TARGET_LAYER_IDS], dim=-1)
        target_last_hidden_states = hs[-1]

    optim = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=5e-3)
    draft.train()
    losses = []
    for _ in range(20):
        optim.zero_grad()
        torch.manual_seed(123)  # fixed anchors -> clean overfit signal
        o = draft(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )
        loss = compute_dspark_loss(
            outputs=o,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
        )
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
