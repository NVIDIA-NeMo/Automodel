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

"""Self-contained correctness tests for the DSpark draft model + loss.

A tiny, CPU-only Qwen3 draft model that exercises the anchor sampler, the block
attention mask, the Markov + confidence heads, and the three-term objective.
"""

import pytest
import torch
from transformers import Qwen3Config

from nemo_automodel.components.speculative.dspark import (
    Qwen3DSparkModel,
    build_draft_config,
    compute_dspark_loss,
)



class _Args(dict):
    """Dict that also supports attribute access.

    ``build_draft_config`` uses both ``model_args.field`` and ``"field" in model_args``.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc


# Tiny shapes that keep the test fast and CPU-friendly.
HIDDEN = 64
NUM_TARGET_LAYERS_TOTAL = 8
TARGET_LAYER_IDS = [1, 3, 5, 7]
BLOCK_SIZE = 4
NUM_ANCHORS = 8
VOCAB = 256
BATCH = 2
SEQ = 16


def _tiny_target_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=NUM_TARGET_LAYERS_TOTAL,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


def _tiny_model_args(markov_head_type: str = "vanilla") -> _Args:
    return _Args(
        num_draft_layers=2,
        target_layer_ids=list(TARGET_LAYER_IDS),
        block_size=BLOCK_SIZE,
        mask_token_id=5,
        num_anchors=NUM_ANCHORS,
        markov_rank=16,
        markov_head_type=markov_head_type,
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
    )


def _build_model(
    device: str = "cpu", markov_head_type: str = "vanilla", attn_implementation: str = "sdpa"
) -> Qwen3DSparkModel:
    # Unit tests default to the sdpa (dense-mask) backend so they run on CPU
    # without flex_attention / inductor, matching the DFlash tests. The flex
    # path is exercised by the GPU functional tests.
    draft_config = build_draft_config(_tiny_target_config(), _tiny_model_args(markov_head_type))
    draft_config._attn_implementation = attn_implementation
    model = Qwen3DSparkModel(draft_config).to(device=device, dtype=torch.float32).eval()
    # Seed the frozen embedding + head from a fake target, exactly as the recipe will.
    fake_embed = torch.nn.Embedding(VOCAB, HIDDEN)
    fake_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    model.initialize_embeddings_and_head(embed_tokens=fake_embed, lm_head=fake_head, freeze=True)
    return model


def _batch(seed: int = 0, device: str = "cpu") -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen).to(device),
        "loss_mask": torch.ones(BATCH, SEQ, dtype=torch.uint8, device=device),
        "target_hidden_states": torch.randn(BATCH, SEQ, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen).to(device),
        "target_last_hidden_states": torch.randn(BATCH, SEQ, HIDDEN, generator=gen).to(device),
    }


def _forward(model: Qwen3DSparkModel, batch: dict, seed: int = 1234):
    # Anchor sampling uses the global RNG inside forward; seed it for reproducibility.
    torch.manual_seed(seed)
    return model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"],
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"],
    )


def test_forward_shapes():
    model = _build_model()
    with torch.no_grad():
        out = _forward(model, _batch())

    assert out.draft_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert out.target_ids.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.eval_mask.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.block_keep_mask.shape == (BATCH, NUM_ANCHORS)
    # Confidence head + L1 target logits are enabled in this config.
    assert out.confidence_pred.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.aligned_target_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert torch.isfinite(out.draft_logits).all()


@pytest.mark.parametrize("head", ["vanilla", "gated", "rnn"])
def test_markov_head_variants_forward(head):
    """All three Markov head variants (incl. the recurrent RNN head) run and bias the logits."""
    model = _build_model(markov_head_type=head)
    assert model.markov_head is not None
    assert model.markov_head.markov_head_type == head
    with torch.no_grad():
        out = _forward(model, _batch())
    assert out.draft_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert torch.isfinite(out.draft_logits).all()


def test_embeddings_and_head_frozen():
    model = _build_model()
    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    # The backbone / fc / heads stay trainable.
    assert model.fc.weight.requires_grad is True
    assert any(p.requires_grad for p in model.layers.parameters())


def test_forward_deterministic_under_seed():
    model = _build_model()
    batch = _batch()
    with torch.no_grad():
        a = _forward(model, batch, seed=7)
        b = _forward(model, batch, seed=7)
    assert torch.equal(a.draft_logits, b.draft_logits)
    assert torch.equal(a.target_ids, b.target_ids)


def test_loss_is_finite_scalar_and_backprops():
    model = _build_model()
    out = _forward(model, _batch())
    loss = compute_dspark_loss(
        outputs=out,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    # Frozen tensors receive no gradient; trainable backbone does.
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all()


def test_loss_decreases_on_fixed_batch():
    model = _build_model()
    model.train()
    batch = _batch(seed=3)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)

    losses = []
    for _ in range(20):
        optim.zero_grad()
        out = _forward(model, batch, seed=99)  # fixed anchors -> clean overfit signal
        loss = compute_dspark_loss(
            outputs=out,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
        )
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
