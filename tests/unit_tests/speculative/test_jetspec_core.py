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

"""Tests for the JetSpec trainer module (causal parallel drafting + forward-KL)."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.speculative.dflash import jetspec_core
from nemo_automodel.components.speculative.dflash.core import NoValidAnchorsError
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.jetspec_core import JetSpecStepMetrics, JetSpecTrainerModule

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1


def _build_trainer(num_anchors=8, attention_backend="sdpa", kd_temperature=1.0, kd_chunk_size=0):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    cfg._attn_implementation = attention_backend
    draft = Qwen3DFlashDraftModel(cfg)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    return JetSpecTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        kd_temperature=kd_temperature,
        kd_chunk_size=kd_chunk_size,
    )


def _inputs(bsz=2, seq_len=24):
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    target_logits = torch.randn(bsz, seq_len, VOCAB)
    return input_ids, hidden, loss_mask, target_logits


def test_init_builds_kd_loss():
    trainer = _build_trainer(kd_temperature=2.0, kd_chunk_size=128)
    assert isinstance(trainer.kd_loss_fn, KDLoss)
    assert trainer.kd_temperature == 2.0
    assert trainer.kd_chunk_size == 128
    assert trainer.kd_loss_fn.temperature == 2.0
    assert trainer.kd_loss_fn.chunk_size == 128
    # JetSpec disables DFlash's decay weighting (uniform over active positions).
    assert trainer.loss_decay_gamma is None


def test_forward_returns_finite_loss_and_grads_flow_to_draft():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask, target_logits = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    assert isinstance(out, JetSpecStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert 0.0 <= out.accuracy.item() <= 1.0
    assert out.valid_tokens.item() > 0
    # accept_len includes the always-accepted anchor (+1) and is capped at block_size.
    assert 1.0 <= out.accept_len.item() <= BLOCK_SIZE
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def test_accuracy_is_vs_target_argmax_not_ground_truth(monkeypatch):
    """Accuracy is the draft-vs-target greedy agreement (acceptance proxy), not vs ground truth.

    Stub the draft to emit a constant hidden so the student logits are uniform and
    ``argmax`` is token 0 everywhere; build target logits whose argmax is also token 0.
    The draft then agrees with the target's greedy at every supervised position, so
    accuracy must be 1.0 -- even though the (random) ground-truth tokens are almost
    never 0, which a vs-ground-truth metric would score near 0.
    """
    trainer = _build_trainer(attention_backend="sdpa")
    monkeypatch.setattr(
        trainer.draft_model,
        "forward",
        lambda position_ids, noise_embedding, target_hidden, attention_mask: torch.zeros_like(noise_embedding),
    )
    input_ids, hidden, loss_mask, _ = _inputs()
    target_logits = torch.full((input_ids.size(0), input_ids.size(1), VOCAB), -10.0)
    target_logits[:, :, 0] = 10.0  # target argmax = token 0 at every position
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    assert out.accuracy.item() == 1.0
    # Every drafted depth agrees with the target -> whole block accepted: accept_len
    # == block_size (all block_size-1 drafted depths + the anchor).
    assert out.accept_len.item() == BLOCK_SIZE


def test_kd_chunking_matches_unchunked_loss():
    """``kd_chunk_size`` is a memory optimisation: the loss must match the unchunked path.

    Both trainers share the same draft weights and inputs, so the only difference
    is the KD reduction path; the resulting loss must be numerically equal.
    """
    full = _build_trainer(kd_chunk_size=0)
    chunked = _build_trainer(kd_chunk_size=3)
    chunked.draft_model.load_state_dict(full.draft_model.state_dict())
    chunked.lm_head.load_state_dict(full.lm_head.state_dict())
    chunked.embed_tokens.load_state_dict(full.embed_tokens.state_dict())

    input_ids, hidden, loss_mask, target_logits = _inputs()
    kwargs = dict(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    # Seed identically before each call so anchor sampling (which advances the
    # global RNG) draws the same anchors -- the only remaining difference is the
    # chunked vs unchunked KD reduction.
    torch.manual_seed(1234)
    out_full = full(**kwargs)
    torch.manual_seed(1234)
    out_chunked = chunked(**kwargs)
    assert torch.allclose(out_full.loss, out_chunked.loss, atol=1e-5)


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_padding_blocks_do_not_nan_loss_or_grads(attention_backend):
    """Mixed-length batches make padding blocks; those must not NaN the loss/grads."""
    trainer = _build_trainer(attention_backend=attention_backend)
    input_ids, hidden, loss_mask, target_logits = _inputs(bsz=2, seq_len=24)
    loss_mask[0, 5:] = 0.0  # sample 0 keeps fewer anchors -> trailing padding blocks

    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in trainer.draft_model.parameters() if p.grad is not None)


def test_no_supervised_positions_raises():
    """Anchors exist but every predicted continuation is loss-masked -> skip the batch."""
    trainer = _build_trainer(num_anchors=4)
    seq_len = 8
    # Supervise only positions 0 and 4 (both valid anchors), but their continuations
    # (1..3, 5..7) are all masked, so no draft position is supervised.
    loss_mask = torch.zeros(1, seq_len)
    loss_mask[0, 0] = 1.0
    loss_mask[0, 4] = 1.0
    input_ids = torch.randint(0, VOCAB - 1, (1, seq_len))
    hidden = torch.randn(1, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    target_logits = torch.randn(1, seq_len, VOCAB)
    with pytest.raises(NoValidAnchorsError):
        trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)


def test_gather_teacher_logits_alignment():
    """Predicted offset k (k>=1) at seq pos anchor+k uses the teacher logits at anchor+k-1.

    So the gathered teacher source positions are anchor+0 .. anchor+block_size-2.
    """
    trainer = _build_trainer()
    bsz, seq_len = 1, 20
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    anchors = torch.tensor([[2, 10]])
    keep = torch.tensor([[True, True]])
    label_indices, _, _ = trainer._build_block_targets(input_ids, loss_mask, anchors, keep, seq_len)

    # Channel 0 of the target logits encodes the absolute sequence position.
    target_logits = torch.zeros(bsz, seq_len, VOCAB)
    target_logits[0, :, 0] = torch.arange(seq_len, dtype=torch.float32)

    teacher = trainer._gather_teacher_logits(target_logits, label_indices, seq_len)
    n = anchors.size(1)
    gathered_positions = teacher[..., 0].reshape(bsz, n, BLOCK_SIZE - 1)
    expected_positions = label_indices[:, :, :-1].float()  # anchor + 0 .. anchor + bs-2
    assert torch.equal(gathered_positions, expected_positions)
    # Concretely: block 0 (anchor 2) -> teacher at 2,3,4; block 1 (anchor 10) -> 10,11,12.
    assert gathered_positions.tolist() == [[[2.0, 3.0, 4.0], [10.0, 11.0, 12.0]]]


def test_forward_passes_causal_mask(monkeypatch):
    """JetSpec must build the block mask with causal=True (its defining change)."""
    seen = {}
    real = jetspec_core.create_dflash_sdpa_mask

    def _spy(*args, **kwargs):
        seen["causal"] = kwargs.get("causal")
        return real(*args, **kwargs)

    monkeypatch.setattr(jetspec_core, "create_dflash_sdpa_mask", _spy)
    trainer = _build_trainer(attention_backend="sdpa")
    input_ids, hidden, loss_mask, target_logits = _inputs()
    trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    assert seen["causal"] is True


def test_forward_flex_backend_builds_causal_block_mask(monkeypatch):
    """The flex_attention branch builds a causal FlexAttention BlockMask.

    The flex_attention kernel needs CUDA, so the real draft forward can't run on
    CPU; stub the draft to return a same-shaped hidden so the branch (and the rest
    of the loss path) is still exercised, and assert the block-mask builder was
    called with causal=True.
    """
    seen = {}

    def _spy_block_mask(*args, **kwargs):
        seen["causal"] = kwargs.get("causal")
        return "sentinel-block-mask"

    monkeypatch.setattr(jetspec_core, "create_dflash_block_mask", _spy_block_mask)
    trainer = _build_trainer(attention_backend="flex_attention")
    # Bypass the (CUDA-only) flex attention kernel: the draft just echoes a
    # zero hidden of the noise-block shape so lm_head / loss still run on CPU.
    # (Patch the module's forward, not the registered submodule itself.)
    monkeypatch.setattr(
        trainer.draft_model,
        "forward",
        lambda position_ids, noise_embedding, target_hidden, attention_mask: torch.zeros_like(noise_embedding),
    )

    input_ids, hidden, loss_mask, target_logits = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    assert seen["causal"] is True
    assert torch.isfinite(out.loss)


def test_build_block_targets_masks_out_of_bounds_and_loss_masked():
    trainer = _build_trainer()
    bsz, seq_len = 1, 12
    input_ids = torch.arange(1, seq_len + 1).view(1, seq_len)
    loss_mask = torch.ones(bsz, seq_len)
    loss_mask[0, 9:] = 0.0  # positions 9,10,11 not supervised
    anchors = torch.tensor([[6]])  # block covers 6,7,8,9 -> position 9 is loss-masked
    keep = torch.tensor([[True]])
    label_indices, target_ids, block_mask = trainer._build_block_targets(input_ids, loss_mask, anchors, keep, seq_len)
    assert label_indices.tolist() == [[[6, 7, 8, 9]]]
    assert target_ids.tolist() == [[[7, 8, 9, 10]]]  # input_ids are 1..seq_len
    # Offsets 0,1,2 supervised (positions 6,7,8); offset 3 (position 9) loss-masked.
    assert block_mask.squeeze().tolist() == [1.0, 1.0, 1.0, 0.0]
