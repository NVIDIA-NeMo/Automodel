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

"""Sequence-packing tests for the Domino and JetSpec DFlash variants."""

from __future__ import annotations

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.domino_core import DominoTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.jetspec_core import JetSpecTrainerModule

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
EMB_DIM = 16
GRU_HIDDEN = 24


def _draft_config(projector_type=None, pure_prefix=1, shift_label=True):
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
    dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    if projector_type is not None:
        dflash_config.update(
            {
                "projector_type": projector_type,
                "emb_dim": EMB_DIM,
                "gru_hidden_dim": GRU_HIDDEN,
                "pure_draft_prefix_len": pure_prefix,
                "shift_label": shift_label,
            }
        )
    cfg.dflash_config = dflash_config
    cfg._attn_implementation = "sdpa"
    return cfg


def _build_domino(num_anchors=8, shift_label=True):
    torch.manual_seed(0)
    draft = Qwen3DFlashDraftModel(_draft_config(projector_type="domino", shift_label=shift_label))
    return DominoTrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        shift_label=shift_label,
    )


def _build_jetspec(num_anchors=8):
    torch.manual_seed(0)
    draft = Qwen3DFlashDraftModel(_draft_config())
    return JetSpecTrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=num_anchors,
    )


def _uniform_pack_meta(num_docs: int, doc_len: int):
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    return position_ids, seq_lens, doc_remaining


def _single_doc_meta(T: int):
    position_ids = torch.arange(T).unsqueeze(0)
    seq_lens = torch.tensor([[T]], dtype=torch.long)
    doc_remaining = torch.arange(T - 1, -1, -1).unsqueeze(0)
    return position_ids, seq_lens, doc_remaining


def _inputs(T: int, with_logits: bool = False):
    torch.manual_seed(7)
    input_ids = torch.randint(0, VOCAB - 1, (1, T))
    hidden = torch.randn(1, T, len(TARGET_LAYER_IDS) * HIDDEN)
    loss_mask = torch.ones(1, T)
    if with_logits:
        return input_ids, hidden, loss_mask, torch.randn(1, T, VOCAB)
    return input_ids, hidden, loss_mask


# ---------------------------------------------------------------------------
# Domino
# ---------------------------------------------------------------------------


def test_domino_packed_single_doc_matches_unpacked():
    """A single full-width document must reproduce the unpacked Domino forward."""
    for shift_label in (False, True):
        trainer = _build_domino(shift_label=shift_label)
        T = 24
        input_ids, hidden, loss_mask = _inputs(T)
        position_ids, seq_lens, doc_remaining = _single_doc_meta(T)

        torch.manual_seed(42)
        out_unpacked = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.5)
        torch.manual_seed(42)
        out_packed = trainer(
            input_ids=input_ids,
            hidden_states=hidden,
            loss_mask=loss_mask,
            lambda_base=0.5,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
        assert out_packed.valid_tokens.item() == out_unpacked.valid_tokens.item(), f"{shift_label=}"
        torch.testing.assert_close(out_packed.loss, out_unpacked.loss)


def test_domino_packed_two_doc_runs_and_backprops():
    trainer = _build_domino()
    doc_len, num_docs = 8, 2
    input_ids, hidden, loss_mask = _inputs(doc_len * num_docs)
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    torch.manual_seed(3)
    out = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        lambda_base=0.5,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert torch.isfinite(out.loss) and out.valid_tokens.item() > 0
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def test_domino_shift_label_truncates_at_document_boundary(monkeypatch):
    """With shift_label the last label sits at anchor + block_size, one past the
    span anchor sampling keeps in-document; it must be masked under packing."""
    trainer = _build_domino(shift_label=True)
    doc_len, num_docs = 8, 2
    T = doc_len * num_docs
    input_ids, hidden, loss_mask = _inputs(T)
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)

    # Pin a single anchor at doc0's last valid position (4): the block is 4..7
    # (inside doc0) but the shifted labels are 5..8, and position 8 is doc1's
    # first token.
    anchor = doc_len - BLOCK_SIZE
    fixed = (
        torch.tensor([[anchor]], dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )
    monkeypatch.setattr(trainer, "_sample_anchor_positions", lambda *a, **k: fixed)

    out_unpacked = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    out_packed = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    # Unpacked supervises all block_size shifted labels; packed drops exactly the
    # one label that crosses into doc1.
    assert out_unpacked.valid_tokens.item() == BLOCK_SIZE
    assert out_packed.valid_tokens.item() == BLOCK_SIZE - 1


# ---------------------------------------------------------------------------
# JetSpec
# ---------------------------------------------------------------------------


def test_jetspec_packed_single_doc_matches_unpacked():
    """A single full-width document must reproduce the unpacked JetSpec forward."""
    trainer = _build_jetspec()
    T = 24
    input_ids, hidden, loss_mask, target_logits = _inputs(T, with_logits=True)
    position_ids, seq_lens, doc_remaining = _single_doc_meta(T)

    torch.manual_seed(42)
    out_unpacked = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, target_logits=target_logits)
    torch.manual_seed(42)
    out_packed = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        target_logits=target_logits,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert out_packed.valid_tokens.item() == out_unpacked.valid_tokens.item()
    torch.testing.assert_close(out_packed.loss, out_unpacked.loss)


def test_jetspec_packed_two_doc_runs_and_backprops():
    trainer = _build_jetspec()
    doc_len, num_docs = 8, 2
    input_ids, hidden, loss_mask, target_logits = _inputs(doc_len * num_docs, with_logits=True)
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    torch.manual_seed(3)
    out = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        target_logits=target_logits,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert torch.isfinite(out.loss) and out.valid_tokens.item() > 0
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def test_jetspec_packed_anchors_stay_in_document(monkeypatch):
    """The packed JetSpec forward must sample anchors under the document bound."""
    trainer = _build_jetspec(num_anchors=16)
    doc_len, num_docs = 6, 3
    T = doc_len * num_docs
    input_ids, hidden, loss_mask, target_logits = _inputs(T, with_logits=True)
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)

    seen = {}
    real = trainer._sample_anchor_positions

    def _spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen["doc_remaining"] = kwargs.get("doc_remaining")
        seen["anchors"], seen["keep"] = out
        return out

    monkeypatch.setattr(trainer, "_sample_anchor_positions", _spy)
    torch.manual_seed(5)
    trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        target_logits=target_logits,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert seen["doc_remaining"] is not None
    kept = seen["anchors"][seen["keep"]]
    assert bool(((kept % doc_len) <= doc_len - BLOCK_SIZE).all())
