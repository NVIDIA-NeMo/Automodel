# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU tests for EAGLE-3 sequence packing (dataset, masks, trainer integration)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.datasets.llm.eagle3 import (
    _pack_collate,
    build_packed_eagle3_dataset,
)
from nemo_automodel.components.datasets.llm.packed_sequence import build_block_causal_additive_mask
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama import (
    LlamaEagle3DraftModel,
    _seq_lens_to_cu_seqlens,
)
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel


def _src(samples: list[tuple[list[int], list[int]]]) -> list[dict]:
    """Build a tiny in-memory source dataset of (input_ids, loss_mask) pairs."""
    return [{"input_ids": ids, "loss_mask": loss} for ids, loss in samples]


def test_build_packed_dataset_structure():
    """Packing resets position_ids per doc, sums seq_lens to T, and builds doc_remaining."""
    # Two short docs that fit one row of width 8, plus a third that starts a new row.
    src = _src(
        [
            ([10, 11, 12], [0, 1, 1]),  # doc A, len 3
            ([20, 21], [1, 1]),  # doc B, len 2  -> A+B = 5 <= 8, same pack
            ([30, 31, 32, 33, 34, 35, 36], [1, 1, 1, 1, 1, 1, 1]),  # doc C len 7 -> new pack
        ]
    )
    packs = build_packed_eagle3_dataset(src, packed_sequence_size=8, pad_token_id=0)
    assert len(packs) == 2

    pack0 = packs[0]
    # input_ids: A(3) + B(2) + pad(3) == 8
    assert pack0["input_ids"] == [10, 11, 12, 20, 21, 0, 0, 0]
    # position_ids reset per doc, pad continues the last doc (clamped).
    assert pack0["position_ids"][:5] == [0, 1, 2, 0, 1]
    # seq_lens: trailing pad folded into the final doc -> [3, 2+3] sums to 8.
    assert pack0["seq_lens"] == [3, 5]
    assert sum(pack0["seq_lens"]) == 8
    # doc_remaining: tokens after each slot within its (real) doc; pad -> 0.
    assert pack0["doc_remaining"] == [2, 1, 0, 1, 0, 0, 0, 0]
    # attention_mask: 1 for the 5 real tokens, 0 for the 3 pad.
    assert pack0["attention_mask"] == [1, 1, 1, 1, 1, 0, 0, 0]

    pack1 = packs[1]
    assert pack1["input_ids"] == [30, 31, 32, 33, 34, 35, 36, 0]
    assert pack1["seq_lens"] == [8]  # len 7 + 1 pad folded
    assert pack1["doc_remaining"] == [6, 5, 4, 3, 2, 1, 0, 0]


def test_pack_collate_pads_ragged_seq_lens():
    """_pack_collate stacks fixed-width fields and 0-pads ragged seq_lens."""
    src = _src([([1, 2, 3], [1, 1, 1]), ([4, 5], [1, 1]), ([6, 7, 8, 9], [1, 1, 1, 1])])
    packs = build_packed_eagle3_dataset(src, packed_sequence_size=6, pad_token_id=0)
    batch = _pack_collate(packs)
    bsz = len(packs)
    assert batch["input_ids"].shape == (bsz, 6)
    assert batch["doc_remaining"].shape == (bsz, 6)
    assert batch["position_ids"].shape == (bsz, 6)
    # seq_lens padded to [B, max_docs]; each row's nonzero entries sum to 6.
    assert batch["seq_lens"].shape[0] == bsz
    assert torch.all(batch["seq_lens"].sum(dim=1) == 6)


def test_block_causal_additive_mask():
    """Block-causal mask is in-doc lower-triangular and blocks cross-document attention."""
    seq_lens = torch.tensor([[3, 2]], dtype=torch.long)  # docs over T=5
    mask = build_block_causal_additive_mask(seq_lens, seq_length=5, dtype=torch.float32, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 5, 5)
    neg = torch.finfo(torch.float32).min
    m = mask[0, 0]
    # Doc A (0..2): causal within doc.
    assert m[0, 0] == 0 and m[2, 0] == 0 and m[2, 2] == 0
    assert m[0, 1] == neg  # future within doc A is masked
    # Doc B (3..4): causal within doc.
    assert m[3, 3] == 0 and m[4, 3] == 0 and m[4, 4] == 0
    # Cross-document is masked both directions.
    assert m[3, 0] == neg and m[3, 2] == neg  # doc B query cannot see doc A
    assert m[2, 3] == neg  # doc A query cannot see doc B


def test_seq_lens_to_cu_seqlens():
    """cu_seqlens is int32, monotonic, row-major, and sums to B*T; max_seqlen is the longest doc."""
    seq_lens = torch.tensor([[3, 2], [4, 1]], dtype=torch.long)  # B=2, T=5
    cu, max_seqlen = _seq_lens_to_cu_seqlens(seq_lens, seq_length=5)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 5, 9, 10]  # row-major doc lens [3,2,4,1]
    assert int(cu[-1]) == 2 * 5
    assert max_seqlen == 4


def _build_tiny_draft_model(attn_implementation: str = "eager") -> LlamaEagle3DraftModel:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    config.attn_implementation = attn_implementation
    return LlamaEagle3DraftModel(config).to(torch.float32)


def test_packed_draft_attention_isolates_documents():
    """Block-causal draft forward must isolate documents: perturbing only doc B's
    inputs leaves doc A's output bit-identical."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model("eager")
    draft.eval()

    seq_len = 6
    seq_lens = torch.tensor([[3, 3]], dtype=torch.long)  # doc A: 0..2, doc B: 3..5
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    input_ids = torch.randint(0, 16, (1, seq_len))
    projected = torch.randn(1, seq_len, draft.config.hidden_size)

    def run(ids, proj):
        return draft(
            input_ids=ids,
            projected_hidden_states=proj,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            position_ids=position_ids,
            cache_hidden=[[], []],
            seq_lens=seq_lens,
        )

    out_ref = run(input_ids, projected)

    # Perturb only document B (slots 3..5).
    ids_b = input_ids.clone()
    ids_b[:, 3:] = torch.randint(0, 16, (1, 3))
    proj_b = projected.clone()
    proj_b[:, 3:] = torch.randn(1, 3, draft.config.hidden_size)
    out_perturbed = run(ids_b, proj_b)

    # Document A (slots 0..2) output must be unchanged; document B may differ.
    torch.testing.assert_close(out_ref[:, :3], out_perturbed[:, :3])
    assert not torch.allclose(out_ref[:, 3:], out_perturbed[:, 3:])


def test_packed_trainer_forward_runs_and_backprops():
    """End-to-end packed trainer forward produces a finite loss and non-NaN grads."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model("eager")
    config = draft.config
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=3,
    )

    src = _src(
        [
            ([3, 4, 5, 6], [0, 1, 1, 1]),
            ([7, 8, 9], [0, 1, 1]),
            ([10, 11, 12, 13, 14], [0, 1, 1, 1, 1]),
        ]
    )
    packs = build_packed_eagle3_dataset(src, packed_sequence_size=8, pad_token_id=0)
    batch = _pack_collate(packs)
    bsz, seq_len = batch["input_ids"].shape

    target_logits = torch.randn(bsz, seq_len, config.vocab_size)
    aux_hidden_states = torch.randn(bsz, seq_len, config.hidden_size * 3)

    metrics = trainer(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        loss_mask=batch["loss_mask"],
        aux_hidden_states=aux_hidden_states,
        target_logits=target_logits,
        position_ids=batch["position_ids"],
        seq_lens=batch["seq_lens"],
        doc_remaining=batch["doc_remaining"],
    )
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    grads = [p.grad for p in trainer.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter to receive a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_packing_matches_padding_loss_and_grads():
    """Golden parity: packing N docs into one row == N padded single-doc rows.

    Both layouts run through the same target wrapper + draft trainer:
      * no packing: ``[D, T]``, each doc ``L`` real tokens padded to ``T``;
      * packing: the same docs as ``[1, T]`` (``D * L == T``) with per-document
        position_ids, block-causal attention, and ``doc_remaining`` gating.

    They supervise the identical (doc, position, TTT step) triples against
    identical targets, so loss and every gradient match (CPU/fp32, tight tol).
    Scaling L/T up and the tiny target to Qwen3-8B on GPU is a drop-in change.
    """
    hidden = 32
    vocab = 128
    num_docs = 4
    doc_len = 16
    total = num_docs * doc_len  # packed row width T

    target_config = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=8,  # deep enough for the default aux ids [1, 3, 4]
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=total,
    )
    torch.manual_seed(1)
    target = LlamaForCausalLM(target_config).to(torch.float32).eval()
    target_wrapper = HFEagle3TargetModel(target)

    draft_config = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=total,
    )
    draft_config.draft_vocab_size = vocab  # full vocab so every position is supervised
    draft_config.target_hidden_size = hidden
    draft_config.attn_implementation = "eager"

    def build_trainer():
        torch.manual_seed(123)  # identical draft init for both layouts
        draft = LlamaEagle3DraftModel(draft_config).to(torch.float32)
        selected_token_ids = torch.arange(vocab, dtype=torch.long)
        selected_token_mask = torch.ones(vocab, dtype=torch.bool)
        return Eagle3TrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=3,
        )

    torch.manual_seed(7)
    docs = [torch.randint(0, vocab, (doc_len,)) for _ in range(num_docs)]

    # Layout A: D padded single-document rows.
    ids_a = torch.zeros(num_docs, total, dtype=torch.long)
    loss_a = torch.zeros(num_docs, total, dtype=torch.long)
    attn_a = torch.zeros(num_docs, total, dtype=torch.long)
    for d in range(num_docs):
        ids_a[d, :doc_len] = docs[d]
        loss_a[d, :doc_len] = 1
        attn_a[d, :doc_len] = 1
    trainer_a = build_trainer()
    batch_a = target_wrapper.generate_batch(input_ids=ids_a, attention_mask=attn_a, loss_mask=loss_a)
    metrics_a = trainer_a(**batch_a.to_trainer_inputs())
    metrics_a.loss.backward()
    grads_a = {n: p.grad.clone() for n, p in trainer_a.named_parameters() if p.grad is not None}

    # Layout B: the same documents packed into one row (no padding).
    ids_b = torch.cat(docs).unsqueeze(0)
    loss_b = torch.ones(1, total, dtype=torch.long)
    attn_b = torch.ones(1, total, dtype=torch.long)
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    trainer_b = build_trainer()
    batch_b = target_wrapper.generate_batch(
        input_ids=ids_b,
        attention_mask=attn_b,
        loss_mask=loss_b,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    metrics_b = trainer_b(**batch_b.to_trainer_inputs())
    metrics_b.loss.backward()
    grads_b = {n: p.grad.clone() for n, p in trainer_b.named_parameters() if p.grad is not None}

    # Identical supervised-token count, loss, and gradients (tight CPU/fp32 tol).
    assert metrics_a.valid_tokens.item() == metrics_b.valid_tokens.item()
    torch.testing.assert_close(metrics_a.loss, metrics_b.loss, rtol=1e-4, atol=1e-5)
    assert set(grads_a) == set(grads_b)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-4, atol=1e-5, msg=f"grad mismatch: {name}")


def test_doc_remaining_gating_masks_cross_document_supervision():
    """At TTT step k, supervision is dropped where k >= doc_remaining (cross-doc target)."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model("eager")
    config = draft.config
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=4,
    )

    # Single row, two docs of length 4 each (no padding): T=8.
    seq_len = 8
    seq_lens = torch.tensor([[4, 4]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    doc_remaining = torch.tensor([[3, 2, 1, 0, 3, 2, 1, 0]], dtype=torch.long)
    input_ids = torch.randint(0, 16, (1, seq_len))
    # Supervise every position; bias the target argmax into the draft vocab so
    # ``selected_token_mask`` always passes and masking is driven purely by the
    # loss_mask shift + doc_remaining gating.
    loss_mask = torch.ones(1, seq_len, dtype=torch.long)
    target_logits = torch.randn(1, seq_len, config.vocab_size)
    target_logits[..., : config.draft_vocab_size] += 50.0
    aux_hidden_states = torch.randn(1, seq_len, config.hidden_size * 3)

    def valid_count(doc_rem):
        return trainer(
            input_ids=input_ids,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_rem,
        ).valid_tokens.item()

    # Without gating the loss_mask only shrinks via the per-step left-shift
    # (zero-filled tail): step k supervises 8-k slots -> 8+7+6+5 = 26.
    assert valid_count(None) == 26
    # With gating, slot t at step k is kept only while k < doc_remaining[t]:
    #   step0: dr>0 at {0,1,2,4,5,6} -> 6
    #   step1: dr>1 at {0,1,4,5}     -> 4
    #   step2: dr>2 at {0,4}         -> 2
    #   step3: dr>3                  -> 0
    # total 12, strictly fewer than 26 -- every cross-document target is dropped.
    assert valid_count(doc_remaining) == 12


class _RecordingFlashTarget(torch.nn.Module):
    """Minimal causal-LM stub reporting ``_attn_implementation='flash_attention_2'``.

    Records the ``attention_mask`` / ``position_ids`` it is forwarded so packing's
    FlashAttention dispatch can be asserted on CPU (real FA kernels need a GPU).
    """

    def __init__(self, num_layers: int = 8, hidden: int = 16, vocab: int = 32):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2", num_hidden_layers=num_layers)
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        # _get_transformer_layers() looks for ``self.model.layers``; Identity layers
        # let the aux forward-hooks fire without a real attention implementation.
        self.model = SimpleNamespace(layers=torch.nn.ModuleList(torch.nn.Identity() for _ in range(num_layers)))
        self.received: dict = {}

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.received = {"attention_mask": attention_mask, "position_ids": position_ids}
        hidden = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], self.embed_tokens.num_embeddings)
        return SimpleNamespace(logits=logits)


def test_packed_flash_target_passes_position_ids_not_4d_mask():
    """FlashAttention packing must forward ``attention_mask=None`` + per-doc
    position_ids (FA infers cu_seqlens from them); a 4D mask would blow up its
    unpad gather."""
    target = _RecordingFlashTarget()
    wrapper = HFEagle3TargetModel(target)

    seq_len = 6
    input_ids = torch.randint(0, 32, (1, seq_len))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    seq_lens = torch.tensor([[3, 3]], dtype=torch.long)
    doc_remaining = torch.tensor([[2, 1, 0, 2, 1, 0]], dtype=torch.long)

    wrapper.generate_batch(
        input_ids=input_ids,
        attention_mask=torch.ones(1, seq_len, dtype=torch.long),
        loss_mask=torch.ones(1, seq_len, dtype=torch.long),
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )

    # The target saw no explicit mask (FA handles causality) and the per-doc positions.
    assert target.received["attention_mask"] is None
    torch.testing.assert_close(target.received["position_ids"], position_ids)


def test_packed_flash_target_rejects_batch_gt_1():
    """transformers only packs from position_ids at batch size 1, so a FA target
    must reject micro_batch_size > 1 instead of silently leaking across documents."""
    target = _RecordingFlashTarget()
    wrapper = HFEagle3TargetModel(target)

    seq_len = 6
    input_ids = torch.randint(0, 32, (2, seq_len))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]], dtype=torch.long)
    seq_lens = torch.tensor([[3, 3], [3, 3]], dtype=torch.long)
    doc_remaining = torch.tensor([[2, 1, 0, 2, 1, 0], [2, 1, 0, 2, 1, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="micro_batch_size=1"):
        wrapper.generate_batch(
            input_ids=input_ids,
            attention_mask=torch.ones(2, seq_len, dtype=torch.long),
            loss_mask=torch.ones(2, seq_len, dtype=torch.long),
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
