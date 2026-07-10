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

"""Sequence-packing tests for the EAGLE-1 / EAGLE-2 draft, target wrapper, and trainer."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel

_HIDDEN = 16
_VOCAB = 64


def _draft_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=_VOCAB,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_num_hidden_layers = 1
    return config


def _build_draft() -> LlamaEagleDraftModel:
    torch.manual_seed(0)
    return LlamaEagleDraftModel(_draft_config()).to(torch.float32)


def _build_target(num_hidden_layers: int = 4) -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=_VOCAB,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    config.torch_dtype = torch.float32
    torch.manual_seed(1)
    return LlamaForCausalLM(config).to(torch.float32).eval()


def _uniform_pack_meta(num_docs: int, doc_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(position_ids, seq_lens, doc_remaining)`` for ``num_docs`` equal docs over one row."""
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    return position_ids, seq_lens, doc_remaining


def test_packed_single_doc_matches_unpacked_forward():
    """A packed row that is one full-width document must match the plain causal forward."""
    draft = _build_draft().eval()
    batch, seq = 1, 6
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    target_hidden = torch.randn(batch, seq, _HIDDEN)
    attn = torch.ones(batch, seq, dtype=torch.long)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    out_unpacked = draft(input_ids, target_hidden, attn, position_ids=position_ids)
    out_packed = draft(
        input_ids, target_hidden, attn, position_ids=position_ids, seq_lens=torch.tensor([[seq]], dtype=torch.long)
    )
    torch.testing.assert_close(out_packed, out_unpacked)


def test_packed_draft_attention_isolates_documents():
    """Block-causal packing must isolate documents in the draft forward."""
    torch.manual_seed(0)
    draft = _build_draft().eval()
    seq_len = 6
    seq_lens = torch.tensor([[3, 3]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    input_ids = torch.randint(0, _VOCAB, (1, seq_len))
    target_hidden = torch.randn(1, seq_len, _HIDDEN)

    def run(ids, th):
        return draft(ids, th, torch.ones(1, seq_len, dtype=torch.long), position_ids=position_ids, seq_lens=seq_lens)

    out_ref = run(input_ids, target_hidden)
    ids_b = input_ids.clone()
    ids_b[:, 3:] = torch.randint(0, _VOCAB, (1, 3))
    th_b = target_hidden.clone()
    th_b[:, 3:] = torch.randn(1, 3, _HIDDEN)
    out_perturbed = run(ids_b, th_b)

    torch.testing.assert_close(out_ref[:, :3], out_perturbed[:, :3])  # doc A unchanged
    assert not torch.allclose(out_ref[:, 3:], out_perturbed[:, 3:])  # doc B changed


def test_packing_requires_position_ids():
    """Packing without per-document position_ids must fail loud in the draft."""
    draft = _build_draft().eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    th = torch.randn(1, 6, _HIDDEN)
    with pytest.raises(ValueError, match="per-document position_ids"):
        draft(ids, th, torch.ones(1, 6, dtype=torch.long), seq_lens=torch.tensor([[3, 3]], dtype=torch.long))


def test_recipe_packing_kwargs_helper():
    """The recipe helper extracts packing metadata only when the batch is packed."""
    from nemo_automodel.recipes.llm.train_eagle1 import _packing_kwargs

    assert _packing_kwargs({"input_ids": torch.zeros(1, 4)}) == {}
    packed = {
        "input_ids": torch.zeros(1, 4),
        "position_ids": torch.zeros(1, 4),
        "seq_lens": torch.tensor([[4]]),
        "doc_remaining": torch.zeros(1, 4),
    }
    assert set(_packing_kwargs(packed)) == {"position_ids", "seq_lens", "doc_remaining"}


def test_target_wrapper_packing_isolates_documents_and_carries_metadata():
    """generate_batch under packing must isolate the target's hidden states per document
    and carry position_ids / seq_lens / doc_remaining through to the trainer."""
    torch.manual_seed(2)
    target = _build_target()
    wrapper = HFEagleTargetModel(target)
    num_docs, doc_len = 2, 6
    seq = num_docs * doc_len
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    input_ids = torch.randint(0, _VOCAB, (1, seq))
    attn = torch.ones(1, seq, dtype=torch.long)
    loss = torch.ones(1, seq, dtype=torch.long)

    def input_hidden(ids):
        b = wrapper.generate_batch(
            ids, attn, loss, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
        )
        assert b.position_ids is position_ids and b.seq_lens is seq_lens and b.doc_remaining is doc_remaining
        return b.input_hidden_states

    ref = input_hidden(input_ids)
    ids_b = input_ids.clone()
    ids_b[:, doc_len:] = torch.randint(0, _VOCAB, (1, doc_len))
    pert = input_hidden(ids_b)
    torch.testing.assert_close(ref[:, :doc_len], pert[:, :doc_len])  # doc A hidden unchanged
    assert not torch.allclose(ref[:, doc_len:], pert[:, doc_len:])  # doc B hidden changed


def _build_trainer(target: LlamaForCausalLM) -> EagleTrainerModule:
    torch.manual_seed(123)  # identical draft init across layouts
    draft = LlamaEagleDraftModel(_draft_config()).to(torch.float32)
    return EagleTrainerModule(draft, target_lm_head=target.lm_head, hidden_loss_weight=1.0, token_loss_weight=0.1)


def test_packing_matches_padding_loss_and_grads():
    """Golden parity: packing N docs into one row == N padded single-doc rows.

    Both layouts run through the real ``HFEagleTargetModel`` + ``EagleTrainerModule``:
      * padding: ``[D, T]`` with each doc's ``L`` real tokens padded to ``T`` (the
        wrapper's global left-shift drops each doc's last token via padding);
      * packing: the same docs as ``[1, T]`` with per-document position_ids, the
        block-causal target/draft masks, and the ``doc_remaining`` gate.
    They supervise the identical (doc, position) pairs against identical targets, so
    loss and every draft gradient match to tight CPU/fp32 tolerance.
    """
    num_docs, doc_len = 3, 6
    total = num_docs * doc_len
    target = _build_target()
    wrapper = HFEagleTargetModel(target)

    torch.manual_seed(7)
    docs = [torch.randint(0, _VOCAB, (doc_len,)) for _ in range(num_docs)]

    # Layout A: D padded single-document rows.
    ids_a = torch.zeros(num_docs, total, dtype=torch.long)
    loss_a = torch.zeros(num_docs, total, dtype=torch.long)
    attn_a = torch.zeros(num_docs, total, dtype=torch.long)
    for d in range(num_docs):
        ids_a[d, :doc_len] = docs[d]
        loss_a[d, :doc_len] = 1
        attn_a[d, :doc_len] = 1
    trainer_a = _build_trainer(target)
    batch_a = wrapper.generate_batch(ids_a, attn_a, loss_a)
    metrics_a = trainer_a(
        input_ids=batch_a.input_ids,
        attention_mask=batch_a.attention_mask,
        loss_mask=batch_a.loss_mask,
        input_hidden_states=batch_a.input_hidden_states,
        target_hidden_states=batch_a.target_hidden_states,
        target_logits=batch_a.target_logits,
    )
    metrics_a.loss.backward()
    grads_a = {n: p.grad.clone() for n, p in trainer_a.named_parameters() if p.grad is not None}

    # Layout B: the same documents packed into one row.
    ids_b = torch.cat(docs).unsqueeze(0)
    loss_b = torch.ones(1, total, dtype=torch.long)
    attn_b = torch.ones(1, total, dtype=torch.long)
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    trainer_b = _build_trainer(target)
    batch_b = wrapper.generate_batch(
        ids_b, attn_b, loss_b, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
    )
    metrics_b = trainer_b(
        input_ids=batch_b.input_ids,
        attention_mask=batch_b.attention_mask,
        loss_mask=batch_b.loss_mask,
        input_hidden_states=batch_b.input_hidden_states,
        target_hidden_states=batch_b.target_hidden_states,
        target_logits=batch_b.target_logits,
        position_ids=batch_b.position_ids,
        seq_lens=batch_b.seq_lens,
        doc_remaining=batch_b.doc_remaining,
    )
    metrics_b.loss.backward()
    grads_b = {n: p.grad.clone() for n, p in trainer_b.named_parameters() if p.grad is not None}

    assert metrics_a.valid_tokens.item() == metrics_b.valid_tokens.item()
    torch.testing.assert_close(metrics_a.loss, metrics_b.loss, rtol=1e-4, atol=1e-5)
    assert set(grads_a) == set(grads_b)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-4, atol=1e-5, msg=f"grad mismatch: {name}")
