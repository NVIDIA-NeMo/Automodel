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

"""Sequence-packing tests for the DSpark draft (common helpers, draft, target, recipe)."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components.speculative.dspark import Qwen3DSparkModel, build_draft_config
from nemo_automodel.components.speculative.dspark.common import (
    build_anchor_candidate_mask,
    build_eval_mask,
    context_doc_ids,
    create_position_ids,
)
from nemo_automodel.components.speculative.dspark.core import DSparkTrainerModule
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel

VOCAB = 256
HIDDEN = 64
TARGET_LAYER_IDS = [1, 3]
BLOCK_SIZE = 4


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _target_config(num_hidden_layers: int = 4) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


def _build_draft():
    margs = _Args(
        num_draft_layers=2,
        target_layer_ids=TARGET_LAYER_IDS,
        block_size=BLOCK_SIZE,
        mask_token_id=5,
        num_anchors=8,
        markov_rank=16,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    draft_config = build_draft_config(_target_config(), margs)
    draft_config._attn_implementation = "sdpa"
    model = Qwen3DSparkModel(draft_config).to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def _uniform_pack_meta(num_docs: int, doc_len: int):
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    return position_ids, seq_lens, doc_remaining


def test_context_doc_ids_from_seq_lens():
    doc_id = context_doc_ids(torch.tensor([[3, 2, 0]]), seq_len=5, device=torch.device("cpu"))
    assert doc_id.tolist() == [[0, 0, 0, 1, 1]]


def test_anchor_candidate_mask_requires_first_target_in_document():
    """A packed anchor is valid only if its first target (anchor+1) is in its document."""
    # Two docs of length 3 over S=6; loss_mask all ones.
    _, _, doc_remaining = _uniform_pack_meta(num_docs=2, doc_len=3)
    loss_mask = torch.ones(1, 6)
    valid = build_anchor_candidate_mask(seq_len=6, loss_mask=loss_mask, doc_remaining=doc_remaining)
    # Candidates are positions 0..4. doc0 = 0,1,2 (position 2 is doc0's last token: its
    # first target 3 is in the next document -> invalid). doc1 = 3,4,5 (positions 3 and 4
    # each have an in-document next token; 5 is past the candidate range).
    assert valid[0].tolist() == [True, True, False, True, True]


def test_create_position_ids_uses_per_document_positions():
    position_ids, _, _ = _uniform_pack_meta(num_docs=2, doc_len=5)  # [[0..4, 0..4]]
    anchors = torch.tensor([[1, 6]], dtype=torch.long)  # global 1 (doc0 pos1), global 6 (doc1 pos1)
    pos = create_position_ids(anchors, BLOCK_SIZE, position_ids).view(1, 2, BLOCK_SIZE)
    assert pos[0, 0].tolist() == [1, 2, 3, 4]
    assert pos[0, 1].tolist() == [1, 2, 3, 4]


def test_eval_mask_truncates_block_at_document_boundary():
    """A block whose targets cross a document boundary is truncated to the in-document prefix."""
    seq_len = 8  # two docs of length 4: doc0 = 0..3, doc1 = 4..7
    _, _, doc_remaining = _uniform_pack_meta(num_docs=2, doc_len=4)
    loss_mask = torch.ones(1, seq_len)
    anchors = torch.tensor([[2]], dtype=torch.long)  # doc0, one in-doc target left (pos 3)
    label_offsets = torch.arange(1, BLOCK_SIZE + 1).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + label_offsets
    safe = label_indices.clamp(max=seq_len - 1)
    eval_mask = build_eval_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
        label_indices=label_indices,
        safe_label_indices=safe,
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        doc_remaining=doc_remaining,
        anchor_positions=anchors,
    )
    # Only the first target (pos 3, in doc0) is supervised; 4/5/6 are in doc1.
    assert eval_mask[0, 0].tolist() == [True, False, False, False]


def test_packed_single_doc_matches_unpacked():
    """A single full-width document must reproduce the unpacked forward (seeded anchors)."""
    trainer = DSparkTrainerModule(_build_draft(), loss_decay_gamma=4.0)
    b, T = 1, 20
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (b, T), generator=gen)
    hidden = torch.randn(b, T, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen)
    last = torch.randn(b, T, HIDDEN, generator=gen)
    loss_mask = torch.ones(b, T, dtype=torch.uint8)
    position_ids = torch.arange(T).unsqueeze(0)
    seq_lens = torch.tensor([[T]], dtype=torch.long)
    doc_remaining = torch.arange(T - 1, -1, -1).unsqueeze(0)

    with torch.no_grad():
        torch.manual_seed(7)
        out_unpacked = trainer(
            input_ids=input_ids, target_hidden_states=hidden, loss_mask=loss_mask, target_last_hidden_states=last
        )
        torch.manual_seed(7)
        out_packed = trainer(
            input_ids=input_ids,
            target_hidden_states=hidden,
            loss_mask=loss_mask,
            target_last_hidden_states=last,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
    torch.testing.assert_close(out_packed.loss, out_unpacked.loss)


def test_packed_two_doc_trainer_runs_and_backprops():
    trainer = DSparkTrainerModule(_build_draft(), loss_decay_gamma=4.0)
    doc_len, num_docs = 10, 2
    seq = doc_len * num_docs
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(3)
    out = trainer(
        input_ids=torch.randint(0, VOCAB, (1, seq), generator=gen),
        target_hidden_states=torch.randn(1, seq, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen),
        loss_mask=torch.ones(1, seq, dtype=torch.uint8),
        target_last_hidden_states=torch.randn(1, seq, HIDDEN, generator=gen),
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def _build_target(num_hidden_layers: int = 4) -> Qwen3ForCausalLM:
    cfg = _target_config(num_hidden_layers)
    cfg._attn_implementation = "eager"
    torch.manual_seed(1)
    return Qwen3ForCausalLM(cfg).to(torch.float32).eval()


def test_target_wrapper_packing_requires_position_ids():
    wrapper = HFDSparkTargetModel(_build_target(), target_layer_ids=TARGET_LAYER_IDS)
    ids = torch.randint(0, VOCAB, (1, 6))
    with pytest.raises(ValueError, match="per-document position_ids"):
        wrapper.generate_batch(
            ids,
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, dtype=torch.long),
            seq_lens=torch.tensor([[3, 3]], dtype=torch.long),
        )


def test_target_wrapper_packing_isolates_documents_and_carries_metadata():
    wrapper = HFDSparkTargetModel(_build_target(), target_layer_ids=TARGET_LAYER_IDS)
    doc_len, num_docs = 3, 4
    seq = num_docs * doc_len
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    attn = torch.ones(1, seq, dtype=torch.long)
    loss = torch.ones(1, seq, dtype=torch.long)
    ids = torch.randint(0, VOCAB, (1, seq))

    def hid(x):
        b = wrapper.generate_batch(
            x, attn, loss, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
        )
        assert b.seq_lens is seq_lens and b.position_ids is position_ids and b.doc_remaining is doc_remaining
        return b.target_hidden_states

    ref = hid(ids)
    ids_b = ids.clone()
    ids_b[:, doc_len:] = torch.randint(0, VOCAB, (1, seq - doc_len))
    pert = hid(ids_b)
    torch.testing.assert_close(ref[:, :doc_len], pert[:, :doc_len])  # doc 0 hidden unchanged
    assert not torch.allclose(ref[:, doc_len:], pert[:, doc_len:])


def test_recipe_packing_helpers_and_gates():
    from nemo_automodel.recipes.llm.train_dspark import _packing_kwargs, _validate_packing_gates

    assert _packing_kwargs({"input_ids": torch.zeros(1, 4)}) == {}
    packed = {
        "input_ids": torch.zeros(1, 4),
        "position_ids": torch.zeros(1, 4),
        "seq_lens": torch.tensor([[4]]),
        "doc_remaining": torch.zeros(1, 4),
    }
    assert set(_packing_kwargs(packed)) == {"position_ids", "seq_lens", "doc_remaining"}

    _validate_packing_gates(cp_size=1, target_attn_impl="sdpa", micro_batch_size=4)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _validate_packing_gates(cp_size=2, target_attn_impl="sdpa", micro_batch_size=1)
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        _validate_packing_gates(cp_size=1, target_attn_impl="flash_attention_2", micro_batch_size=2)
