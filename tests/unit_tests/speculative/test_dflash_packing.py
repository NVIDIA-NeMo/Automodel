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

"""Sequence-packing tests for the DFlash draft (mask, anchor sampling, trainer, target, recipe)."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.attention.dflash_mask import create_dflash_sdpa_mask
from nemo_automodel.components.speculative.dflash.core import (
    DFlashTrainerModule,
    _context_doc_ids,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.target import HFDFlashTargetModel

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1


def _build_trainer(num_anchors=8, attention_backend="sdpa"):
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
    torch.manual_seed(0)
    draft = Qwen3DFlashDraftModel(cfg)
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
    )


def _uniform_pack_meta(num_docs: int, doc_len: int):
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    return position_ids, seq_lens, doc_remaining


def test_context_doc_ids_from_seq_lens():
    seq_lens = torch.tensor([[3, 2, 0]], dtype=torch.long)  # docs of len 3 and 2 over S=5
    doc_id = _context_doc_ids(seq_lens, seq_len=5, device=torch.device("cpu"))
    assert doc_id.tolist() == [[0, 0, 0, 1, 1]]


def test_sdpa_mask_restricts_context_to_anchor_document():
    """A block's context prefix must not cross into an earlier document."""
    ctx_len, bs = 8, BLOCK_SIZE  # two docs of length 4: doc0 = 0..3, doc1 = 4..7
    anchors = torch.tensor([[2, 6]], dtype=torch.long)  # anchor in doc0, anchor in doc1
    keep = torch.ones(1, 2, dtype=torch.bool)
    ctx_doc_id = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    anchor_doc_id = torch.tensor([[0, 1]], dtype=torch.long)
    m = create_dflash_sdpa_mask(
        anchors,
        keep,
        ctx_len,
        bs,
        torch.device("cpu"),
        torch.float32,
        ctx_doc_id=ctx_doc_id,
        anchor_doc_id=anchor_doc_id,
    )[0, 0]
    neg = float("-inf")
    # Block 1 (doc1, anchor=6) first query row = block index 1 * bs = 4.
    row = bs
    assert m[row, 4] == 0 and m[row, 5] == 0  # doc1 context before the anchor is visible
    assert m[row, 0] == neg and m[row, 3] == neg  # doc0 context is masked (different document)
    # Block 0 (doc0, anchor=2) first query row = 0: sees doc0 context < 2, never doc1.
    assert m[0, 0] == 0 and m[0, 1] == 0
    assert m[0, 4] == neg


def test_flex_mask_mod_matches_sdpa_document_restriction():
    """The flex_attention (default backend) doc-gating must match the dense SDPA mask.

    FlexAttention's kernel is CUDA-only, but its ``mask_mod`` can be materialised on
    CPU with ``create_mask``; assert the resulting per-token mask equals the SDPA mask
    so the production-default flex doc-gating branch is covered.
    """
    from torch.nn.attention.flex_attention import create_mask

    from nemo_automodel.components.attention.dflash_mask import build_dflash_mask_mod

    ctx_len, bs, num_blocks = 8, BLOCK_SIZE, 2
    anchors = torch.tensor([[2, 6]], dtype=torch.long)
    keep = torch.ones(1, num_blocks, dtype=torch.bool)
    ctx_doc_id = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
    anchor_doc_id = torch.tensor([[0, 1]], dtype=torch.long)
    q_len, kv_len = num_blocks * bs, ctx_len + num_blocks * bs

    mod = build_dflash_mask_mod(
        anchors, keep, ctx_len, bs, num_blocks, causal=False, ctx_doc_id=ctx_doc_id, anchor_doc_id=anchor_doc_id
    )
    flex_bool = create_mask(mod, 1, 1, q_len, kv_len, device="cpu")[0, 0]
    sdpa = create_dflash_sdpa_mask(
        anchors,
        keep,
        ctx_len,
        bs,
        torch.device("cpu"),
        torch.float32,
        ctx_doc_id=ctx_doc_id,
        anchor_doc_id=anchor_doc_id,
    )[0, 0]
    sdpa_bool = sdpa == 0
    assert torch.equal(flex_bool, sdpa_bool)
    # And the doc restriction is actually present: doc1 block cannot see doc0 context.
    assert flex_bool[bs, 4] and not flex_bool[bs, 0]


def test_anchor_sampling_keeps_blocks_in_document():
    """With doc_remaining, no sampled anchor's block may cross a document boundary."""
    trainer = _build_trainer()
    doc_len, num_docs = 6, 3
    seq = doc_len * num_docs
    _, _, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    loss_mask = torch.ones(1, seq)
    torch.manual_seed(1)
    anchors, keep = trainer._sample_anchor_positions(seq, loss_mask, torch.device("cpu"), doc_remaining=doc_remaining)
    kept = anchors[keep]
    # Each kept anchor must have at least block_size-1 real tokens left in its doc,
    # i.e. its within-document offset is <= doc_len - block_size.
    within_doc = kept % doc_len
    assert bool((within_doc <= doc_len - BLOCK_SIZE).all())


def test_create_position_ids_uses_per_document_positions():
    trainer = _build_trainer()
    doc_len, num_docs = 5, 2
    position_ids, _, _ = _uniform_pack_meta(num_docs, doc_len)  # [[0..4, 0..4]]
    anchors = torch.tensor([[1, 6]], dtype=torch.long)  # global 1 (doc0 pos1), global 6 (doc1 pos1)
    pos = trainer._create_position_ids(anchors, position_ids).view(1, 2, BLOCK_SIZE)
    # Both anchors are at per-document position 1, so both blocks start at 1.
    assert pos[0, 0].tolist() == [1, 2, 3, 4]
    assert pos[0, 1].tolist() == [1, 2, 3, 4]


def test_packed_single_doc_matches_unpacked():
    """A single full-width document (seq_lens=[[T]]) must reproduce the unpacked forward."""
    trainer = _build_trainer()
    bsz, T = 1, 24
    input_ids = torch.randint(0, VOCAB - 1, (bsz, T))
    hidden = torch.randn(bsz, T, len(TARGET_LAYER_IDS) * HIDDEN)
    loss_mask = torch.ones(bsz, T)
    position_ids = torch.arange(T).unsqueeze(0)
    seq_lens = torch.tensor([[T]], dtype=torch.long)
    doc_remaining = torch.arange(T - 1, -1, -1).unsqueeze(0)

    torch.manual_seed(42)
    out_unpacked = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    torch.manual_seed(42)
    out_packed = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert out_packed.valid_tokens.item() == out_unpacked.valid_tokens.item()
    torch.testing.assert_close(out_packed.loss, out_unpacked.loss)


def test_packed_two_doc_trainer_runs_and_backprops():
    trainer = _build_trainer()
    doc_len, num_docs = 8, 2
    seq = doc_len * num_docs
    position_ids, seq_lens, doc_remaining = _uniform_pack_meta(num_docs, doc_len)
    input_ids = torch.randint(0, VOCAB - 1, (1, seq))
    hidden = torch.randn(1, seq, len(TARGET_LAYER_IDS) * HIDDEN)
    loss_mask = torch.ones(1, seq)
    torch.manual_seed(3)
    out = trainer(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert torch.isfinite(out.loss) and out.valid_tokens.item() > 0
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def _build_target(num_hidden_layers: int = 6) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(1)
    return LlamaForCausalLM(cfg).to(torch.float32).eval()


def test_target_wrapper_packing_requires_position_ids():
    wrapper = HFDFlashTargetModel(_build_target(), target_layer_ids=TARGET_LAYER_IDS)
    ids = torch.randint(0, VOCAB, (1, 6))
    with pytest.raises(ValueError, match="per-document position_ids"):
        wrapper.generate_batch(
            ids,
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, dtype=torch.long),
            seq_lens=torch.tensor([[3, 3]], dtype=torch.long),
        )


def test_target_wrapper_packing_isolates_documents_and_carries_metadata():
    wrapper = HFDFlashTargetModel(_build_target(), target_layer_ids=TARGET_LAYER_IDS)
    doc_len, num_docs = 2, 6
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
        return b.hidden_states

    ref = hid(ids)
    ids_b = ids.clone()
    ids_b[:, doc_len:] = torch.randint(0, VOCAB, (1, seq - doc_len))
    pert = hid(ids_b)
    torch.testing.assert_close(ref[:, :doc_len], pert[:, :doc_len])  # doc 0 hidden unchanged
    assert not torch.allclose(ref[:, doc_len:], pert[:, doc_len:])


def test_recipe_packing_helpers_and_gates():
    from nemo_automodel.recipes.llm.train_dflash import _packing_kwargs, _validate_packing_gates

    assert _packing_kwargs({"input_ids": torch.zeros(1, 4)}) == {}
    packed = {
        "input_ids": torch.zeros(1, 4),
        "position_ids": torch.zeros(1, 4),
        "seq_lens": torch.tensor([[4]]),
        "doc_remaining": torch.zeros(1, 4),
    }
    assert set(_packing_kwargs(packed)) == {"position_ids", "seq_lens", "doc_remaining"}

    _validate_packing_gates(cp_size=1, target_attn_impl="sdpa", micro_batch_size=4)  # ok
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _validate_packing_gates(cp_size=2, target_attn_impl="sdpa", micro_batch_size=1)
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        _validate_packing_gates(cp_size=1, target_attn_impl="flash_attention_2", micro_batch_size=2)
