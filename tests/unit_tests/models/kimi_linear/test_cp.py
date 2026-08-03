# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib

import pytest
import torch

from nemo_automodel.components.models.kimi_linear.cp import (
    _PAD_DOC_ID,
    KimiPackedContext,
    build_document_causal_mask,
    doc_ids_from_cu_seqlens,
    doc_ids_from_seq_lens,
    segment_cu_seqlens,
    shard_batch_for_kimi_cp,
)


class _FakeCPMesh:
    """Minimal stand-in for a one-dimensional CP device mesh."""

    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


def _batch(seq_len: int = 8, batch_size: int = 1) -> dict:
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def test_segment_cu_seqlens_splits_documents_and_padding():
    doc_ids = torch.tensor([1, 1, 1, 2, 2, 0, 0], dtype=torch.int32)

    assert segment_cu_seqlens(doc_ids).tolist() == [0, 3, 5, 7]


def test_segment_cu_seqlens_single_document_covers_row():
    doc_ids = torch.ones(6, dtype=torch.int32)

    assert segment_cu_seqlens(doc_ids).tolist() == [0, 6]


def test_doc_ids_from_seq_lens_marks_documents_and_trailing_padding():
    seq_lens = torch.tensor([[3, 2, -1000]], dtype=torch.int32)

    doc_ids = doc_ids_from_seq_lens(seq_lens, seq_len=7)

    assert doc_ids.tolist() == [[1, 1, 1, 2, 2, 0, 0]]


def test_doc_ids_from_cu_seqlens_matches_segment_boundaries():
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    doc_ids = doc_ids_from_cu_seqlens(cu_seqlens, seq_len=7)

    assert doc_ids.tolist() == [[1, 1, 1, 2, 2, 0, 0]]


def test_packed_context_row_cu_seqlens_is_cached():
    context = KimiPackedContext(doc_ids=torch.tensor([[1, 1, 2, 2]], dtype=torch.int32))

    first = context.row_cu_seqlens(0)
    second = context.row_cu_seqlens(0)

    assert first[0] is second[0]
    assert first[0].tolist() == [0, 2, 4]
    assert first[1].device.type == "cpu"


def test_document_causal_mask_blocks_cross_document_and_padding():
    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.int32)

    mask = build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)
    allowed = mask[0, 0] == 0

    assert allowed.tolist() == [
        [True, False, False, False],  # first token of document 1
        [True, True, False, False],  # second token of document 1
        [False, False, True, False],  # document 2 cannot see document 1
        [True, False, False, False],  # padding query is redirected to position 0
    ]


def test_document_causal_mask_shifts_queries_by_global_offset():
    q_doc_ids = torch.tensor([[1, 1]], dtype=torch.int32)
    kv_doc_ids = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32)

    mask = build_document_causal_mask(q_doc_ids, kv_doc_ids, q_global_start=2, dtype=torch.float32)
    allowed = mask[0, 0] == 0

    assert allowed.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_shard_batch_keeps_contiguous_slice_per_rank():
    shards = []
    for rank in range(2):
        _, batch, _ = shard_batch_for_kimi_cp(_FakeCPMesh(2, rank), None, _batch(seq_len=8))
        shards.append(batch)

    assert shards[0]["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert shards[1]["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert shards[0]["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert shards[1]["position_ids"].tolist() == [[4, 5, 6, 7]]
    assert shards[1]["kimi_packed_context"].seq_start == 4
    # Every rank keeps the full document map so attention can mask gathered keys.
    assert shards[1]["kimi_packed_context"].doc_ids.shape == (1, 8)


def test_shard_batch_pads_sequence_up_to_cp_size():
    context_factory, batch, layout = shard_batch_for_kimi_cp(
        _FakeCPMesh(4, 3), None, _batch(seq_len=6), padding_token_id=7
    )

    assert isinstance(context_factory(), contextlib.nullcontext)
    assert batch["input_ids"].shape == (1, 2)
    # Sequence 6 -> 8, so the last rank owns one real and one padded token.
    assert batch["input_ids"].tolist() == [[7, 7]]
    assert batch["labels"].tolist() == [[-100, -100]]
    assert batch["kimi_packed_context"].doc_ids[0, -2:].tolist() == [_PAD_DOC_ID, _PAD_DOC_ID]
    # The sharder's token verbs pad side tensors against this reported layout.
    assert (layout.original_seq_len, layout.padded_seq_len) == (6, 8)


def test_shard_batch_derives_documents_from_indexed_mask():
    batch = _batch(seq_len=8)
    batch["attention_mask"] = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 0]], dtype=torch.int32)

    _, sharded, _ = shard_batch_for_kimi_cp(_FakeCPMesh(2, 0), None, batch)

    assert "attention_mask" not in sharded
    assert sharded["kimi_packed_context"].doc_ids.tolist() == [[1, 1, 1, 2, 2, 2, 2, 0]]
    assert sharded["kimi_packed_context"].row_cu_seqlens(0)[0].tolist() == [0, 3, 7, 8]


def test_shard_batch_drops_thd_metadata_that_no_longer_matches():
    batch = _batch(seq_len=8)
    batch["seq_lens"] = torch.tensor([[5, 3]], dtype=torch.int32)
    batch["cu_seqlens"] = torch.tensor([0, 5, 8], dtype=torch.int32)
    batch["qkv_format"] = "thd"

    _, sharded, _ = shard_batch_for_kimi_cp(_FakeCPMesh(2, 1), None, batch)

    assert not {"seq_lens", "cu_seqlens", "qkv_format"} & set(sharded)
    assert sharded["kimi_packed_context"].doc_ids.tolist() == [[1, 1, 1, 1, 1, 2, 2, 2]]


def test_shard_batch_without_cp_mesh_is_a_no_op():
    batch = _batch(seq_len=6)

    _, sharded, _ = shard_batch_for_kimi_cp(None, None, batch)

    assert sharded["input_ids"].shape == (1, 6)
    assert not sharded["kimi_packed_context"].cp_enabled
    assert sharded["kimi_packed_context"].local_doc_ids.shape == (1, 6)


def test_local_doc_ids_selects_the_rank_slice():
    context = KimiPackedContext(
        doc_ids=torch.tensor([[1, 1, 2, 2]], dtype=torch.int32),
        seq_start=2,
        cp_size=2,
    )

    assert context.cp_enabled
    assert context.local_doc_ids.tolist() == [[2, 2]]


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_shard_batch_shards_loss_mask_with_labels(cp_size):
    loss_mask = torch.ones(1, 8, dtype=torch.long)

    _, sharded, _ = shard_batch_for_kimi_cp(_FakeCPMesh(cp_size, 0), None, _batch(seq_len=8), loss_mask=loss_mask)

    assert sharded["loss_mask"].shape == (1, 8 // cp_size)


def test_kimi_model_reports_cp_and_packing_support():
    from nemo_automodel._transformers.capabilities import ModelSupports
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM
    from tests.unit_tests.models.kimi_linear.test_model import _tiny_kimi_config

    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=BackendConfig(attn="eager"))
    supports = ModelSupports(model, None)

    assert supports.supports_cp is True
    assert supports.supports_sequence_packing is True


def test_doc_ids_from_cu_seqlens_drops_thd_padding_sentinels():
    cu_seqlens = torch.tensor([0, 3, 5, -1000, -1000], dtype=torch.int32)

    doc_ids = doc_ids_from_cu_seqlens(cu_seqlens, seq_len=6)

    assert doc_ids.tolist() == [[1, 1, 1, 2, 2, 0]]


def test_document_causal_mask_keeps_every_row_attendable_when_all_padding():
    """An all-padding shard must not produce a fully masked row (softmax would be NaN)."""
    doc_ids = torch.zeros(1, 6, dtype=torch.int32)

    mask = build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)

    allowed = mask[0, 0] == 0
    assert allowed.any(dim=-1).all()
    assert allowed[:, 0].all()


def test_document_causal_mask_is_finite_for_a_padding_tail():
    doc_ids = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.int32)

    mask = build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)

    assert torch.isfinite(mask).all()
    assert (mask[0, 0] == 0).any(dim=-1).all()


def test_shard_batch_gives_the_trailing_rank_an_all_padding_shard():
    """Pre-padding a dataset can leave a whole CP rank without a real token."""
    batch = _batch(seq_len=8)
    batch["attention_mask"] = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int32)
    batch["labels"] = torch.tensor([[1, 2, 3, 4, -100, -100, -100, -100]])

    _, sharded, _ = shard_batch_for_kimi_cp(_FakeCPMesh(2, 1), None, batch)

    context = sharded["kimi_packed_context"]
    assert context.local_doc_ids.tolist() == [[0, 0, 0, 0]]
    assert sharded["labels"].tolist() == [[-100, -100, -100, -100]]
    assert sharded["padding_mask"].all()
    # The document map still tiles the full sequence so FLA's CP partitioning lines up.
    assert context.row_cu_seqlens(0)[0].tolist() == [0, 4, 8]
