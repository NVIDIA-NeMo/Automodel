# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Backward K96/Q128 classification against independent token-pair semantics."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.fa4 import _BackwardBlockPartition, _split_backward_full_blocks
from nemo_automodel.components.models.step5.sparse import region_metadata
from tests.unit_tests.models.step5.test_forward_full_blocks import _assert_existing_fields_equal


def _lists(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent legal recency-route reverse lists [2,1,Ktiles,Qtiles]."""
    nq, nk = (length + 127) // 128, (length + 95) // 96
    counts = torch.zeros((2, 1, nk), dtype=torch.int32)
    ids = torch.full((2, 1, nk, nq), nq, dtype=torch.int32)
    for kblock in range(nk):
        chosen = []
        for qblock in range(nq):
            first_q, last_q = qblock * 128, min(length, (qblock + 1) * 128) - 1
            first_k, last_k = kblock * 96, min(length, (kblock + 1) * 96) - 1
            if first_k <= last_q and last_k >= max(0, first_q // 8 - 512) * 8:
                chosen.append(qblock)
        counts[:, :, kblock] = len(chosen)
        ids[:, :, kblock, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return counts, ids


def _oracle(
    counts: torch.Tensor,
    indices: torch.Tensor,
    split: _BackwardBlockPartition,
    *,
    length: int,
    document_reset: int | None = None,
) -> None:
    """Check partition union/order and all token pairs of every full rectangle.

    Args:
        counts: Original int32 [providers, 1, key_tiles] counts.
        indices: Original int32 [providers, 1, key_tiles, query_tiles] IDs.
        split: Partial/full counts and IDs in the same respective layouts.
        length: Unpadded document token count.
        document_reset: Independent hypothetical packed boundary for negative control.
    """
    for provider in range(counts.shape[0]):
        for kblock in range(counts.shape[-1]):
            old = indices[provider, 0, kblock, : int(counts[provider, 0, kblock])].tolist()
            partial = split.partial_indices[
                provider, 0, kblock, : int(split.partial_counts[provider, 0, kblock])
            ].tolist()
            full = split.full_indices[provider, 0, kblock, : int(split.full_counts[provider, 0, kblock])].tolist()
            assert old == sorted(set(old)) and all(v >= 0 for v in old)
            assert sorted(partial + full) == old
            assert partial == [v for v in old if v not in full]
            assert full == sorted(set(full))
            k = torch.arange(kblock * 96, min((kblock + 1) * 96, length))[None]
            for qblock in full:
                q = torch.arange(qblock * 128, min((qblock + 1) * 128, length))[:, None]
                allowed = (k <= q) & (k // 8 >= (q // 8 - 512).clamp_min(0))
                if document_reset is not None:
                    allowed &= (q >= document_reset) == (k >= document_reset)
                assert allowed.all(), (provider, kblock, qblock)


@pytest.mark.parametrize("length", [256, 4096, 4104, 4224, 8192])
def test_reverse_partition_has_exact_union_and_valid_full_pairs(length: int) -> None:
    counts, indices = _lists(length)
    old_counts, old_ids = counts.clone(), indices.clone()
    result = _split_backward_full_blocks.__wrapped__(counts, indices, prefix_length=min(length, 4104))
    assert result.full_counts.sum() > 0
    _oracle(counts, indices, result, length=length)
    assert torch.equal(counts, old_counts) and torch.equal(indices, old_ids)
    if length == 8192:
        assert int(result.full_counts[0, 0].sum()) == 651
        assert torch.all(result.full_indices[result.full_counts[..., None] > torch.arange(indices.shape[-1])] < 32)


@pytest.mark.parametrize("mutation", ["boundary", "document", "partial_order", "full_order"])
def test_wrong_boundary_document_or_order_is_rejected(mutation: str) -> None:
    counts, ids = _lists(8192)
    result = _split_backward_full_blocks.__wrapped__(
        counts, ids, prefix_length=4224 if mutation == "boundary" else 4104
    )
    if mutation == "partial_order":
        result.partial_indices[0, 0, 0, :2] = result.partial_indices[0, 0, 0, :2].flip(0)
    if mutation == "full_order":
        result.full_indices[0, 0, 0, :2] = result.full_indices[0, 0, 0, :2].flip(0)
    with pytest.raises(AssertionError):
        _oracle(counts, ids, result, length=8192, document_reset=128 if mutation == "document" else None)


def test_holes_empty_and_short_rows_preserve_all_entries() -> None:
    counts, ids = _lists(8192)
    counts[:, :, 0] = 0
    counts[:, :, 1] = 1
    row = ids[0, 0, 2, : int(counts[0, 0, 2])].tolist()
    row.remove(5)
    counts[0, 0, 2] = len(row)
    ids[0, 0, 2, : len(row)] = torch.tensor(row, dtype=torch.int32)
    result = _split_backward_full_blocks.__wrapped__(counts, ids, prefix_length=4104)
    _oracle(counts, ids, result, length=8192)
    assert result.full_counts[0, 0, 0] == 0


def test_full_graph_has_no_host_tensor_dependency() -> None:
    counts, ids = _lists(512)
    expected = _split_backward_full_blocks.__wrapped__(counts, ids, prefix_length=512)
    compiled = torch.compile(_split_backward_full_blocks.__wrapped__, backend="eager", fullgraph=True)
    actual = compiled(counts, ids, prefix_length=512)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "case", ["single", "packed", "batch2", "cp_first", "cp_second", "pad", "internal_pad", "repeat", "missing"]
)
def test_backward_proof_is_independent_and_conservative(case: str) -> None:
    positions = torch.arange(512)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    start, queries = 0, 512
    if case == "packed":
        positions[:, 256:] -= 256
    elif case == "batch2":
        positions, padding = positions.expand(2, -1), padding.expand(2, -1)
    elif case == "cp_first":
        queries = 256
    elif case == "cp_second":
        start, queries = 256, 256
    elif case in ("pad", "internal_pad"):
        pos = 500 if case == "pad" else 111
        padding[:, pos:] = case == "pad"
        padding[:, pos] = True
        positions[padding] = 0
    elif case == "repeat":
        positions[:, 111] = 110
    elif case == "missing":
        positions[:, 111:] += 1
    old = region_metadata(positions, padding, query_start=start, queries=queries, region_size=8, topk=32)
    new = region_metadata(
        positions, padding, query_start=start, queries=queries, region_size=8, topk=32, csa_backward_full_blocks=True
    )
    assert new.backward_full_block_prefix == (264 if case == "single" else 0)
    assert new.forward_full_block_prefix == 0
    _assert_existing_fields_equal(old, replace(new, backward_full_block_prefix=0))


@pytest.mark.parametrize("value", [1, "true", None])
def test_flag_is_strict_bool(value: int | str | None) -> None:
    with pytest.raises(TypeError, match="csa_backward_full_blocks"):
        Step5BackendConfig(attn="fa4", csa_backward_full_blocks=value)


def test_wrong_backend_is_rejected_and_default_stays_off() -> None:
    assert not Step5BackendConfig(attn="fa4").csa_backward_full_blocks
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="sdpa", csa_backward_full_blocks=True)
