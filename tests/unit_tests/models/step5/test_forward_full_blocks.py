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

"""Forward-only full-block proofs, token-pair oracle and traversal invariants."""

from dataclasses import fields

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.fa4 import _split_forward_full_blocks
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.models.step5.sparse import RegionMetadata, Step5SparseIndexer, region_metadata
from tests.unit_tests.models.step5.test_model import tiny_config


def _metadata(
    positions: torch.Tensor,
    padding: torch.Tensor,
    *,
    enabled: bool,
    start: int = 0,
    queries: int | None = None,
    topk: int = 512,
) -> RegionMetadata:
    """Prepare the shared CPU region layout.

    Args:
        positions: Integer positions [batch, global_keys].
        padding: Boolean padding flags with the same layout.
        enabled: Whether to request the optional forward proof.
        start: Physical global offset of the local query shard.
        queries: Local query count, or the complete key length.
        topk: Selected historical region budget.

    Returns:
        RegionMetadata with the tensor layouts documented on that dataclass.
    """
    return region_metadata(
        positions,
        padding,
        query_start=start,
        queries=positions.shape[1] if queries is None else queries,
        region_size=8,
        topk=topk,
        csa_forward_full_blocks=enabled,
    )


def _assert_existing_fields_equal(first: RegionMetadata, second: RegionMetadata) -> None:
    """Compare old fields of two shared region layouts.

    Args:
        first: RegionMetadata with global [batch, keys] token/document tensors,
            local [batch, queries] positions, [batch, queries, slots] route
            templates and optional compact region/document layouts.
        second: Metadata with the same per-field layouts and dtypes.
    """

    def equal(a: object, b: object) -> None:
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype and a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, tuple):
            assert isinstance(b, tuple) and len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        elif a is None or isinstance(a, (bool, int)):
            assert a == b
        else:
            assert type(a) is type(b)
            for field in fields(a):
                equal(a.__dict__[field.name], b.__dict__[field.name])

    for field in fields(first):
        if field.name not in ("forward_full_block_prefix", "canonical_region_size"):
            equal(first.__dict__[field.name], second.__dict__[field.name])


def _route_block_lists(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CPU int32 lists [2,1,Qtiles] / [2,1,Qtiles,Ktiles].

    A valid synthetic router chooses the most recent 512 completed regions,
    plus the current region. Each region contains eight tokens. This is an
    independent legal routing fixture, not a model score implementation.
    """
    rows, columns = (length + 127) // 128, (length + 111) // 112
    counts = torch.zeros(2, 1, rows, dtype=torch.int32)
    indices = torch.full((2, 1, rows, columns), columns, dtype=torch.int32)
    for row in range(rows):
        start, stop = row * 128, min((row + 1) * 128, length)
        first_allowed_key = max(0, start // 8 - 512) * 8
        blocks = list(range(first_allowed_key // 112, (stop + 111) // 112))
        counts[:, :, row] = len(blocks)
        indices[:, :, row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return counts, indices


def _assert_pair_oracle(
    counts: torch.Tensor,
    indices: torch.Tensor,
    partial_counts: torch.Tensor,
    partial_indices: torch.Tensor,
    full_counts: torch.Tensor,
    *,
    length: int,
    document_reset: int | None = None,
) -> None:
    """Check block partitioning against explicit token-pair semantics.

    Args:
        counts: Original int32 counts [providers, 1, query_tiles].
        indices: Sorted original indices [providers, 1, query_tiles, key_tiles].
        partial_counts: Remaining partial counts with the counts layout.
        partial_indices: Remaining partial indices with the indices layout.
        full_counts: Full counts with the counts layout; indices aliases its list.
        length: Actual token count, excluding tile padding.
        document_reset: Optional physical boundary making cross-document pairs
            invalid independently of the candidate metadata.
    """
    for provider in range(counts.shape[0]):
        for row in range(counts.shape[-1]):
            old = indices[provider, 0, row, : int(counts[provider, 0, row])].tolist()
            partial = partial_indices[provider, 0, row, : int(partial_counts[provider, 0, row])].tolist()
            full = indices[provider, 0, row, : int(full_counts[provider, 0, row])].tolist()
            assert old == sorted(set(old)) and all(block >= 0 for block in old)
            assert len(set(partial + full)) == len(old)
            assert set(partial + full) == set(old)
            assert list(reversed(partial)) + list(reversed(full)) == list(reversed(old))
            q = torch.arange(row * 128, min((row + 1) * 128, length))[:, None]
            for block in full:
                k = torch.arange(block * 112, min((block + 1) * 112, length))[None]
                # Explicit independent causal / selected-region / document mask.
                allowed = (k <= q) & (k // 8 >= (q // 8 - 512).clamp_min(0))
                if document_reset is not None:
                    allowed &= (q >= document_reset) == (k >= document_reset)
                assert allowed.all(), (row, block)


@pytest.mark.parametrize("length", [4096, 4104, 4224, 8192])
def test_single_document_proof_and_prior_metadata_unchanged(length: int) -> None:
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    old = _metadata(positions, padding, enabled=False)
    candidate = _metadata(positions, padding, enabled=True)
    assert old.forward_full_block_prefix == 0
    assert candidate.forward_full_block_prefix == min(length, 4104)
    _assert_existing_fields_equal(old, candidate)
    if length <= 4104:
        assert candidate.causal_layout is not None  # Existing native path stays first.


@pytest.mark.parametrize(
    "case",
    [
        "packed",
        "batch2",
        "cp_first",
        "cp_second",
        "trailing_pad",
        "internal_pad",
        "repeated",
        "missing",
        "nonzero_start",
        "negative",
    ],
)
def test_unproven_layout_preserves_old_metadata(case: str) -> None:
    positions = torch.arange(512)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    start, queries = 0, 512
    if case == "packed":
        positions[:, 256:] -= 256
    elif case == "batch2":
        positions = positions.expand(2, -1)
        padding = padding.expand(2, -1)
    elif case == "cp_first":
        queries = 256
    elif case == "cp_second":
        start, queries = 256, 256
    elif case == "trailing_pad":
        padding[:, 479:] = True
        positions[:, 479:] = 0
    elif case == "internal_pad":
        padding[:, 111] = True
        positions[:, 111] = 0
    elif case == "repeated":
        positions[:, 111] = 110
    elif case == "missing":
        positions[:, 111:] += 1
    elif case == "nonzero_start":
        positions += 8
    elif case == "negative":
        positions[:, 0] = -1
    old = _metadata(positions, padding, enabled=False, start=start, queries=queries, topk=2)
    candidate = _metadata(positions, padding, enabled=True, start=start, queries=queries, topk=2)
    assert candidate.forward_full_block_prefix == 0
    _assert_existing_fields_equal(old, candidate)


@pytest.mark.parametrize("length", [4096, 8192])
def test_full_blocks_match_independent_pairs_union_and_order(length: int) -> None:
    counts, indices = _route_block_lists(length)
    saved_counts, saved_indices = counts.clone(), indices.clone()
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=min(length, 4104)
    )
    assert full_counts.sum() > 0
    assert int(full_counts[0, 0, 31]) == 35
    if length > 4104:
        assert torch.equal(full_counts[:, :, 32:], torch.zeros_like(full_counts[:, :, 32:]))
    _assert_pair_oracle(counts, indices, partial_counts, partial_indices, full_counts, length=length)
    assert torch.equal(counts, saved_counts) and torch.equal(indices, saved_indices)


@pytest.mark.parametrize("missing", [0, 2])
def test_discontinuous_low_block_prefix_is_not_marked_full(missing: int) -> None:
    counts, indices = _route_block_lists(4096)
    # Sorted unique nonnegative indices can still have a hole in the low prefix.
    indices[:, :, 4, missing:-1] = indices[:, :, 4, missing + 1 :].clone()
    counts[:, :, 4] -= 1
    _, _, full_counts = _split_forward_full_blocks.__wrapped__(counts, indices, prefix_length=4096)
    assert torch.equal(full_counts[:, :, 4], torch.zeros_like(full_counts[:, :, 4]))


@pytest.mark.parametrize("mutation", ["boundary", "document", "order"])
def test_independent_oracle_detects_deliberately_wrong_full_blocks(mutation: str) -> None:
    counts, indices = _route_block_lists(8192)
    prefix = 4224 if mutation == "boundary" else 4104
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=prefix
    )
    if mutation == "order":
        partial_indices = partial_indices.clone()
        n = int(partial_counts[0, 0, 40])
        partial_indices[0, 0, 40, :n] = partial_indices[0, 0, 40, :n].flip(0)
    with pytest.raises(AssertionError):
        _assert_pair_oracle(
            counts,
            indices,
            partial_counts,
            partial_indices,
            full_counts,
            length=8192,
            document_reset=128 if mutation == "document" else None,
        )


def test_integer_split_has_one_full_graph() -> None:
    counts, indices = _route_block_lists(8192)
    expected = _split_forward_full_blocks.__wrapped__(counts, indices, prefix_length=4104)
    compiled = torch.compile(_split_forward_full_blocks.__wrapped__, backend="eager", fullgraph=True)
    actual = compiled(counts, indices, prefix_length=4104)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("topk,length", [(512, 16), (2, 32)])
def test_indexer_carries_shared_proof_without_changing_routes(topk: int, length: int) -> None:
    torch.manual_seed(533)
    config = tiny_config(
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=topk,
            region_block_size=8,
        )
    )
    indexer = Step5SparseIndexer(config, 3, dtype=torch.float32)
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, length, config.hidden_size)
    with torch.no_grad():
        keys, logits = indexer.project_keys(hidden, positions)
        selections = []
        for enabled in (False, True):
            meta = _metadata(positions, padding, enabled=enabled, topk=topk)
            selections.append(indexer(hidden, positions, keys, logits, positions, padding, metadata=meta))
    assert selections[0].forward_full_block_prefix == 0
    assert selections[1].forward_full_block_prefix == min(length, (topk + 1) * 8)
    assert torch.equal(selections[0].region_ids, selections[1].region_ids)


@pytest.mark.parametrize("value", [1, "true", None])
def test_backend_requires_strict_bool(value: int | str | None) -> None:
    with pytest.raises(TypeError, match="csa_forward_full_blocks"):
        Step5BackendConfig(attn="fa4", csa_forward_full_blocks=value)


def test_backend_requires_fa4_and_preserves_default_model_behavior() -> None:
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="sdpa", csa_forward_full_blocks=True)
    for backend in (
        BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm"),
        Step5BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm"),
        Step5BackendConfig(
            attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm", csa_forward_full_blocks=True
        ),
    ):
        model = Step5ForCausalLM(tiny_config(), backend=backend)
        expected = isinstance(backend, Step5BackendConfig) and backend.csa_forward_full_blocks
        assert model.model.csa_forward_full_blocks == expected


def test_empty_and_short_counts_do_not_read_outside_allocated_indices() -> None:
    counts, indices = _route_block_lists(4096)
    counts[:, :, 8] = 0
    indices[:, :, 8] = indices.shape[-1]
    counts[:, :, 16] = 1
    indices[:, :, 16, 1:] = indices.shape[-1]
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=4096
    )
    assert torch.equal(full_counts[:, :, (8, 16)], torch.zeros_like(full_counts[:, :, (8, 16)]))
    assert torch.equal(partial_counts[:, :, (8, 16)], counts[:, :, (8, 16)])
    _assert_pair_oracle(counts, indices, partial_counts, partial_indices, full_counts, length=4096)
