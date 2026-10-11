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

"""CPU coverage of indexed-attention eligibility and physical region bounds."""

import pytest
import torch

from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.sparse import region_metadata


def test_indexed_backend_is_opt_in_and_fa4_only() -> None:
    assert not Step5BackendConfig().indexed_csa
    assert Step5BackendConfig(attn="fa4", indexed_csa=True).indexed_csa
    with pytest.raises(TypeError, match="indexed_csa"):
        Step5BackendConfig(attn="fa4", indexed_csa=1)
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="flex", indexed_csa=True)


@pytest.mark.parametrize("length,enabled", [(8192, False), (16384, False), (32768, False), (40960, True)])
def test_indexed_layout_requires_long_effective_history(length: int, enabled: bool) -> None:
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions,
        padding,
        query_start=length - 4096,
        queries=4096,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert (metadata.indexed_layout is not None) == enabled
    assert (
        region_metadata(
            positions, padding, query_start=length - 4096, queries=4096, region_size=8, topk=512
        ).indexed_layout
        is None
    )
    if enabled:
        layout = metadata.indexed_layout
        assert layout.starts.dtype == layout.ends.dtype == torch.int32
        torch.testing.assert_close(layout.starts[0], torch.arange(0, length, 8, dtype=torch.int32))
        torch.testing.assert_close(layout.ends[0], torch.arange(8, length + 1, 8, dtype=torch.int32))


def test_physical_bounds_preserve_short_document_before_long_padding() -> None:
    # Last valid document has three tokens; all 15 padding positions are zero.
    # Reusing compression slots would overwrite its first valid key with padding.
    positions = torch.cat((torch.arange(98304), torch.arange(3), torch.zeros(15, dtype=torch.long)))[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[:, -15:] = True
    metadata = region_metadata(
        positions,
        padding,
        query_start=65536,
        queries=positions.shape[1] - 65536,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert metadata.indexed_layout is not None
    layout = metadata.indexed_layout
    assert layout.starts[0, -1].item() == 98304
    assert layout.ends[0, -1].item() == 98322
    torch.testing.assert_close(layout.starts[0, :-1], torch.arange(0, 98304, 8, dtype=torch.int32))
    torch.testing.assert_close(layout.ends[0, :-1], torch.arange(8, 98305, 8, dtype=torch.int32))


@pytest.mark.parametrize("defect", ["internal_padding", "repeated_position", "negative_position"])
def test_irregular_layout_keeps_existing_attention(defect: str) -> None:
    positions = torch.arange(98304)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    if defect == "internal_padding":
        padding[:, 100] = True
    elif defect == "repeated_position":
        positions[:, 100] = 99
    else:
        positions[:, 100] = -1
    metadata = region_metadata(
        positions, padding, query_start=65536, queries=4096, region_size=8, topk=512, indexed_csa=True
    )
    assert metadata.indexed_layout is None


def test_long_pack_of_short_documents_does_not_enable_indexed_attention() -> None:
    positions = torch.arange(1024).repeat(128)[None]
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=98304,
        queries=4096,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert not metadata.score_tiles
    assert metadata.indexed_layout is None


def test_indexed_layout_supports_batch_and_empty_query_fallback() -> None:
    positions = torch.arange(98304)[None].expand(2, -1).contiguous()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions, padding, query_start=65536, queries=4096, region_size=8, topk=512, indexed_csa=True
    )
    assert metadata.indexed_layout is not None
    torch.testing.assert_close(metadata.indexed_layout.starts[0], metadata.indexed_layout.starts[1])
    empty = region_metadata(positions, padding, query_start=0, queries=0, region_size=8, topk=512, indexed_csa=True)
    assert empty.indexed_layout is None


@pytest.mark.parametrize("keys,start,indexed", [(40960, 36864, True), (8192, 4096, False)])
@pytest.mark.parametrize("long_row", [0, 1])
@pytest.mark.parametrize("partner", ["prefix", "allpad", "partialpad"])
def test_indexed_b2_preserves_row_proofs(keys: int, start: int, indexed: bool, long_row: int, partner: str) -> None:
    queries = 4096
    positions = torch.arange(keys)[None].expand(2, -1).clone()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    short = 1 - long_row
    positions[short] = torch.arange(keys) % 1024
    if partner != "prefix":
        valid = 2048 if partner == "allpad" else start + 1317
        padding[short, valid:] = True
        positions[short, valid:] = 0
    metadata = region_metadata(
        positions,
        padding,
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert (metadata.indexed_layout is not None) == indexed
    if indexed:
        assert metadata.indexed_layout.row_eligible == tuple(row == long_row for row in range(2))
    assert len(metadata.causal_row_layouts) == 2
    assert metadata.causal_row_layouts[long_row] is None
    proof = metadata.causal_row_layouts[short]
    assert proof is not None
    assert proof.valid_queries == {"prefix": 4096, "allpad": 0, "partialpad": 1317}[partner]
    assert metadata.causal_batch_layouts is None
    if indexed:
        assert metadata.region_count == metadata.indexed_layout.starts.shape[1]
    disabled = region_metadata(
        positions,
        padding,
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
    )
    assert disabled.indexed_layout is None
    assert disabled.causal_row_layouts == ()

    # Keeping an already-computed proof cannot change score or selection input.
    assert metadata.score_tiles == disabled.score_tiles
    assert metadata.region_count == disabled.region_count
    for actual, original in (
        (metadata.token_regions, disabled.token_regions),
        (metadata.documents, disabled.documents),
        (metadata.dense_region_ids, disabled.dense_region_ids),
        (metadata.sparse_queries, disabled.sparse_queries),
    ):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


@pytest.mark.parametrize("long_row", [0, 1])
@pytest.mark.parametrize("partner", ["prefix", "partialpad", "allpad"])
def test_below_threshold_row_proof_matches_independent_document_mask(long_row: int, partner: str) -> None:
    keys, start, queries = 32, 16, 16
    positions = torch.arange(keys)[None].expand(2, -1).clone()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    short = 1 - long_row
    positions[short] %= 4
    valid = {"prefix": keys, "partialpad": start + 7, "allpad": 7}[partner]
    positions[short, valid:] = 0
    padding[short, valid:] = True
    metadata = region_metadata(
        positions, padding, query_start=start, queries=queries, region_size=2, topk=2, indexed_csa=True
    )
    assert metadata.indexed_layout is None
    assert len(metadata.causal_row_layouts) == 2
    assert metadata.causal_row_layouts[long_row] is None
    proof = metadata.causal_row_layouts[short]
    assert proof is not None
    # Scan the input document starts; do not derive the expected mask from
    # production region IDs, sparse flags, or cumulative sequence lengths.
    document = -1
    documents = []
    for position, padded in zip(positions[short].tolist(), padding[short].tolist(), strict=True):
        if position == 0 and not padded:
            document += 1
        documents.append(document)
    expected = torch.tensor(
        [
            [
                not padding[short, k]
                and documents[start + q] == documents[k]
                and positions[short, k] <= positions[short, start + q]
                for k in range(keys)
            ]
            for q in range(queries)
        ]
    )
    actual = torch.zeros_like(expected)
    qb, kb = proof.cu_seqlens_q.tolist(), proof.cu_seqlens_k.tolist()
    for q0, q1, k0, k1 in zip(qb, qb[1:], kb, kb[1:]):
        for q in range(q0, q1):
            for k in range(k0, k1):
                actual[q, proof.key_start + k] = k - k0 <= q - q0 + (k1 - k0) - (q1 - q0)
    if proof.padding_key is not None:
        actual[proof.valid_queries :, proof.padding_key] = True
    torch.testing.assert_close(actual, expected)
    assert metadata.sparse_queries[long_row].all()
    assert not metadata.sparse_queries[short].any()
    # A wrong document origin must be detected by the independent mask.
    assert not torch.equal(actual.roll(1, dims=1), expected)


@pytest.mark.parametrize("defect", ["empty", "internal_padding", "nonzero_padding", "sparse_final_document"])
def test_mixed_row_proof_rejects_unproven_padding(defect: str) -> None:
    positions = torch.arange(32)[None].expand(2, -1).clone()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    if defect == "empty":
        padding[1] = True
        positions[1] = 0
    elif defect == "internal_padding":
        positions[1] %= 8
        padding[1, 17] = True
    elif defect == "nonzero_padding":
        positions[1] %= 4
        padding[1, 16:] = True
        positions[1, 16:] = 1
    else:
        padding[1, 9:] = True
        positions[1, 9:] = 0
    metadata = region_metadata(positions, padding, query_start=16, queries=16, region_size=2, topk=2, indexed_csa=True)
    assert metadata.indexed_layout is None
    assert metadata.causal_row_layouts == (None, None)
