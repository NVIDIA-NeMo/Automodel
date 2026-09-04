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

"""CPU contracts for MiniMax M3 MSA packed-document planning."""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSAPackedLayout, _resolve_canonical_document_map


def _block_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build an independent same-document causal mask.

    Args:
        doc_ids: Integer tensor of shape [batch, sequence], with 0 for padding.

    Returns:
        Bool tensor of shape [batch, 1, sequence, sequence].
    """
    real = doc_ids > 0
    same_document = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.ones(doc_ids.shape[-1], doc_ids.shape[-1], dtype=torch.bool, device=doc_ids.device).tril()
    return (real.unsqueeze(-1) & real.unsqueeze(-2) & same_document & causal).unsqueeze(1)


def test_packed_layout_maps_adversarial_documents_and_gradients() -> None:
    """Exercise residues, padding locations, batch isolation, and autograd once."""
    doc_ids = torch.zeros(3, 262, dtype=torch.int64)
    doc_ids[0, 1:128] = 42
    doc_ids[0, 130:259] = 7
    doc_ids[1, :128] = 7
    doc_ids[1, 128] = 9
    layout = _MSAPackedLayout.build(doc_ids)
    external = torch.randn(3, 262, 3, requires_grad=True)
    upstream = torch.randn_like(external)

    packed = layout.pack(external)
    restored = layout.unpack(packed)
    metadata = layout.launch_metadata()
    assert packed.shape == (385, 3)
    assert torch.equal(packed, external[doc_ids > 0])
    assert torch.equal(restored[doc_ids > 0], external[doc_ids > 0])
    assert torch.count_nonzero(restored[doc_ids == 0]) == 0
    assert layout.has_padding is True
    assert layout.has_multiple_documents_per_row is True
    assert (metadata.total_tokens, metadata.workspace_size, metadata.max_seqlen) == (385, 640, 129)
    assert metadata.cu_seqlens.tolist() == [0, 127, 256, 384, 385]
    assert metadata.document_workspace_starts.tolist() == [0, 128, 384, 512]
    assert metadata.workspace_positions[[0, 126, 127, 255, 256, 383, 384]].tolist() == [
        0,
        126,
        128,
        256,
        384,
        511,
        512,
    ]

    restored.backward(upstream)
    assert torch.equal(external.grad[doc_ids > 0], upstream[doc_ids > 0])
    assert torch.count_nonzero(external.grad[doc_ids == 0]) == 0


@pytest.mark.parametrize(
    ("packed_seq_ids", "attention_mask", "padding_mask", "expected"),
    [
        pytest.param(
            torch.tensor([[9, 9, 4, 4, 0]], dtype=torch.int32),
            torch.ones(1, 5, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.bool),
            torch.tensor([[9, 9, 4, 4, 0]]),
            id="packed-ids-win",
        ),
        pytest.param(
            None,
            torch.tensor([[3, 3, 8, 8, 0]], dtype=torch.int32),
            None,
            torch.tensor([[3, 3, 8, 8, 0]]),
            id="indexed-mask",
        ),
        pytest.param(
            None,
            torch.tensor([[True, True, False, True, False]]),
            None,
            torch.tensor([[1, 1, 0, 1, 0]]),
            id="keep-mask",
        ),
        pytest.param(
            None,
            _block_causal_mask(torch.tensor([[1, 1, 0, 2, 2]])),
            None,
            torch.tensor([[1, 1, 0, 2, 2]]),
            id="block-causal-mask",
        ),
        pytest.param(
            None,
            None,
            torch.tensor([[False, False, True, False, True]]),
            torch.tensor([[1, 1, 0, 1, 0]]),
            id="padding-mask",
        ),
        pytest.param(None, None, None, torch.ones(1, 5, dtype=torch.int64), id="single-document"),
    ],
)
def test_document_map_source_precedence(
    packed_seq_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    expected: torch.Tensor,
) -> None:
    """Recover one canonical document map from the supported metadata sources.

    Args:
        packed_seq_ids: Optional integer tensor of shape [batch, sequence].
        attention_mask: Optional tensor of shape [batch, sequence] or [batch, 1, sequence, sequence].
        padding_mask: Optional tensor of shape [batch, sequence], true for padding.
        expected: Expected int64 tensor of shape [batch, sequence].
    """
    recovered = _resolve_canonical_document_map(
        torch.empty(1, 5, 8),
        packed_seq_ids=packed_seq_ids,
        attention_mask=attention_mask,
        padding_mask=padding_mask,
    )

    assert recovered.dtype == torch.int64
    assert recovered.is_contiguous()
    assert torch.equal(recovered, expected)


@pytest.mark.parametrize(
    ("doc_ids", "match"),
    [
        pytest.param(torch.ones(4, dtype=torch.int64), r"\[batch, sequence\]", id="rank"),
        pytest.param(torch.ones(1, 4), "integer tensor", id="dtype"),
        pytest.param(torch.tensor([[1, -1, 1]]), "non-negative", id="negative"),
        pytest.param(torch.zeros(1, 4, dtype=torch.int64), "at least one real token", id="all-padding"),
        pytest.param(torch.tensor([[1, 0, 1]]), "contiguous run", id="resumed-document"),
    ],
)
def test_packed_layout_rejects_invalid_document_maps(doc_ids: torch.Tensor, match: str) -> None:
    """Reject one representative for every canonical-map invariant.

    Args:
        doc_ids: Candidate tensor whose required shape is [batch, sequence].
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _MSAPackedLayout.build(doc_ids)


@pytest.mark.parametrize(
    ("attention_mask", "match"),
    [
        pytest.param(torch.ones(1, 4), "integer or bool 2-D", id="float-2d"),
        pytest.param(torch.ones(1, 2, 4, 4, dtype=torch.bool), "only a bool 4-D", id="heads-4d"),
        pytest.param(torch.ones(1, 4, 4, dtype=torch.bool), "must have shape", id="rank-3"),
        pytest.param(
            torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "standard bool block-causal",
            id="noncausal-4d",
        ),
    ],
)
def test_document_map_rejects_ambiguous_attention_masks(attention_mask: torch.Tensor, match: str) -> None:
    """Reject ambiguous mask tensors before document recovery.

    Args:
        attention_mask: Candidate mask whose supported layouts are [batch, sequence] and [batch, 1, sequence, sequence].
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _resolve_canonical_document_map(
            torch.empty(1, 4, 8),
            packed_seq_ids=None,
            attention_mask=attention_mask,
            padding_mask=None,
        )
