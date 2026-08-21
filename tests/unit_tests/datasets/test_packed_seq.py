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

import pytest
import torch

from nemo_automodel.components.datasets.packed_seq import (
    PackedSeqParams,
    packed_seq_params_from_doc_ids,
    to_flash_attention_kwargs,
)
from nemo_automodel.components.datasets.utils import _indexed_mask_to_4d_block_causal


def _block_causal_from_cu_seqlens(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Reference allowed-attention mask [total, total] from cu_seqlens.

    Args:
        cu_seqlens: Int tensor of shape [num_segments + 1], segment offsets over
            the flattened token axis.
        total: Number of flattened tokens (batch * sequence).

    Returns:
        Bool tensor of shape [total, total]; True means query i may attend key j
        (same segment and j <= i).
    """
    seg_id = torch.zeros(total, dtype=torch.long)
    for k in range(cu_seqlens.numel() - 1):
        seg_id[cu_seqlens[k] : cu_seqlens[k + 1]] = k
    same = seg_id.unsqueeze(1) == seg_id.unsqueeze(0)
    causal = torch.ones(total, total, dtype=torch.bool).tril()
    return same & causal


def test_basic_cu_seqlens_and_max_seqlen():
    doc_ids = torch.tensor([[1, 1, 2, 2, 2, 0]])
    params = packed_seq_params_from_doc_ids(doc_ids)
    # segments: doc1 (len 2), doc2 (len 3), padding (len 1)
    assert params.cu_seqlens.dtype == torch.int32
    assert params.cu_seqlens.tolist() == [0, 2, 5, 6]
    assert params.max_seqlen == 3
    assert int(params.cu_seqlens[-1]) == doc_ids.numel()
    assert torch.equal(params.doc_ids, doc_ids)


def test_cu_seqlens_spans_full_flattened_batch_with_padding():
    doc_ids = torch.tensor([[1, 1, 2, 2, 2, 0], [1, 2, 2, 3, 3, 3]])
    params = packed_seq_params_from_doc_ids(doc_ids)
    # row0: 2,3,1 (pad) ; row1: 1,2,3 -> offsets accumulate across the flattened 12 tokens
    assert params.cu_seqlens.tolist() == [0, 2, 5, 6, 7, 9, 12]
    assert int(params.cu_seqlens[-1]) == doc_ids.numel()


def test_row_boundary_prevents_cross_row_merge():
    """A row fully filled by document 1 must not merge with the next row's doc 1."""
    doc_ids = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2]])
    params = packed_seq_params_from_doc_ids(doc_ids)
    # Without a forced row boundary this would be [0, 6, 8] (rows merged).
    assert params.cu_seqlens.tolist() == [0, 4, 6, 8]


def test_parity_with_sdpa_block_causal_mask():
    """cu_seqlens segmentation must match the SDPA 4D block-causal mask exactly.

    Equal masks == identical cross-document isolation, i.e. no attention leakage
    between the flash (cu_seqlens) and sdpa (block-causal) packing paths.
    """
    doc_ids = torch.tensor([[1, 1, 2, 2, 2, 0], [1, 1, 1, 2, 0, 0]])
    batch, seq = doc_ids.shape
    params = packed_seq_params_from_doc_ids(doc_ids)

    flash_allowed = _block_causal_from_cu_seqlens(params.cu_seqlens, batch * seq)

    sdpa_4d = _indexed_mask_to_4d_block_causal(doc_ids)  # [B, 1, S, S]
    # The flash path treats the batch as one flattened stream; compare per row
    # (block-diagonal across rows is guaranteed by the forced row boundaries).
    # Compare only non-padding positions: padding tokens attend themselves in the
    # flash varlen path (own 1-length segment) but nothing under SDPA. That row
    # differs harmlessly because padding labels are ignored by the loss.
    for b in range(batch):
        valid = doc_ids[b] > 0
        block = flash_allowed[b * seq : (b + 1) * seq, b * seq : (b + 1) * seq]
        assert torch.equal(block[valid][:, valid], sdpa_4d[b, 0][valid][:, valid]), f"row {b} mask mismatch"
        # cross-row blocks must be all-disallowed
        for b2 in range(batch):
            if b2 == b:
                continue
            cross = flash_allowed[b * seq : (b + 1) * seq, b2 * seq : (b2 + 1) * seq]
            assert not cross.any(), f"cross-row leakage between {b} and {b2}"


def test_to_flash_attention_kwargs_shape_and_keys():
    doc_ids = torch.tensor([[1, 1, 2, 2, 0]])
    params = packed_seq_params_from_doc_ids(doc_ids)
    kwargs = to_flash_attention_kwargs(params)
    assert set(kwargs) == {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
    assert torch.equal(kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"])
    assert torch.equal(kwargs["cu_seq_lens_q"], params.cu_seqlens)
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == params.max_seqlen
    assert isinstance(kwargs["max_length_q"], int)


def test_single_document_no_padding():
    doc_ids = torch.tensor([[1, 1, 1, 1]])
    params = packed_seq_params_from_doc_ids(doc_ids)
    assert params.cu_seqlens.tolist() == [0, 4]
    assert params.max_seqlen == 4


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        packed_seq_params_from_doc_ids(torch.tensor([1, 1, 2, 2]))


def test_returns_typed_params():
    params = packed_seq_params_from_doc_ids(torch.tensor([[1, 1, 0]]))
    assert isinstance(params, PackedSeqParams)
