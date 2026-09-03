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

import pytest
import torch

from nemo_automodel.components.datasets.packing import (
    DEFAULT_PACKED_SEQUENCE_CONTRACT,
    build_packed_sequence_metadata,
    get_seqlens_in_batch,
    get_unpad_data,
)


def test_default_packed_sequence_contract_preserves_block_causal_behavior() -> None:
    assert DEFAULT_PACKED_SEQUENCE_CONTRACT.packed_mask_type == "block_causal"
    assert DEFAULT_PACKED_SEQUENCE_CONTRACT.requires_packed_sequence_metadata is False


def test_get_seqlens_in_batch_preserves_batch_document_order() -> None:
    document_ids = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )

    assert get_seqlens_in_batch(document_ids).tolist() == [2, 3, 1, 2, 3]


def test_get_unpad_data_builds_indexed_document_metadata() -> None:
    document_ids = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )

    indices, cu_seqlens, max_seqlen = get_unpad_data(document_ids)

    assert indices.tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert cu_seqlens.tolist() == [0, 2, 5, 6, 8, 11]
    assert cu_seqlens.dtype == torch.int32
    assert max_seqlen == 3


def test_get_unpad_data_treats_each_binary_mask_row_as_one_sequence() -> None:
    attention_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)

    assert indices.tolist() == [0, 1, 3, 4, 5]
    assert cu_seqlens.tolist() == [0, 2, 5]
    assert max_seqlen == 3


def test_build_packed_sequence_metadata_uses_backend_neutral_field_names() -> None:
    document_ids = torch.tensor([[1, 1, 2, 2, 0]])

    metadata = build_packed_sequence_metadata(document_ids)

    assert set(metadata) == {"packed_token_indices", "cu_seqlens", "max_seqlen"}
    assert metadata["packed_token_indices"].tolist() == [[0, 1, 2, 3, -1]]
    assert metadata["cu_seqlens"].tolist() == [[0, 2, 4]]
    assert metadata["max_seqlen"] == 2


def test_build_packed_sequence_metadata_is_batch_major_for_pipeline_splitting() -> None:
    document_ids = torch.tensor(
        [
            [1, 1, 2, 2, 0],
            [1, 2, 2, 0, 0],
        ]
    )

    metadata = build_packed_sequence_metadata(document_ids)

    assert metadata["packed_token_indices"].tolist() == [
        [0, 1, 2, 3, -1],
        [0, 1, 2, -1, -1],
    ]
    assert metadata["cu_seqlens"].tolist() == [[0, 2, 4], [0, 1, 3]]


@pytest.mark.parametrize(
    "attention_mask, message",
    [
        (torch.ones(2, 3, 1), "shape"),
        (torch.zeros(1, 3, dtype=torch.long), "at least one valid token"),
    ],
)
def test_get_unpad_data_rejects_invalid_masks(attention_mask: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        get_unpad_data(attention_mask)
