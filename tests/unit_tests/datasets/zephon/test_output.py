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

"""Batch-format conversion preserves tokens, model positions and supervision."""

import pytest
import torch

from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.attention import enable_xdoc_attention
from nemo_automodel.components.datasets.zephon.output import convert_batch_output
from nemo_automodel.components.distributed.thd_utils import process_input_for_thd, split_batch_into_thd_chunks


def _batch():
    ids = torch.tensor([[1, 1, 2, 2, 2, 0], [1, 1, 1, 1, 0, 0]])
    tokens = torch.tensor([[2, 1, 3, 4, 5, 1], [6, 7, 1, 8, 1, 1]])
    return dict(
        input_ids=tokens,
        labels=tokens.roll(-1, -1),
        position_ids=torch.tensor([[0, 1, 0, 1, 2, 0], [0, 1, 2, 3, 0, 0]]),
        document_ids=ids,
        attention_document_ids=ids,
        padding_mask=ids.eq(0),
    )


def test_thd_lengths_preserve_shifted_tokens_and_padding_even_when_pad_is_eos():
    batch = _batch()
    original_tensors = tuple(batch.items())
    assert convert_batch_output(batch, output_format="batched") is batch
    result = convert_batch_output(batch, output_format="thd")
    assert result is batch
    assert result["seq_lens"].tolist() == [[2, 3], [4, -1000]]
    assert result["seq_lens_padded"].tolist() == [[2, 4], [6, -1000]]
    assert result["qkv_format"] == "thd"
    for key, tensor in original_tensors:
        assert result[key] is tensor
    result["_packed_seq_ids"] = result["document_ids"]
    prepared = process_input_for_thd(result)
    torch.testing.assert_close(prepared["padding_mask"], batch["padding_mask"].flatten())
    assert prepared["padding_mask"][1].item() is False  # Real EOS shares the PAD id.
    assert prepared["cu_seqlens_padded"].tolist() == [0, 2, 6, 12]
    assert prepared["_packed_seq_ids"].tolist() == [1, 1, 2, 2, 2, 0, 3, 3, 3, 3, 0, 0]


def test_xdoc_thd_joins_attention_only_within_each_batch_document():
    batch = _batch()
    labels = batch["labels"].clone()
    enable_xdoc_attention(batch, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT)
    result = convert_batch_output(batch, output_format="thd")
    assert result["seq_lens"].tolist() == [[5], [4]]
    assert result["seq_lens_padded"].tolist() == [[6], [6]]
    assert result["position_ids"][0].tolist() == [0, 1, 2, 3, 4, 0]
    torch.testing.assert_close(result["labels"], labels)
    assert result["document_ids"][0].tolist() == [1, 1, 2, 2, 2, 0]


def test_thd_pp_chunks_keep_independent_lengths_and_original_ids():
    result = convert_batch_output(_batch(), output_format="thd")
    result["_packed_seq_ids"] = result["document_ids"]
    chunks = split_batch_into_thd_chunks(result, num_chunks=2)
    assert chunks["input_ids"].shape == (2, 6)
    assert chunks["cu_seqlens_padded"].tolist() == [[0, 2, 6], [0, 6, -1000]]
    assert chunks["_packed_seq_ids"].tolist() == [[1, 1, 2, 2, 2, 0], [1, 1, 1, 1, 0, 0]]


def test_thd_preserves_vlm_positions_and_media():
    batch = _batch()
    batch["position_ids"] = batch["position_ids"].unsqueeze(0).expand(3, -1, -1).clone()
    batch["pixel_values"] = torch.randn(2, 3, 4, 4)
    result = convert_batch_output(batch, output_format="thd")
    assert result["pixel_values"] is batch["pixel_values"]
    prepared = process_input_for_thd(result)
    torch.testing.assert_close(prepared["position_ids"], batch["position_ids"].reshape(3, 1, -1))
    assert prepared["pixel_values"] is batch["pixel_values"]


@pytest.mark.parametrize("case", ["exclusion", "internal_padding", "empty", "repeated_document"])
def test_thd_rejects_unrepresentable_masks(case):
    batch = _batch()
    if case == "exclusion":
        batch["attention_document_ids"] = batch["document_ids"].clone()
        batch["attention_document_ids"][0, 0] = 0
    elif case == "internal_padding":
        batch["padding_mask"][0, 1] = True
        batch["attention_document_ids"][0, 1] = 0
    elif case == "empty":
        batch["padding_mask"][0] = True
        batch["attention_document_ids"][0] = 0
    else:
        batch["attention_document_ids"][0, :5] = torch.tensor([1, 1, 2, 2, 1])
    with pytest.raises(ValueError, match="THD"):
        convert_batch_output(batch, output_format="thd")
