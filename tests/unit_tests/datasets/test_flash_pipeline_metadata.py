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

"""Native NEAT collation preserves the document batch axis before PP splitting."""

import pytest
import torch

from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common.packing import PackingCapabilities, configure_packing


@pytest.mark.parametrize("collater", [neat_packed_collater, neat_packed_vlm_collater])
def test_deferred_native_offsets_match_direct_microbatch_collation(collater):
    documents = [
        {
            "input_ids": [2, 3, 4, 5, 0],
            "labels": [3, 4, 5, 6, -100],
            "position_ids": [0, 1, 0, 1, 0],
            "attention_mask": [1, 1, 2, 2, 0],
        },
        {
            "input_ids": [7, 8, 9, 10, 11],
            "labels": [8, 9, 10, 11, 12],
            "position_ids": [0, 0, 1, 2, 3],
            "attention_mask": [1, 2, 2, 2, 2],
        },
    ]
    contract = PackingCapabilities("flash_varlen")
    direct = collater(documents, packing=contract)
    assert direct["cu_seq_lens_q"].tolist() == [0, 2, 4, 5, 6, 10]
    assert "_packed_seq_ids" in direct
    deferred = collater(documents, packing=contract, defer_flash_metadata=True)
    assert "cu_seq_lens_q" not in deferred
    assert "attention_mask" not in deferred
    torch.testing.assert_close(deferred["_packed_seq_ids"], torch.tensor([[1, 1, 2, 2, 0], [1, 2, 2, 2, 2]]))

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            """Return metadata for tokens [batch, sequence] without modifying it."""
            return kwargs

    decoder = Decoder()
    configure_packing("flash_attention_2", model=decoder)
    for index in range(2):
        expected = collater(documents[index : index + 1], packing=contract)
        actual = decoder(
            deferred["input_ids"][index : index + 1],
            _packed_seq_ids=deferred["_packed_seq_ids"][index : index + 1],
        )
        for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            torch.testing.assert_close(actual[key], expected[key])
    for key in ("input_ids", "labels", "position_ids", "_packed_seq_ids"):
        torch.testing.assert_close(direct[key], deferred[key])


def test_deferral_does_not_change_dense_native_mask():
    document = {"input_ids": [1, 2], "labels": [2, 3], "position_ids": [0, 1], "attention_mask": [1, 1]}
    ordinary = neat_packed_collater([document])
    deferred = neat_packed_collater([document], defer_flash_metadata=True)
    assert ordinary.keys() == deferred.keys()
    for key in ordinary:
        torch.testing.assert_close(ordinary[key], deferred[key])


def test_text_collater_keeps_single_document_boundaries():
    batch = [
        {
            "input_ids": torch.tensor([2, 3, 0]),
            "labels": torch.tensor([3, -100, -100]),
            "position_ids": torch.tensor([0, 1, 0]),
            "attention_mask": torch.tensor([1, 1, 0]),
        }
    ]
    result = neat_packed_collater(batch)
    assert result["_packed_seq_ids"].tolist() == [[1, 1, 0]]
