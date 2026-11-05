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
from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing,
    get_packing_capabilities,
)


@pytest.mark.parametrize("collater", [neat_packed_collater, neat_packed_vlm_collater])
def test_flash_offsets_are_built_for_the_batch_reaching_model_entry(collater):
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
    batch = collater(documents, packing=contract)
    assert "cu_seq_lens_q" not in batch
    assert "cu_seq_lens_k" not in batch
    assert "attention_mask" not in batch
    assert all(value.shape[0] == 2 for value in batch.values() if isinstance(value, torch.Tensor))
    torch.testing.assert_close(batch["_packed_seq_ids"], torch.tensor([[1, 1, 2, 2, 0], [1, 2, 2, 2, 2]]))

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            """Return metadata for tokens [batch, sequence] without modifying it."""
            return kwargs

    decoder = Decoder()
    configure_packing("flash_attention_2", model=decoder)
    # The whole non-PP batch and each PP microbatch use the same entry path.
    for selection, offsets, maximum in (
        (slice(None), [0, 2, 4, 5, 6, 10], 4),
        (slice(0, 1), [0, 2, 4, 5], 2),
        (slice(1, 2), [0, 1, 5], 4),
    ):
        actual = decoder(
            batch["input_ids"][selection],
            position_ids=batch["position_ids"][selection],
            _packed_seq_ids=batch["_packed_seq_ids"][selection],
        )
        assert actual["cu_seq_lens_q"].tolist() == offsets
        torch.testing.assert_close(actual["cu_seq_lens_k"], actual["cu_seq_lens_q"])
        assert actual["max_length_q"] == actual["max_length_k"] == maximum
        torch.testing.assert_close(actual["position_ids"], batch["position_ids"][selection])


def test_flash_and_dense_collation_preserve_the_same_training_tensors():
    document = {"input_ids": [1, 2], "labels": [2, 3], "position_ids": [0, 1], "attention_mask": [1, 1]}
    ordinary = neat_packed_collater([document], packing=get_packing_capabilities("sdpa"))
    flash = neat_packed_collater([document], packing=get_packing_capabilities("flash_attention_2"))
    for key in ("input_ids", "labels", "position_ids", "_packed_seq_ids"):
        torch.testing.assert_close(ordinary[key], flash[key])


def test_text_collater_keeps_single_document_boundaries():
    batch = [
        {
            "input_ids": torch.tensor([2, 3, 0]),
            "labels": torch.tensor([3, -100, -100]),
            "position_ids": torch.tensor([0, 1, 0]),
            "attention_mask": torch.tensor([1, 1, 0]),
        }
    ]
    result = neat_packed_collater(batch, packing=get_packing_capabilities("sdpa"))
    assert result["_packed_seq_ids"].tolist() == [[1, 1, 0]]
