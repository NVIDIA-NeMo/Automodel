# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


"""Image expansion before native document packing."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.kimi_k3.processing import tokenize_conversation


class _Processor:
    def __init__(self, ids: list[int], grids: list[list[int]] | None):
        self.ids = ids
        self.grids = grids
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 99)

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
        return "rendered"

    def __call__(self, *, text, medias, return_tensors):
        assert isinstance(medias, list)
        result = {
            "input_ids": torch.tensor([self.ids]),
            "attention_mask": torch.ones(1, len(self.ids), dtype=torch.long),
        }
        if self.grids is not None:
            result["grid_thws"] = torch.tensor(self.grids)
            result["pixel_values"] = torch.zeros(sum(t * h * w for t, h, w in self.grids), 3, 14, 14)
        return result


def test_expands_each_image_without_shifting_tokens():
    processor = _Processor([1, 99, 2, 99, 3], [[1, 4, 4], [1, 2, 4]])
    result = tokenize_conversation([], processor)
    assert result["input_ids"].tolist() == [[1, 99, 99, 99, 99, 2, 99, 99, 3]]
    assert result["attention_mask"].tolist() == [[1] * 9]
    assert result["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert processor.ids == [1, 99, 2, 99, 3]
    assert "labels" not in result


def test_text_only_supplies_empty_medias():
    result = tokenize_conversation([], _Processor([1, 2, 3], None))
    assert result["input_ids"].tolist() == [[1, 2, 3]]
    assert "image_grid_thw" not in result


@pytest.mark.parametrize(
    "ids,grid",
    [
        ([1, 99, 2], [[1, 4, 4], [1, 2, 4]]),
        ([1, 99, 2], [[2, 4, 4]]),
        ([1, 99, 2], [[1, 1, 1]]),
    ],
)
def test_rejects_invalid_image_layout(ids, grid):
    with pytest.raises(ValueError):
        tokenize_conversation([], _Processor(ids, grid))
