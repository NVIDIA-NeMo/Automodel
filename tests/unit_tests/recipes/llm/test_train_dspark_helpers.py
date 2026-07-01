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
import torch

from nemo_automodel.recipes.llm.train_dspark import _extract_mm_kwargs


def test_extract_mm_kwargs_empty_for_text_only_batch():
    batch = {"input_ids": torch.zeros(1), "attention_mask": torch.ones(1), "loss_mask": torch.ones(1)}
    assert _extract_mm_kwargs(batch) == {}


def test_extract_mm_kwargs_passes_through_present_media_keys():
    pixel_values = torch.randn(2, 3, 4, 4)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    batch = {
        "input_ids": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    mm_kwargs = _extract_mm_kwargs(batch)
    assert mm_kwargs == {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def test_extract_mm_kwargs_ignores_unrelated_keys():
    batch = {"input_ids": torch.zeros(1), "seq_lens": torch.tensor([1, 2]), "doc_remaining": torch.tensor([0])}
    assert _extract_mm_kwargs(batch) == {}
