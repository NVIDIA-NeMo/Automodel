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

"""Staged media ownership is independent of token scans and modality presence."""

import pytest
import torch

from nemo_automodel.shared.vlm_media import next_vlm_media_chunk


def test_empty_then_image_consumes_each_chunk_exactly_once():
    stage = torch.nn.Module()
    image = torch.ones(1, 3, 4, 4)
    grid = torch.tensor([[1, 2, 2]])
    stage._vlm_pixel_values_chunks = [image[:0], image]
    stage._vlm_image_grid_hws_chunks = [grid[:0], grid]
    stage._vlm_chunk_idx = 0
    first = next_vlm_media_chunk(stage)
    assert first.pixel_values is None and first.image_grid is None
    second = next_vlm_media_chunk(stage)
    assert second.pixel_values is image and second.image_grid is grid
    assert stage._vlm_chunk_idx == 2
    with pytest.raises(ValueError, match="exhausted at microbatch 2"):
        next_vlm_media_chunk(stage)
    assert stage._vlm_chunk_idx == 2


def test_missing_metadata_chunk_does_not_partially_advance_cursor():
    stage = torch.nn.Module()
    stage._vlm_pixel_values_chunks = [torch.ones(1, 3, 4, 4)]
    stage._vlm_image_grid_hws_chunks = []
    with pytest.raises(ValueError, match="image metadata chunks exhausted"):
        next_vlm_media_chunk(stage)
    assert not hasattr(stage, "_vlm_chunk_idx")


def test_no_staged_media_leaves_module_unchanged():
    stage = torch.nn.Module()
    assert next_vlm_media_chunk(stage) is None
    assert not hasattr(stage, "_vlm_chunk_idx")


def test_pixel_padding_keeps_original_coordinates_and_source_values():
    from nemo_automodel.shared.vlm_media import pad_image_tensors

    square = torch.ones(1, 3, 4, 4)
    wide = torch.full((2, 3, 2, 6), 2.0)
    padded = pad_image_tensors([square, wide])
    assert padded.shape == (3, 3, 4, 6)
    torch.testing.assert_close(padded[:1, :, :, :4], square)
    torch.testing.assert_close(padded[1:, :, :2], wide)
    assert not padded[:1, :, :, 4:].any()
    assert not padded[1:, :, 2:].any()
    assert square.shape == (1, 3, 4, 4) and wide.shape == (2, 3, 2, 6)


def test_mistral_example_spatial_padding_ratio():
    from nemo_automodel.shared.vlm_media import pad_image_tensors

    # The shipped Mistral fixture uses 8x8 and 8x12 processor images.
    images = [torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 12)]
    padded = pad_image_tensors(images)
    assert padded.numel() / sum(image.numel() for image in images) == pytest.approx(1.2)
    opposite_aspects = [torch.zeros(1, 3, 8, 2), torch.zeros(1, 3, 2, 8)]
    padded = pad_image_tensors(opposite_aspects)
    assert padded.numel() / sum(image.numel() for image in opposite_aspects) == 4
