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


def test_empty_then_image_then_video_consumes_each_chunk_exactly_once():
    stage = torch.nn.Module()
    image = torch.ones(1, 3, 4, 4)
    video = torch.ones(2, 8)
    grid = torch.tensor([[1, 2, 2]])
    stage._vlm_pixel_values_chunks = [image[:0], image, image[:0]]
    stage._vlm_image_grid_hws_chunks = [grid[:0], grid, grid[:0]]
    stage._vlm_pixel_values_videos_chunks = [video[:0], video[:0], video]
    stage._vlm_video_grid_thw_chunks = [grid[:0], grid[:0], grid]
    stage._vlm_chunk_idx = 0
    first = next_vlm_media_chunk(stage)
    assert first.pixel_values is None and first.pixel_values_videos is None
    second = next_vlm_media_chunk(stage)
    assert second.pixel_values is image and second.image_grid is grid
    assert second.pixel_values_videos is None
    third = next_vlm_media_chunk(stage)
    assert third.pixel_values is None
    assert third.pixel_values_videos is video and third.video_grid is grid
    assert stage._vlm_chunk_idx == 3
    with pytest.raises(ValueError, match="exhausted at microbatch 3"):
        next_vlm_media_chunk(stage)
    assert stage._vlm_chunk_idx == 3


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


def test_patch_media_and_optional_grid_are_consumed_atomically():
    stage = torch.nn.Module()
    pixels = torch.randn(1, 3, 4, 4)
    patches = torch.randn(3, 3, 2, 2)
    counts = torch.tensor([3])
    newline = torch.tensor([False, True, False])
    stage._vlm_pixel_values_chunks = [pixels[:0], pixels]
    stage._vlm_image_grid_hws_chunks = [None, None]
    stage._vlm_patch_pixel_values_chunks = [patches[:0], patches]
    stage._vlm_num_patches_chunks = [counts[:0], counts]
    stage._vlm_patch_newline_mask_chunks = [newline[:0], newline]
    first = next_vlm_media_chunk(stage)
    assert first.pixel_values is None and first.patch_pixel_values is None
    second = next_vlm_media_chunk(stage)
    assert second.image_grid is None
    assert second.patch_pixel_values is patches and second.num_patches is counts
    assert second.patch_newline_mask is newline
    stage._vlm_chunk_idx = 1
    stage._vlm_num_patches_chunks = []
    with pytest.raises(ValueError, match="patch counts chunks exhausted"):
        next_vlm_media_chunk(stage)
    assert stage._vlm_chunk_idx == 1


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
