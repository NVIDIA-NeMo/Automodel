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

"""Native PP media ownership follows document image counts."""

import pytest
import torch

from nemo_automodel.components.datasets.vlm.pp_media import chunk_vlm_media


@pytest.mark.parametrize("pixel_shape", [(2, 4, 6), (2, 3, 4, 4)])
def test_explicit_counts_override_accidental_batch_size_match(pixel_shape):
    pixels = torch.arange(torch.tensor(pixel_shape).prod()).reshape(pixel_shape)
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixel_chunks, grid_chunks = chunk_vlm_media(
        pixels, grids, batch_size=2, n_microbatches=2, n_images_per_sample=torch.tensor([2, 0])
    )
    torch.testing.assert_close(pixel_chunks[0], pixels)
    torch.testing.assert_close(grid_chunks[0], grids)
    assert pixel_chunks[1].shape[0] == grid_chunks[1].shape[0] == 0


def test_four_dimensional_images_follow_unequal_counts():
    pixels = torch.arange(3 * 3 * 4 * 4).reshape(3, 3, 4, 4)
    sizes = torch.tensor([[4, 4], [4, 4], [4, 4]])
    chunks, size_chunks = chunk_vlm_media(
        pixels, sizes, batch_size=2, n_microbatches=2, n_images_per_sample=torch.tensor([1, 2])
    )
    torch.testing.assert_close(chunks[0], pixels[:1])
    torch.testing.assert_close(chunks[1], pixels[1:])
    torch.testing.assert_close(size_chunks[0], sizes[:1])
    torch.testing.assert_close(size_chunks[1], sizes[1:])
