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

"""Spatial padding preserves image contents and original coordinates."""

import pytest
import torch


def test_pixel_padding_keeps_original_coordinates_and_source_values():
    from nemo_automodel.components.datasets.vlm.media import pad_image_tensors

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
    from nemo_automodel.components.datasets.vlm.media import pad_image_tensors

    # The shipped Mistral fixture uses 8x8 and 8x12 processor images.
    images = [torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 12)]
    padded = pad_image_tensors(images)
    assert padded.numel() / sum(image.numel() for image in images) == pytest.approx(1.2)
    opposite_aspects = [torch.zeros(1, 3, 8, 2), torch.zeros(1, 3, 2, 8)]
    padded = pad_image_tensors(opposite_aspects)
    assert padded.numel() / sum(image.numel() for image in opposite_aspects) == 4
