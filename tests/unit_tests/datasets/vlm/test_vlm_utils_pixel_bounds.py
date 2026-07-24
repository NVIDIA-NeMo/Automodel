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

"""Tests for the shared VLM image pixel-bound helper.

Draft training and answer regeneration must apply identical bounds, so the
helper they share is tested once here rather than per caller.
"""

from types import SimpleNamespace

from nemo_automodel.components.datasets.vlm.utils import set_image_pixel_bounds


def test_caps_both_flat_attributes_and_the_size_dict():
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(max_pixels=99, min_pixels=1, size={"max_pixels": 99, "min_pixels": 1})
    )
    set_image_pixel_bounds(processor, max_pixels=802816, min_pixels=3136)
    assert processor.image_processor.max_pixels == 802816
    assert processor.image_processor.min_pixels == 3136
    assert processor.image_processor.size == {"max_pixels": 802816, "min_pixels": 3136}


def test_unset_bounds_leave_the_processor_default_alone():
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=99, min_pixels=1))
    set_image_pixel_bounds(processor)
    assert processor.image_processor.max_pixels == 99
    assert processor.image_processor.min_pixels == 1


def test_processor_without_an_image_processor_is_a_no_op():
    set_image_pixel_bounds(SimpleNamespace(), max_pixels=802816)


def test_flat_attribute_is_set_even_without_a_size_dict():
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=99, min_pixels=1, size=None))
    set_image_pixel_bounds(processor, max_pixels=802816)
    assert processor.image_processor.max_pixels == 802816
