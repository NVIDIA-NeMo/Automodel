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

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.models.step3p7.processing_step3 import (
    GPUToTensor,
    ImagePatcher,
    Step3VisionProcessor,
    Step3VLProcessor,
)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "<im_patch>": 99,
            "<patch_start>": 1,
            "<patch_end>": 2,
            "<patch_newline>": 3,
            "<im_start>": 4,
            "<im_end>": 5,
        }
        self.last_text = None
        self.last_kwargs = None

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, text, **kwargs):
        self.last_text = text
        self.last_kwargs = kwargs
        return {"input_ids": [[len(str(item))] for item in text]}

    def batch_decode(self, *args, **kwargs):
        return ["decoded", args, kwargs]

    def decode(self, *args, **kwargs):
        return f"decoded:{args}:{kwargs}"


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    return Step3VLProcessor(tokenizer=FakeTokenizer())


def test_gpu_to_tensor_handles_pil_rgb_numpy_rgb_and_grayscale():
    converter = GPUToTensor()
    pil_tensor = converter(Image.new("RGB", (2, 2), (255, 0, 0)))
    assert pil_tensor.shape == (3, 2, 2)

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_tensor = converter(rgb)
    assert rgb_tensor.shape == (3, 2, 2)
    assert rgb_tensor.dtype == torch.float32

    gray = np.zeros((2, 2), dtype=np.uint8)
    gray_tensor = converter(gray)
    assert gray_tensor.shape == (3, 2, 2)


def test_gpu_to_tensor_cpu_branch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tensor = GPUToTensor()(np.zeros((2, 2, 3), dtype=np.float32))
    assert tensor.device.type == "cpu"


def test_step3_vision_processor_main_and_patch_transforms():
    image = Image.new("RGB", (4, 2), (10, 20, 30))
    processor = Step3VisionProcessor(size=4, interpolation_mode="nearest", patch_size=2)
    main = processor(image)["pixel_values"]
    patch = processor(image, is_patch=True)["pixel_values"]
    assert main.shape == (1, 3, 4, 4)
    assert patch.shape == (1, 3, 2, 2)


def test_image_patcher_geometry_and_crop_branches():
    patcher = ImagePatcher()
    assert patcher.determine_window_size(700, 600) == 0
    assert patcher.determine_window_size(700, 300) == 300
    assert patcher.determine_window_size(1200, 200) == 200
    assert patcher.determine_window_size(1200, 800) == 504

    windows, grid = patcher.slide_window(6, 4, [(4, 4)], [(3, 4)])
    assert grid == (2, 1)
    assert windows[-1] == (2, 0, 4, 4)
    windows, grid = patcher.slide_window(4, 6, [(4, 4)], [(4, 3)])
    assert grid == (1, 2)
    assert windows[-1] == (0, 2, 4, 4)
    with pytest.raises(AssertionError):
        patcher.slide_window(6, 4, [(4, 4)], [(3, 4)], img_rate_thr=1.1)

    square = Image.new("RGB", (4, 4))
    assert patcher.square_pad(square) is square
    assert patcher.square_pad(Image.new("RGB", (2, 4))).size == (4, 4)

    assert patcher.get_image_size_for_padding(8, 1) == (8, 8)
    assert patcher.get_image_size_for_padding(8, 8) == (8, 8)
    assert patcher.get_image_size_for_preprocess(6048, 3024) == (3024, 1512)
    assert patcher.get_image_size_for_crop(450, 1000, 504) == (450, 1008)
    assert patcher.get_image_size_for_crop(1000, 450, 504) == (1008, 450)
    assert patcher.patch_crop(Image.new("RGB", (8, 8)), 1, 2, 3, 4).size == (4, 3)


def test_image_patcher_call_and_num_patches_cover_no_patch_and_patch_paths():
    patcher = ImagePatcher()
    assert patcher.get_num_patches(100, 100) == (0, 0)
    num_patches, full_rows = patcher.get_num_patches(1200, 800)
    assert num_patches > 0
    assert full_rows >= 0

    image, patches, mask = patcher(Image.new("RGB", (100, 100)))
    assert image.size == (100, 100)
    assert patches == []
    assert mask is None

    image, patches, mask = patcher(Image.new("RGB", (8, 1)))
    assert image.size == (8, 8)
    assert patches == []
    assert mask is None

    image, patches, mask = patcher(Image.new("RGB", (1200, 200)))
    assert image.size == (1200, 200)
    assert patches

    image, patches, mask = patcher(Image.new("RGB", (1200, 800)))
    assert image.size[0] <= 3024
    assert patches
    assert mask is not None


def test_processor_helpers_and_placeholder_replacement(processor):
    assert processor.image_token_id == 99
    assert processor.get_num_image_tokens(100, 100) == 171

    text, token_ids = processor._get_patch_repl(2, [True, False])
    assert "<patch_newline>" in text
    assert token_ids.count(99) == 2 * processor.num_patch_feature_size

    image_text, image_ids = processor._get_image_repl(2)
    assert image_text.count("<im_start>") == 2
    assert image_ids.count(99) == 2 * processor.num_image_feature_size

    merged_text, merged_ids = processor._get_image_repl_features(1, 0, None)
    assert "<im_start>" in merged_text
    assert merged_ids.count(99) == processor.num_image_feature_size

    assert processor.replace_placeholder("a <im_patch> b", "<im_patch>", ["IMG"]) == "a IMG b"
    with pytest.raises(ValueError, match="placeholders"):
        processor.replace_placeholder("<im_patch> <im_patch>", "<im_patch>", ["IMG"])

    assert processor.batch_decode([1])[0] == "decoded"
    assert processor.decode([1]).startswith("decoded:")

    class FakeImagePreprocessor:
        def __call__(self, image, is_patch=False):
            return {"pixel_values": torch.full((1, 1), 1 if is_patch else 0)}

    processor.image_preprocessor = FakeImagePreprocessor()
    assert [item.item() for item in processor._convert_images_to_pixel_values([Image.new("RGB", (1, 1))])] == [0]
    assert [
        item.item() for item in processor._convert_images_to_pixel_values([Image.new("RGB", (1, 1))], is_patch=True)
    ] == [1]


def test_normalize_batched_images_branches(processor):
    img = Image.new("RGB", (2, 2))
    assert processor._normalize_batched_images(None, 2) == [[], []]
    assert processor._normalize_batched_images(img, 1) == [[img]]
    assert processor._normalize_batched_images([], 2) == [[], []]
    assert processor._normalize_batched_images([[img], []], 2) == [[img], []]
    assert processor._normalize_batched_images([img], 1) == [[img]]
    assert processor._normalize_batched_images([img, img], 2) == [[img], [img]]
    with pytest.raises(ValueError, match="Expected 2 image groups"):
        processor._normalize_batched_images([[img]], 2)
    with pytest.raises(ValueError, match="Batched Step3 image inputs"):
        processor._normalize_batched_images([img, img, img], 2)


def test_processor_call_without_images_forwards_tokenizer_kwargs(processor):
    result = processor(
        text="hello",
        images=None,
        padding=True,
        return_attention_mask=True,
        return_dict=True,
    )
    assert result["input_ids"] == [[5]]
    assert processor.tokenizer.last_text == ["hello"]
    assert processor.tokenizer.last_kwargs == {"padding": True, "return_attention_mask": True}

    empty = processor(text=None, images=None)
    assert empty["input_ids"] == []


def test_processor_call_with_images_builds_pixels_patches_and_replaced_text(processor):
    raw_img = Image.new("RGB", (4, 4), (128, 64, 32))
    patch_img = Image.new("RGB", (2, 2), (32, 64, 128))
    processor.image_preprocessor.fetch_images = lambda images: images
    processor.patcher = lambda image: (image, [patch_img, patch_img], [True, False])

    def fake_pixels(image, is_patch=False):
        value = 2.0 if is_patch else 1.0
        return {"pixel_values": torch.full((1, 3, 2, 2), value)}

    processor.image_preprocessor.__call__ = fake_pixels
    processor._convert_images_to_pixel_values = lambda images, is_patch=False: [
        fake_pixels(image, is_patch=is_patch)["pixel_values"] for image in images
    ]

    batch = processor(text=["x <im_patch> y", "z"], images=[[raw_img], []])

    assert batch["pixel_values"].shape == (1, 3, 2, 2)
    assert batch["patch_pixel_values"].shape == (2, 3, 2, 2)
    assert batch["num_patches"] == [2]
    assert batch["patch_newline_mask"].tolist() == [True, False]
    assert "<patch_start>" in processor.tokenizer.last_text[0]
    assert processor.tokenizer.last_text[1] == "z"
