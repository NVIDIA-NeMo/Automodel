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

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from nemo_automodel.components.datasets.vlm.mock import build_mock_vlm_dataset


@pytest.fixture(scope="module")
def toy_ds():
    return build_mock_vlm_dataset(num_samples=5, seed=42)


# ---------- basic structure ---------------------------------------------------


def test_dataset_length(toy_ds):
    assert len(toy_ds) == 5


def test_sample_has_conversation_key(toy_ds):
    for sample in toy_ds:
        assert list(sample.keys()) == ["conversation"]


def test_conversation_has_user_and_assistant_turns(toy_ds):
    for sample in toy_ds:
        conv = sample["conversation"]
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"


# ---------- user turn ---------------------------------------------------------


def test_user_content_has_image_and_text(toy_ds):
    for sample in toy_ds:
        content = sample["conversation"][0]["content"]
        types = [item["type"] for item in content]
        assert types[-1] == "text"
        assert all(t == "image" for t in types[:-1])


def test_images_are_pil(toy_ds):
    for sample in toy_ds:
        content = sample["conversation"][0]["content"]
        for item in content:
            if item["type"] == "image":
                assert isinstance(item["image"], Image.Image)


def test_image_size(toy_ds):
    for sample in toy_ds:
        content = sample["conversation"][0]["content"]
        for item in content:
            if item["type"] == "image":
                assert item["image"].size == (256, 256)


# ---------- assistant turn ----------------------------------------------------


def test_assistant_content_is_text(toy_ds):
    for sample in toy_ds:
        content = sample["conversation"][1]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert isinstance(content[0]["text"], str)


# ---------- multiple images ---------------------------------------------------


def test_multiple_images_per_sample():
    ds = build_mock_vlm_dataset(num_samples=3, num_images_per_sample=3)
    for sample in ds:
        content = sample["conversation"][0]["content"]
        image_items = [item for item in content if item["type"] == "image"]
        assert len(image_items) == 3


# ---------- custom image size -------------------------------------------------


def test_custom_image_size():
    ds = build_mock_vlm_dataset(num_samples=2, image_size=(64, 64))
    img = ds[0]["conversation"][0]["content"][0]["image"]
    assert img.size == (64, 64)


# ---------- custom responses --------------------------------------------------


def test_custom_responses_cycle():
    responses = ["A", "B"]
    ds = build_mock_vlm_dataset(num_samples=4, responses=responses)
    for i, sample in enumerate(ds):
        text = sample["conversation"][1]["content"][0]["text"]
        assert text == responses[i % len(responses)]


# ---------- determinism -------------------------------------------------------


def test_determinism_with_seed():
    ds1 = build_mock_vlm_dataset(num_samples=3, seed=123)
    ds2 = build_mock_vlm_dataset(num_samples=3, seed=123)

    for s1, s2 in zip(ds1, ds2):
        # Check text matches
        assert s1["conversation"][1] == s2["conversation"][1]
        assert s1["conversation"][0]["content"][-1] == s2["conversation"][0]["content"][-1]

        # Check image pixels match
        for item1, item2 in zip(s1["conversation"][0]["content"], s2["conversation"][0]["content"]):
            if item1["type"] == "image":
                assert list(item1["image"].tobytes()) == list(item2["image"].tobytes())


# ---------- PreTokenizedDatasetWrapper truncate mode --------------------------


def test_pretokenized_wrapper_truncate_mode():
    """Verify truncate mode clips input_ids, labels, and attention_mask to max_length."""
    from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

    SEQ_LEN = 20
    MAX_LEN = 10

    # Fake dataset returning a single conversation
    mock_ds = build_mock_vlm_dataset(num_samples=1, seed=0)

    # Fake processor that returns tensors of known length
    processor = MagicMock()
    processor.return_value = {
        "input_ids": torch.arange(SEQ_LEN).unsqueeze(0),  # (1, 20)
        "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),  # (1, 20)
        "mm_token_type_ids": torch.zeros(SEQ_LEN, dtype=torch.long),  # 1D (20,)
    }
    # chat_template needs to produce some text
    processor.apply_chat_template = MagicMock(return_value="fake template text")
    processor.image_token = "<image>"

    # Mock build_labels_from_template to return non-negative labels for the second half
    labels = torch.full((SEQ_LEN,), -100, dtype=torch.long)
    labels[SEQ_LEN // 2 :] = torch.arange(SEQ_LEN // 2)

    with patch(
        "nemo_automodel.components.datasets.vlm.datasets.build_labels_from_template",
        return_value=labels.unsqueeze(0),
    ), patch(
        "nemo_automodel.components.datasets.vlm.datasets._preload_media",
        side_effect=lambda ex, proc, **kw: ex,
    ), patch(
        "nemo_automodel.components.datasets.vlm.datasets._conversation_has_media",
        return_value=True,
    ), patch(
        "nemo_automodel.components.datasets.vlm.datasets._extract_media_from_conversations",
        return_value=([], []),
    ):
        wrapper = PreTokenizedDatasetWrapper(mock_ds, processor, max_length=MAX_LEN, truncate=True)
        sample = wrapper[0]

    assert sample["input_ids"].shape[0] == MAX_LEN
    assert sample["attention_mask"].shape[0] == MAX_LEN
    assert sample["labels"].shape[0] == MAX_LEN
    # Labels should not all be -100 (label building happened before truncation)
    assert not torch.all(sample["labels"] == -100)
    # 1D mm_token_type_ids should also be truncated
    if "mm_token_type_ids" in sample:
        assert sample["mm_token_type_ids"].shape[0] == MAX_LEN
