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


# ---------- PreTokenizedDatasetWrapper truncation ----------------------------


class _StubProcessor:
    """Minimal processor stub that returns deterministic tokenized output."""

    class _tokenizer:
        pad_token_id = 0
        eos_token = "</s>"

        @staticmethod
        def convert_tokens_to_ids(token):
            return None

        @staticmethod
        def decode(token):
            return str(token.item() if isinstance(token, torch.Tensor) else token)

    tokenizer = _tokenizer()

    def apply_chat_template(self, conversations, *, tokenize=False, **kwargs):
        return "stub template text"

    def __call__(self, *, text, images=None, videos=None, return_tensors="pt", **kwargs):
        seq_len = 64
        input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        # Simulate mm_token_type_ids as 1D (some processors emit this shape)
        mm_token_type_ids = torch.zeros(seq_len, dtype=torch.long)
        mm_token_type_ids[:10] = 1  # first 10 tokens are "image" type
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }


def test_pretokenized_wrapper_truncate_shapes():
    """PreTokenizedDatasetWrapper with truncate=True produces exact max_length output."""
    from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

    raw_ds = build_mock_vlm_dataset(num_samples=3, seed=0)
    ml = 32
    wrapper = PreTokenizedDatasetWrapper(raw_ds, _StubProcessor(), max_length=ml, truncate=True)

    sample = wrapper[0]
    assert sample["input_ids"].shape == (ml,)
    assert sample["attention_mask"].shape == (ml,)
    assert sample["labels"].shape == (ml,)


def test_pretokenized_wrapper_truncate_labels_not_all_ignored():
    """After truncation, labels should contain some non-(-100) values."""
    from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

    raw_ds = build_mock_vlm_dataset(num_samples=3, seed=0)
    ml = 32
    wrapper = PreTokenizedDatasetWrapper(raw_ds, _StubProcessor(), max_length=ml, truncate=True)

    sample = wrapper[0]
    assert (sample["labels"] != -100).any(), "Labels are all -100 after truncation"


def test_pretokenized_wrapper_truncate_mm_token_type_ids():
    """mm_token_type_ids (1D) should be truncated to max_length."""
    from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

    raw_ds = build_mock_vlm_dataset(num_samples=3, seed=0)
    ml = 32
    wrapper = PreTokenizedDatasetWrapper(raw_ds, _StubProcessor(), max_length=ml, truncate=True)

    sample = wrapper[0]
    assert "mm_token_type_ids" in sample, "mm_token_type_ids missing from output"
    assert sample["mm_token_type_ids"].shape[0] <= ml


# ---------- pad_collate_fn with mm_token_type_ids ----------------------------


def test_pad_collate_fn_mm_token_type_ids():
    """pad_collate_fn pads mm_token_type_ids and trims it with autoregressive shift."""
    from nemo_automodel.components.datasets.vlm.collate_fns import pad_collate_fn

    seq_a, seq_b = 10, 8
    examples = [
        {
            "input_ids": torch.arange(seq_a, dtype=torch.long),
            "attention_mask": torch.ones(seq_a, dtype=torch.long),
            "labels": torch.arange(seq_a, dtype=torch.long),
            "mm_token_type_ids": torch.ones(seq_a, dtype=torch.long),
        },
        {
            "input_ids": torch.arange(seq_b, dtype=torch.long),
            "attention_mask": torch.ones(seq_b, dtype=torch.long),
            "labels": torch.arange(seq_b, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(seq_b, dtype=torch.long),
        },
    ]

    batch = pad_collate_fn(examples, _StubProcessor())

    assert "mm_token_type_ids" in batch
    # After autoregressive shift, seq dim is (pad_to - 1)
    expected_seq = max(seq_a, seq_b) - 1
    assert batch["mm_token_type_ids"].shape == (2, expected_seq)
    assert batch["input_ids"].shape == (2, expected_seq)


def test_pad_collate_fn_mm_token_type_ids_2d():
    """pad_collate_fn handles mm_token_type_ids as 2D (1, seq_len) tensor."""
    from nemo_automodel.components.datasets.vlm.collate_fns import pad_collate_fn

    seq = 10
    examples = [
        {
            "input_ids": torch.arange(seq, dtype=torch.long),
            "attention_mask": torch.ones(seq, dtype=torch.long),
            "labels": torch.arange(seq, dtype=torch.long),
            "mm_token_type_ids": torch.ones(1, seq, dtype=torch.long),  # 2D
        },
    ]

    batch = pad_collate_fn(examples, _StubProcessor())

    assert "mm_token_type_ids" in batch
    assert batch["mm_token_type_ids"].shape == (1, seq - 1)
