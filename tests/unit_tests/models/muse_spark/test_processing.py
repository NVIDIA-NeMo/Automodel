# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open media preprocessing and placeholder contracts."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.muse_spark.image_processing import (
    MuseSparkImageProcessor,
    find_resize_target,
)
from nemo_automodel.components.models.muse_spark.processing import (
    _flatten_media_groups,
    _validate_grouped_placeholders,
    build_muse_spark_processor,
)
from tests.unit_tests.models.muse_spark.test_hf_release import tiny_hf_config


def test_resize_and_patchify_match_post_shuffle_token_count() -> None:
    """Raw NaViT patches and post-shuffle placeholders use their distinct grids."""
    assert find_resize_target(8, 12, token_patch_size=4, minimum_tokens=1, maximum_tokens=16) == (8, 12, 6)
    processor = MuseSparkImageProcessor(
        patch_size=2,
        patch_temporal=2,
        downsample_factor=2,
        min_image_tokens=1,
        max_image_tokens=16,
    )

    result = processor.preprocess(torch.zeros(3, 8, 12, dtype=torch.uint8), return_tensors="pt")

    assert result["pixel_values"].shape == (24, 24)
    assert result["image_grid_thw"].tolist() == [[1, 4, 6]]
    assert result["image_token_counts"].tolist() == [6]
    assert torch.isfinite(result["pixel_values"]).all()


def test_grouped_media_placeholders_remain_sample_local() -> None:
    """Flattening multiple samples retains the counts used for expansion."""
    flattened, counts = _flatten_media_groups([["a", "b"], ["c"]], batch_size=2)
    assert flattened == ["a", "b", "c"]
    assert counts == [2, 1]
    _validate_grouped_placeholders(["x <image> y <image>", "z <image>"], "<image>", counts, 3)
    with pytest.raises(ValueError, match="sample 1 has 0"):
        _validate_grouped_placeholders(["x <image> y <image>", "z"], "<image>", counts, 3)
    with pytest.raises(ValueError, match="do not match"):
        _validate_grouped_placeholders(["x <image> y <image>", "z <image>"], "<image>", counts, 2)


def test_video_preprocessing_preserves_temporal_clip_metadata() -> None:
    """Video frames share the image tower and retain clip timestamps."""
    processor = MuseSparkImageProcessor(
        patch_size=2,
        patch_temporal=2,
        downsample_factor=2,
        min_image_tokens=1,
        max_video_tokens=16,
        max_tokens_per_frame=16,
        num_frames_per_video=4,
        sampling_fps=None,
        frame_factor=2,
    )
    frames = torch.stack([torch.full((3, 8, 12), index, dtype=torch.uint8) for index in range(4)])

    result = processor.preprocess_video((frames, {"fps": 2.0, "frames_indices": [0, 1, 2, 3]}))

    assert result.patches.shape == (48, 24)
    assert result.grid_thw == (2, 4, 6)
    assert result.tokens_per_clip == 6
    assert result.timestamps == (0.0, 1.0)


def test_patch_token_placeholders_expand_each_image_once(tmp_path: Path) -> None:
    """v2 expansions may contain the same marker as the next source image."""
    processor = _build_v2_processor(tmp_path)
    texts, offsets = processor.get_text_with_replacements(
        ["a<|patch|>b<|patch|>c", "d<|patch|>e"],
        ["<start><|patch|><|patch|><end>", "<start><|patch|><end>", "<start><|patch|><end>"],
    )
    assert texts == [
        "a<start><|patch|><|patch|><end>b<start><|patch|><end>c",
        "d<start><|patch|><end>e",
    ]
    # Offsets describe each pre-expansion placeholder and its expanded span.
    assert [(o["span"], o["new_span"]) for o in offsets[0]] == [((1, 10), (1, 31)), ((11, 20), (32, 53))]
    assert [(o["span"], o["new_span"]) for o in offsets[1]] == [((1, 10), (1, 22))]


def test_processor_returns_text_replacement_offsets_on_request(tmp_path: Path) -> None:
    """apply_chat_template realigns generation spans from these offsets."""
    processor = _build_v2_processor(tmp_path)
    images = [Image.new("RGB", (8, 4)), Image.new("RGB", (4, 4))]
    assert "text_replacement_offsets" not in processor(images=[images], text=["<|patch|>hello<|patch|>"])
    result = processor(images=[images], text=["<|patch|>hello<|patch|>"], return_text_replacement_offsets=True)
    offsets = result["text_replacement_offsets"]
    assert [o["type"] for o in offsets[0]] == ["image", "image"]
    assert [o["span"] for o in offsets[0]] == [(0, 9), (14, 23)]
    assert offsets[0][0]["replacement"] == "<|image_start|>" + "<|patch|>" * 2 + "<|image_end|>"
    assert offsets[0][1]["new_span"][0] == offsets[0][0]["new_span"][1] + len("hello")


def _build_v2_processor(tmp_path: Path):
    vocabulary = {"<unk>": 0, "hello": 1, "<|patch|>": 2, "<|image_start|>": 3, "<|image_end|>": 4, "<pad>": 5}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|patch|>", "<|image_start|>", "<|image_end|>"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    tiny_hf_config().save_pretrained(tmp_path)
    return build_muse_spark_processor(str(tmp_path))


def test_v2_processor_uses_standard_template_and_nested_vision_config(tmp_path: Path) -> None:
    """The released processor must consume its saved template and patch dimensions."""
    vocabulary = {"<unk>": 0, "hello": 1, "<|patch|>": 2, "<|image_start|>": 3, "<|image_end|>": 4, "<pad>": 5}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|patch|>", "<|image_start|>", "<|image_end|>"],
    )
    template = "{% for m in messages %}{% for p in m.content %}{% if p.type == 'image' %}<|patch|>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}"
    tokenizer.chat_template = template
    tokenizer.save_pretrained(tmp_path)
    tiny_hf_config().save_pretrained(tmp_path)
    processor = build_muse_spark_processor(str(tmp_path))
    assert processor.chat_template == template
    assert processor.image_processor.patch_size == 2
    assert processor.image_processor.downsample_factor == 2
    assert processor.image_token == "<|patch|>"
    images = [Image.new("RGB", (8, 4)), Image.new("RGB", (4, 4))]
    result = processor(images=[images], text=["<|patch|>hello<|patch|>"], return_tensors="pt")
    assert result["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]
    assert (result["input_ids"] == processor.image_token_id).sum().item() == 3
    assert result["pixel_values"].shape == (12, 24)
