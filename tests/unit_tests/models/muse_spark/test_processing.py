# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open media preprocessing and placeholder contracts."""

from pathlib import Path

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
    _replace_grouped_placeholders,
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
    assert _replace_grouped_placeholders(
        ["x <image> y <image>", "z <image>"],
        "<image>",
        ["<A>", "<B>", "<C>"],
        counts,
    ) == ["x <A> y <B>", "z <C>"]


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


def test_patch_token_placeholders_expand_each_image_once() -> None:
    """v2 expansions may contain the same marker as the next source image."""
    assert _replace_grouped_placeholders(
        ["a<|patch|>b<|patch|>c", "d<|patch|>e"],
        "<|patch|>",
        ["<start><|patch|><|patch|><end>", "<start><|patch|><end>", "<start><|patch|><end>"],
        [2, 1],
    ) == [
        "a<start><|patch|><|patch|><end>b<start><|patch|><end>c",
        "d<start><|patch|><end>e",
    ]


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
