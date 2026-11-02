# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Transformers processor construction for Muse Spark Open."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from nemo_automodel.components.models.muse_spark.image_processing import MuseSparkImageProcessor

_AUDIO_SAMPLE_RATE = 24_000
_AUDIO_SAMPLES_PER_TOKEN = 1_920


class MuseSparkProcessorKwargs(ProcessingKwargs, total=False):
    """Standard text/image/video/audio processing arguments."""

    _defaults = {"text_kwargs": {"padding": False}}


def _flatten_media_groups(media: Any, batch_size: int) -> tuple[list[Any], list[int]]:
    """Flatten per-sample nested media while retaining sample counts."""
    if media is None:
        return [], [0] * batch_size
    if batch_size == 1 and not isinstance(media, (list, tuple)):
        return [media], [1]
    values = list(media) if isinstance(media, (list, tuple)) else [media]
    if len(values) == batch_size and all(isinstance(value, (list, tuple)) for value in values):
        groups = [list(value) for value in values]
    elif batch_size == 1:
        groups = [values]
    elif len(values) == batch_size:
        groups = [[value] for value in values]
    else:
        raise ValueError(f"Cannot align {len(values)} Muse Spark Open media values with batch size {batch_size}")
    return [value for group in groups for value in group], [len(group) for group in groups]


def _validate_grouped_placeholders(
    texts: list[str],
    placeholder: str,
    counts: list[int],
    num_replacements: int,
) -> None:
    """Check that every sample has exactly one placeholder per media value."""
    if sum(counts) != num_replacements:
        raise ValueError("Muse Spark Open media counts do not match replacement count")
    for sample_idx, count in enumerate(counts):
        if texts[sample_idx].count(placeholder) != count:
            raise ValueError(
                f"Muse Spark Open sample {sample_idx} has {texts[sample_idx].count(placeholder)} "
                f"{placeholder!r} placeholders for {count} media values"
            )


class MuseSparkProcessor(ProcessorMixin):
    """Combine the Muse Spark Open tokenizer with native multimodal preprocessing."""

    valid_processor_kwargs = MuseSparkProcessorKwargs

    def __init__(
        self,
        image_processor: MuseSparkImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None = None,
        *,
        image_token: str = "<image>",
        patch_token: str = "<|patch|>",
        image_start_token: str = "<|image_start|>",
        image_end_token: str = "<|image_end|>",
        video_token: str = "<video>",
        video_patch_token: str = "<|video|>",
        video_start_token: str = "<|vid_start|>",
        video_end_token: str = "<|vid_end|>",
        video_separator_token: str = "<|vid_frame_separator|>",
        audio_token: str = "<audio>",
        audio_patch_token: str = "<|audio|>",
        audio_start_token: str = "<|soft_audio_start|>",
        audio_end_token: str = "<|soft_audio_end|>",
        timestamp_format: str = "Time: {s:.1f}s",
    ) -> None:
        """Initialize tokens used to expand media placeholders."""
        self.image_token = image_token
        self.patch_token = patch_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.video_token = video_token
        self.video_patch_token = video_patch_token
        self.video_start_token = video_start_token
        self.video_end_token = video_end_token
        self.video_separator_token = video_separator_token
        self.audio_token = audio_token
        self.audio_patch_token = audio_patch_token
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.timestamp_format = timestamp_format
        self.image_token_id = tokenizer.convert_tokens_to_ids(patch_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(video_patch_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_patch_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Any | None = None,
        text: str | list[str] | None = None,
        videos: Any | None = None,
        audio: Any | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Tokenize prompts after expanding image/video/audio placeholders.

        Expansion goes through ``ProcessorMixin.get_text_with_replacements`` so the
        processor can return ``text_replacement_offsets``; ``apply_chat_template``
        needs them to keep ``{% generation %}`` spans aligned with expanded media.
        """
        if images is None and text is None and videos is None and audio is None:
            raise ValueError("MuseSparkProcessor requires text or media")
        texts = [text] if isinstance(text, str) else list(text or [])
        if not texts:
            texts = [""]
        merged = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=getattr(self.tokenizer, "init_kwargs", {}),
            **kwargs,
        )
        text_kwargs = dict(merged["text_kwargs"])
        return_text_replacement_offsets = text_kwargs.pop("return_text_replacement_offsets", False)

        flat_images, image_counts = _flatten_media_groups(images, len(texts))
        image_inputs: dict[str, Any] = {}
        image_replacements: list[str] = []
        if flat_images:
            image_inputs = dict(self.image_processor(flat_images, **merged["images_kwargs"]))
            token_counts = image_inputs.pop("image_token_counts").tolist()
            image_replacements = [
                self.image_start_token + self.patch_token * int(count) + self.image_end_token for count in token_counts
            ]
            _validate_grouped_placeholders(texts, self.image_token, image_counts, len(image_replacements))

        flat_videos, video_counts = _flatten_media_groups(videos, len(texts))
        video_inputs: dict[str, Any] = {}
        video_replacements: list[str] = []
        if flat_videos:
            features = [self.image_processor.preprocess_video(video) for video in flat_videos]
            for feature in features:
                pieces = [self.video_start_token]
                for clip_idx, timestamp in enumerate(feature.timestamps):
                    if clip_idx:
                        pieces.append(self.video_separator_token)
                    pieces.append(self.timestamp_format.format(s=timestamp))
                    pieces.append(self.video_patch_token * feature.tokens_per_clip)
                pieces.append(self.video_end_token)
                video_replacements.append("".join(pieces))
            _validate_grouped_placeholders(texts, self.video_token, video_counts, len(video_replacements))
            video_inputs = {
                "pixel_values_videos": torch.cat([feature.patches for feature in features]),
                "video_grid_thw": torch.tensor([feature.grid_thw for feature in features], dtype=torch.long),
            }

        flat_audio, audio_counts = _flatten_media_groups(audio, len(texts))
        audio_inputs: dict[str, Any] = {}
        audio_replacements: list[str] = []
        if flat_audio:
            waveforms = []
            for waveform in flat_audio:
                value = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
                if value.ndim > 1:
                    value = value.mean(dim=0)
                value = value.flatten()
                waveforms.append(value)
                token_count = math.ceil(value.numel() / _AUDIO_SAMPLES_PER_TOKEN)
                audio_replacements.append(
                    self.audio_start_token + self.audio_patch_token * token_count + self.audio_end_token
                )
            _validate_grouped_placeholders(texts, self.audio_token, audio_counts, len(audio_replacements))
            audio_inputs = {"audio_values": waveforms}

        texts, text_replacement_offsets = self.get_text_with_replacements(
            texts, image_replacements, video_replacements, audio_replacements
        )
        text_inputs = self.tokenizer(texts, **text_kwargs)
        if return_text_replacement_offsets:
            text_inputs["text_replacement_offsets"] = text_replacement_offsets
        data = {**text_inputs, **image_inputs, **video_inputs, **audio_inputs}
        return_tensors = text_kwargs.get("return_tensors")
        return BatchFeature(data=data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    @property
    def model_input_names(self) -> list[str]:
        """Return model inputs produced by this processor."""
        names = [
            *self.tokenizer.model_input_names,
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "audio_values",
        ]
        return list(dict.fromkeys(names))


def _load_chat_template(path_or_id: str, **kwargs: Any) -> str | None:
    """Load the checkpoint's nonstandard Muse chat-template filename."""
    filename = "muse_spark_1_2_oss.jinja"
    local = os.path.join(path_or_id, filename)
    if not os.path.isfile(local):
        if os.path.isdir(path_or_id):
            return None
        try:
            from huggingface_hub import hf_hub_download

            hub_kwargs = {key: kwargs[key] for key in ("cache_dir", "revision", "token") if key in kwargs}
            local = hf_hub_download(path_or_id, filename, **hub_kwargs)
        except EntryNotFoundError:
            return None
    with open(local, encoding="utf-8") as stream:
        return stream.read()


def build_muse_spark_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> MuseSparkProcessor:
    """Build the native processor directly from a Muse Spark Open checkpoint."""
    load_keys = {"cache_dir", "force_download", "local_files_only", "revision", "subfolder", "token"}
    load_kwargs = {key: value for key, value in kwargs.items() if key in load_keys}
    tokenizer_kwargs = dict(load_kwargs)
    tokenizer_kwargs["fix_mistral_regex"] = kwargs.get("fix_mistral_regex", True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config_dict, _ = MuseSpark1xOmniConfig.get_config_dict(pretrained_model_name_or_path, **load_kwargs)
    config_cls = MuseSparkConfig if config_dict.get("model_type") == "muse_spark" else MuseSpark1xOmniConfig
    config = config_cls.from_dict(config_dict)
    navit = config.vision_navit_config
    image_processor = MuseSparkImageProcessor(
        patch_size=int(navit.get("patch_size", 14)),
        patch_temporal=int(navit.get("patch_temporal", 2)),
        downsample_factor=int(navit.get("downsample_factor", 2)),
        min_image_tokens=int(navit.get("min_num_image_tokens", 1)),
        max_image_tokens=int(navit.get("max_num_image_tokens", 4096)),
        max_video_tokens=int(navit.get("max_num_video_tokens", 4096)),
        max_tokens_per_frame=int(navit.get("max_num_tokens_per_frame", 144)),
        num_frames_per_video=int(navit.get("num_frames_per_video", 96)),
        sampling_fps=navit.get("sampling_fps", 2.0),
        frame_factor=int(navit.get("frame_factor", 2) or 2),
        image_mean=tuple(navit.get("image_mean", (0.5, 0.5, 0.5))),
        image_std=tuple(navit.get("image_std", (0.5, 0.5, 0.5))),
    )
    chat_template = _load_chat_template(pretrained_model_name_or_path, **load_kwargs) or tokenizer.chat_template
    # AutoModel's generic VLM label builder asks the tokenizer to render a
    # minimal conversation in order to derive the assistant delimiters.  The
    # released checkpoint stores its template under a non-standard filename,
    # so publish the same template on both processor and tokenizer.
    if chat_template is not None and tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template
    return MuseSparkProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        image_token="<|patch|>" if isinstance(config, MuseSparkConfig) else "<image>",
        video_token="<|video|>" if isinstance(config, MuseSparkConfig) else "<video>",
        audio_token="<|audio|>" if isinstance(config, MuseSparkConfig) else "<audio>",
        timestamp_format=navit.get("video_frame_timestamp_format", "Time: {s:.1f}s"),
    )


__all__ = ["MuseSparkProcessor", "build_muse_spark_processor"]
