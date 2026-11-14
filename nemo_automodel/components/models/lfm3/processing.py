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

"""LFM3-VL processor assembled from the Qwen image and video processors of the pinned Transformers release."""

from __future__ import annotations

import os

from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor
from transformers.processing_utils import ProcessingKwargs

__all__ = ["Lfm3VlProcessor"]


class Lfm3VlProcessorKwargs(ProcessingKwargs, total=False):
    """Call-time defaults: LFM3-VL takes no multimodal token-type ids."""

    _defaults = {
        "text_kwargs": {"padding": False, "return_token_type_ids": False, "return_mm_token_type_ids": False},
        "videos_kwargs": {"return_metadata": True},
    }


class Lfm3VlProcessor(Qwen3VLProcessor):
    """Qwen3-VL prompt construction over LFM3-VL's tokenizer and patch settings.

    Released checkpoints name image and video processors that exist only as
    remote code. Transformers 5.23 maps LFM3-VL to its Qwen2-VL image and
    Qwen3-VL video processors instead; ``from_pretrained`` builds those from the
    checkpoint's stored settings, so no remote code runs.
    """

    valid_processor_kwargs = Lfm3VlProcessorKwargs

    @property
    def model_input_names(self) -> list[str]:
        """Names of the tensors the processor returns, without Qwen3-VL's multimodal token-type ids."""
        return super(Qwen3VLProcessor, self).model_input_names

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: str | bool | None
    ) -> Lfm3VlProcessor:
        """Load the tokenizer, chat template and patch settings of an LFM3-VL checkpoint.

        Args:
            pretrained_model_name_or_path: Hub repository id or local checkpoint directory.
            **kwargs: Hub loading options such as ``revision``, ``cache_dir`` and ``token``.

        Returns:
            Processor backed by the native Qwen image and video processors.
        """
        processor_dict, _ = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls(
            image_processor=Qwen2VLImageProcessor.from_dict(processor_dict["image_processor"]),
            tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
            video_processor=Qwen3VLVideoProcessor.from_dict(processor_dict["video_processor"]),
            chat_template=processor_dict.get("chat_template"),
        )
