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

# The placeholder expansion follows the HuggingFace Inkling 2 processor (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/processing_inkling2.py

"""Native Inkling 2 processor: image tiles plus one placeholder token per image tile.

Code provenance: the placeholder expansion follows ``Inkling2Processor`` in Hugging Face
Transformers (https://github.com/huggingface/transformers, commit
109a7d19bf667168837f32929e00b791e195d14d, ``src/transformers/models/inkling2/processing_inkling2.py``,
Apache-2.0, Copyright 2026 the HuggingFace Team).
"""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.hub import resolve_revision

from .image_processing import Inkling2ImageProcessor

_END_OF_SAMPLING_TOKEN = "<|content_model_end_sampling|>"


class Inkling2Processor(ProcessorMixin):
    """Combine the Inkling 2 image processor with the tokenizer.

    ``__call__`` and ``apply_chat_template`` come from ``ProcessorMixin``: each image placeholder in
    the text is expanded to one ``image_token`` per image tile, so ``input_ids`` holds exactly as
    many placeholders as the vision tower returns embeddings.
    """

    def __init__(
        self,
        image_processor: Inkling2ImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None = None,
        image_token: str = "<|unused_200054|>",
    ) -> None:
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            raise ValueError(f"The tokenizer has no id for the image token {image_token!r}")
        self.image_token = image_token
        self.image_token_id = image_token_id
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: BatchFeature, image_idx: int, **kwargs: Any) -> str:
        """Return the placeholder run for one image.

        Args:
            image_inputs: Image-processor output; ``image_grid_thw`` is a tensor of shape
                ``[images, 3]`` holding each image's ``(time, rows, columns)`` in tiles.
            image_idx: Index of the image in the flattened batch.
            **kwargs: Image-processing arguments forwarded by ``ProcessorMixin``, unused here.

        Returns:
            ``image_token`` repeated once per tile of the image.
        """
        del kwargs
        return self.image_token * int(image_inputs["image_grid_thw"][image_idx].prod())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> Inkling2Processor:
        """Load the processor from a checkpoint with the pinned Transformers release.

        ``ProcessorMixin.from_pretrained`` resolves the image processor through ``AutoImageProcessor``,
        which does not know ``Inkling2ImageProcessor`` before Transformers registers ``inkling2``.
        This reads ``processor_config.json`` and the chat template directly and builds both parts.
        The tokenizer gets ``<|content_model_end_sampling|>`` as its end-of-sequence token and as its
        padding token when the checkpoint defines neither, as batched training requires padding.

        Args:
            pretrained_model_name_or_path: Hugging Face model ID or local snapshot directory.
            cache_dir: Optional Hugging Face cache directory.
            force_download: Whether to re-download cached files.
            local_files_only: Whether to use only cached or local files.
            token: Optional Hugging Face access token.
            revision: Branch, tag or commit; resolved once so every file comes from one commit.
            **kwargs: Further download arguments forwarded to the processor-config and tokenizer loaders.

        Returns:
            The configured processor.
        """
        load_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "token": token,
            **kwargs,
        }
        load_kwargs["revision"] = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        processor_dict, _ = cls.get_processor_dict(pretrained_model_name_or_path, **load_kwargs)
        image_config = dict(processor_dict.get("image_processor", {}))
        image_processor_type = image_config.pop("image_processor_type", Inkling2ImageProcessor.__name__)
        if image_processor_type != Inkling2ImageProcessor.__name__:
            raise ValueError(
                f"Expected an {Inkling2ImageProcessor.__name__} config in {pretrained_model_name_or_path}, "
                f"got {image_processor_type!r}"
            )

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = _END_OF_SAMPLING_TOKEN
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        init_kwargs = {key: processor_dict[key] for key in ("chat_template", "image_token") if key in processor_dict}
        return cls(Inkling2ImageProcessor(**image_config), tokenizer, **init_kwargs)


__all__ = ["Inkling2Processor"]
