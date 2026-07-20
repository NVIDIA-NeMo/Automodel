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

"""Processor construction helpers for Inkling fine-tuning."""

from typing import Any

from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

_INKLING_END_OF_SAMPLING_TOKEN = "<|content_model_end_sampling|>"


def build_inkling_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> ProcessorMixin:
    """Load Inkling's processor and configure padding with an existing token.

    The published tokenizer does not declare EOS or padding tokens even though
    its chat template terminates assistant responses with an existing
    end-of-sampling token. Reusing that token keeps the checkpoint vocabulary
    unchanged and enables padded fine-tuning batches.

    Args:
        pretrained_model_name_or_path: Hugging Face model ID or local snapshot.
        **kwargs: Additional arguments forwarded to ``AutoProcessor.from_pretrained``.

    Returns:
        The configured Inkling processor.
    """
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = _INKLING_END_OF_SAMPLING_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor
