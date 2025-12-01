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

from typing import Iterable

import torch


def default_stop_tokens(processor) -> Iterable[str]:
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    candidates = [
        "<end_of_turn>",
        "<|im_end|>",
        "<|eot_id|>",
    ]
    if eos_token is not None:
        candidates.append(eos_token)
    return tuple(candidates)


def json2token(obj, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence.

    From NeMo's automodel_datasets.py
    """
    if type(obj) is dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for k in keys:
            output += rf"<s_{k}>" + json2token(obj[k], sort_json_key) + rf"</s_{k}>"
        return output
    if type(obj) is list:
        return r"<sep/>".join([json2token(item, sort_json_key) for item in obj])
    return str(obj)


def process_text_batch(
    processor,
    texts: list[str],
    images: list | None = None,
) -> dict[str, torch.Tensor]:
    """
    Process a batch of texts and optionally images.

    Args:
        processor: The processor to use for tokenization and image processing
        texts: List of text strings to process
        images: Optional list of images to process

    Returns:
        Dict containing processed batch data
    """
    if images is not None:
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    else:
        batch = processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )

    return batch
