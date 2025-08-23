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

import torch

# Common special tokens across VLM models
QWEN_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
LLAVA_TOKENS = ["<image>", "<pad>"]
LLAMA_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
    "<|image|>",
]
GEMMA_TOKENS = ["<image_soft_token>"]

GEMMA_3N_TOKENS = [
    "<image_soft_token>",
    "<audio_soft_token>",
    "<start_of_audio>",
    "<start_of_image>",
    "<end_of_audio>",
    "<end_of_image>",
]

PAD_TOKENS = set(QWEN_TOKENS + LLAVA_TOKENS + LLAMA_TOKENS + GEMMA_TOKENS + GEMMA_3N_TOKENS)


def extract_skipped_token_ids(processor):
    """
    Returns list of tokens to mask in labels.

    Extracted from NeMo's HFAutoModelForImageTextToText.extract_skipped_token_ids
    """
    if processor is None:
        return torch.IntTensor([])
    tokenizer = getattr(processor, "tokenizer", processor)

    skipped_token_ids = []
    for key, val in tokenizer.added_tokens_decoder.items():
        if str(val) in PAD_TOKENS:
            skipped_token_ids.append(key)

    return torch.IntTensor(list(set(skipped_token_ids)))


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
