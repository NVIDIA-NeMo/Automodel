# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import MagicMock

import torch

from nemo_automodel.components.datasets.vlm.utils import extract_skipped_token_ids
from nemo_automodel.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False
    process_vision_info = MagicMock()


def create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token=None):
    r"""
    Create loss mask by finding start of turn token positions, similar to squad.py approach.

    Args:
        input_ids: List or tensor of token IDs for a single example
        processor: Processor/tokenizer to convert token string to ID
        start_of_response_token: String token that marks the start of turns (e.g., "<start_of_turn>model\n")

    Returns:
        loss_mask: List of 0/1 flags where 0 = masked (prompt), 1 = unmasked (response)
    """

    def find_sequence_in_list(input_ids, target_sequence):
        """Find the starting index of target_sequence in input_ids"""
        if not target_sequence:
            return -1
        for i in range(len(input_ids) - len(target_sequence) + 1):
            if input_ids[i : i + len(target_sequence)] == target_sequence:
                return i
        return -1

    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = input_ids.tolist()

    if start_of_response_token is None:
        return [1] * len(input_ids)

    if isinstance(start_of_response_token, str):
        start_of_response_token_ids = tokenizer(start_of_response_token, add_special_tokens=False)["input_ids"]
        first_occurrence = find_sequence_in_list(input_ids, start_of_response_token_ids)
        response_start = first_occurrence if first_occurrence >= 0 else 0
    else:
        response_start = 0

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0
    loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)

    for i, token_id in enumerate(input_ids):
        if token_id == pad_token_id:
            loss_mask[i] = 0

    return loss_mask


def phi4_mm_collate_fn(examples, processor):
    """Collate function for Phi-4 MM model audio input"""

    # Extract conversations and audio data
    conversations = [example["conversation"] for example in examples]
    audios = [example["audio"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audio_inputs = [(audio["array"], audio["sampling_rate"]) if isinstance(audio, dict) else audio for audio in audios]
    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)

    loss_masks = []
    for i, conversation in enumerate(conversations):
        input_ids = batch["input_ids"][i].tolist()

        assistant_content = conversation[1]["content"]
        assistant_tokens = processor.tokenizer(assistant_content, add_special_tokens=False)["input_ids"]

        loss_mask = [0] * len(input_ids)
        for start_idx in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[start_idx : start_idx + len(assistant_tokens)] == assistant_tokens:
                for j in range(len(assistant_tokens)):
                    loss_mask[start_idx + j] = 1
                break
        loss_masks.append(loss_mask)

    max_len = max(len(mask) for mask in loss_masks)
    padded_loss_masks = [mask + [0] * (max_len - len(mask)) for mask in loss_masks]
    batch["loss_mask"] = torch.tensor(padded_loss_masks, dtype=torch.float)

    labels[batch["loss_mask"] == 0] = -100
    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(
    examples: list, processor, start_of_response_token="<|im_start|>assistant\n"
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    texts = [processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["conversation"])[0] for example in examples]

    batch = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token)
        for input_ids in batch["input_ids"]
    ]
    batch["loss_mask"] = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    return batch


def default_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token)
        for input_ids in batch["input_ids"]
    ]
    batch["loss_mask"] = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "default": default_collate_fn,
}
