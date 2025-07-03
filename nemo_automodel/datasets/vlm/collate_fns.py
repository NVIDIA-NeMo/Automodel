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
import torch

from nemo_automodel.datasets.vlm.utils import extract_skipped_token_ids
from nemo_automodel.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG
from unittest.mock import MagicMock

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False
    process_vision_info = MagicMock()


def create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token=None):
    """
    Create loss mask by finding start of turn token positions, similar to squad.py approach.
    
    Args:
        input_ids: List or tensor of token IDs for a single example
        processor: Processor/tokenizer to convert token string to ID
        start_of_response_token: String token that marks the start of turns (e.g., "<start_of_turn>model\n")
        
    Returns:
        loss_mask: List of 0/1 flags where 0 = masked (prompt), 1 = unmasked (response)
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = input_ids.tolist()
    if isinstance(start_of_response_token, str):
        start_of_response_token_id = tokenizer(start_of_response_token, add_special_tokens=False)["input_ids"]
        start_of_turn_token_id = start_of_response_token_id[0]
        first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
        response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + len(start_of_response_token_id) - 1
    else:
        response_start = 0
    
    loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)
    return loss_mask


def create_batch_loss_masks(batch_input_ids, processor, start_of_response_token=None):
    """
    Create loss masks for a batch of input_ids.
    
    Args:
        batch_input_ids: Tensor of shape (batch_size, seq_len)
        processor: Processor/tokenizer to convert token string to ID
        start_of_response_token: String token that marks the start of turns
        
    Returns:
        loss_mask: Tensor of shape (batch_size, seq_len) where 0 = masked, 1 = unmasked
    """
    batch_size, seq_len = batch_input_ids.shape
    loss_masks = []
    
    for i in range(batch_size):
        single_input_ids = batch_input_ids[i]
        loss_mask = create_loss_mask_with_start_of_response_token(
            single_input_ids, processor, start_of_response_token
        )
        # Pad or truncate to seq_len
        if len(loss_mask) < seq_len:
            loss_mask.extend([0] * (seq_len - len(loss_mask)))
        elif len(loss_mask) > seq_len:
            loss_mask = loss_mask[:seq_len]
        
        loss_masks.append(loss_mask)
    
    return torch.tensor(loss_masks, dtype=torch.float, device=batch_input_ids.device)


def qwen2_5_collate_fn(examples: list, processor, start_of_response_token="<|im_start|>assistant\n") -> dict[str, torch.Tensor]:
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
    loss_mask = create_batch_loss_masks(batch["input_ids"], processor, start_of_response_token=start_of_response_token)
    batch["loss_mask"] = loss_mask


    return batch


def default_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_mask = create_batch_loss_masks(batch["input_ids"], processor, start_of_response_token=start_of_response_token)
    batch["loss_mask"] = loss_mask
    # To check the unmasked tokens
    # print(processor.tokenizer.decode(batch['input_ids'][loss_mask == 1], skip_special_tokens=True))

    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "default": default_collate_fn,
}
