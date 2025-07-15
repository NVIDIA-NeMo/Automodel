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
import io
import torch
from transformers import BatchFeature

from nemo_automodel.datasets.vlm.utils import extract_skipped_token_ids
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
    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = input_ids.tolist()

    if start_of_response_token is None:
        return [1] * len(input_ids)

    if isinstance(start_of_response_token, str):
        start_of_response_token_id = tokenizer(start_of_response_token, add_special_tokens=False)["input_ids"]
        start_of_turn_token_id = start_of_response_token_id[0]
    if isinstance(start_of_response_token, str) and input_ids.count(start_of_turn_token_id) >= 2:
        first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
        response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + len(
            start_of_response_token_id
        )
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
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'
    speech_prompt = "Transcribe the Turkish audio clip."
    answer_suffix = "<|end|>"

    def pad_sequence(sequences, padding_side='right', padding_value=0):
        assert padding_side in ['right', 'left']
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
        for i, seq in enumerate(sequences):
            length = seq.size(0)
            if padding_side == 'right':
                output.data[i, :length] = seq
            else:
                output.data[i, -length:] = seq
        return output
    
    def cat_with_pad(tensors, dim, padding_value=0):
        ndim = tensors[0].dim()
        assert all(t.dim() == ndim for t in tensors[1:]), 'All tensors must have the same number of dimensions'
        out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
        out_size[dim] = sum(t.shape[dim] for t in tensors)
        output = tensors[0].new_full(out_size, padding_value)
        index = 0
        for t in tensors:
            slices = [slice(0, t.shape[d]) for d in range(ndim)]
            slices[dim] = slice(index, index + t.shape[dim])
            output[slices] = t
            index += t.shape[dim]
        return output

    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []

    for example in examples:
        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + speech_prompt,
        }
        prompt = processor.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, audios=[(example["audio"]["array"], example["audio"]["sampling_rate"])], return_tensors='pt')
        answer = f"{example['transcription']}{answer_suffix}"
        answer_ids = processor.tokenizer(answer, return_tensors='pt').input_ids
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, -answer_ids.shape[1]:] = answer_ids
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        input_audio_embeds_list.append(inputs.input_audio_embeds)
        audio_embed_sizes_list.append(inputs.audio_embed_sizes)
        audio_attention_mask_list.append(inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool))
    
    # Squeeze batch dimension before padding, then unsqueeze back
    input_ids_squeezed = [ids.squeeze(0) for ids in input_ids_list]
    labels_squeezed = [labels.squeeze(0) for labels in labels_list]
    
    input_ids = pad_sequence(input_ids_squeezed, padding_side='left', padding_value=0)
    labels = pad_sequence(labels_squeezed, padding_side='left', padding_value=0)
    audio_attention_mask = (
        pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
        if len(audio_attention_mask_list) > 1 else None
    )
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)
    batch_size = input_ids.shape[0]
    return BatchFeature({
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'input_audio_embeds': input_audio_embeds,
        'audio_embed_sizes': audio_embed_sizes,
        'audio_attention_mask': audio_attention_mask,
        'input_mode': torch.full((batch_size,), 2, dtype=torch.long),
    })


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
        add_generation_prompt=False,
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
