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

from nemo_automodel.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False
    process_vision_info = MagicMock()

try:
    from qwen_omni_utils import process_mm_info

    HAVE_QWEN_OMNI_UTILS = True
except ImportError:
    HAVE_QWEN_OMNI_UTILS = False
    process_mm_info = MagicMock()

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from nemo_automodel.components.datasets.vlm.utils import default_stop_tokens


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


def _extract_assistant_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return ""


def build_labels(
    input_ids_batch: torch.Tensor,
    conversations: Sequence[Sequence[Dict[str, Any]]],
    processor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct label and optional loss-mask tensors aligned to assistant responses."""
    tokenizer = getattr(processor, "tokenizer", processor)

    labels_list: List[torch.Tensor] = []

    for encoded, conversation in zip(input_ids_batch, conversations):
        labels = torch.full_like(encoded, -100)
        search_start_index = 0

        for message in conversation:
            if message.get("role") != "assistant":
                continue

            assistant_text = _extract_assistant_text(message)
            if not assistant_text:
                continue

            assistant_tokens = tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(encoded.device)

            answer_start, answer_end = _find_pattern_indices(encoded, assistant_tokens, search_start_index)

            if answer_end < len(encoded):
                next_token_str = tokenizer.decode(encoded[answer_end])
                if next_token_str.strip() in default_stop_tokens(processor):
                    answer_end += 1

            if answer_start >= 0:
                labels[answer_start:answer_end] = encoded[answer_start:answer_end]
                search_start_index = answer_end
            else:
                logger.warning(
                    (
                        "Unable to find answer segment in the tokenized conversation. "
                        "Skipping labeling for this and subsequent answers. Details:"
                        "\n- Processed Text: %s"
                        "\n- Tokens: %s"
                        "\n- Target Answer Tokens: %s"
                        "\n- Search Start Index: %d"
                    ),
                    conversation,
                    encoded,
                    assistant_tokens,
                    search_start_index,
                )
                break

        labels_list.append(labels)

    labels_tensor = torch.stack(labels_list)
    return labels_tensor


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

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = [example["conversation"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]

    batch = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    return batch


def qwen3_omni_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    use_audio_in_video: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collate function for Qwen3 Omni processors."""
    if not HAVE_QWEN_OMNI_UTILS:
        raise ImportError(
            "qwen_omni_utils is required for qwen3_omni_collate_fn. Install it with: pip install qwen-omni-utils"
        )

    conversations = [example["conversation"] for example in examples]
    texts = [
        processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        for conversation in conversations
    ]

    all_audios = []
    all_images = []
    all_videos = []
    for conversation in conversations:
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        all_audios.append(audios)
        all_images.append(images)
        all_videos.append(videos)

    def has_data(modality_list):
        for item in modality_list:
            if item is None:
                continue
            if isinstance(item, list) and len(item) == 0:
                continue
            return True
        return False

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "padding_side": "right",
    }

    if has_data(all_audios):
        processor_kwargs["audio"] = all_audios
    if has_data(all_images):
        processor_kwargs["images"] = all_images
    if has_data(all_videos):
        processor_kwargs["videos"] = all_videos

    batch = processor(**processor_kwargs)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]
    return batch


def kimi_vl_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for KimiVL processors."""
    conversations = [example["conversation"] for example in examples]
    texts = [
        processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        for conversation in conversations
    ]

    images: List[Any] = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item.get("image"))

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["padding"] = "max_length"
    if images:
        processor_kwargs["images"] = images

    batch = processor(**processor_kwargs)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]
    return batch


def nemotron_parse_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown>",
) -> Dict[str, torch.Tensor]:
    """
    Collate function for NVIDIA Nemotron-Parse models.

    The Nemotron-Parse processor does not expose a chat template, so we build the
    prompt + answer string manually, mask the prompt tokens, and keep the
    image preprocessing handled by the processor.
    """

    conversations = [example["conversation"] for example in examples]

    images: List[Any] = []
    targets: List[str] = []
    for conversation in conversations:
        image = None
        assistant_text = ""

        for message in conversation:
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image = item.get("image")
                            break
            elif role == "assistant" and not assistant_text:
                assistant_text = _extract_assistant_text(message)

            if image is not None and assistant_text:
                break

        images.append(image)
        targets.append(assistant_text)

    texts = [f"{task_prompt}{target}" for target in targets]

    batch = processor(images=images, text=texts, padding=True, return_tensors="pt")

    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    decoder_start_token_id = getattr(tokenizer, "decoder_start_token_id", None) or getattr(
        tokenizer, "bos_token_id", None
    )
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None or decoder_start_token_id is None:
        raise ValueError("Nemotron-Parse collate_fn requires pad_token_id and decoder_start_token_id.")

    decoder_input_ids = batch["input_ids"].clone()
    decoder_input_ids[:, 0] = decoder_start_token_id
    decoder_input_ids[:, 1:] = batch["input_ids"][:, :-1]

    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

    batch["decoder_input_ids"] = decoder_input_ids[:, 1:]
    batch["decoder_attention_mask"] = decoder_attention_mask[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    return batch


def default_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Default collate function for multimodal VLM datasets."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = [example["conversation"] for example in examples]
    processor_kwargs = {
        "tokenize": True,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length
        processor_kwargs["padding"] = "max_length"
    batch = processor.apply_chat_template(conversations, **processor_kwargs)

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in batch:
        if batch[key].shape == input_shape and key != "labels":
            batch[key] = batch[key][:, :-1]
    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "KimiVLProcessor": kimi_vl_collate_fn,
    "NemotronParseProcessor": nemotron_parse_collate_fn,
    "default": default_collate_fn,
}
