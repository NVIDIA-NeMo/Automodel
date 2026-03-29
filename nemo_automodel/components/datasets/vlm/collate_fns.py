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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image as PILImage

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


def _decode_single_token(tokenizer, token_id: int) -> str:
    """Decode a single token id across tokenizer implementations.

    Some tokenizers accept an `int` token id, while others require a sequence of
    ids (e.g., `List[int]`). We try the common forms in order.
    """
    try:
        return tokenizer.decode(token_id)
    except Exception:
        try:
            return tokenizer.decode([token_id])
        except Exception:
            try:
                return tokenizer.decode(torch.tensor([token_id]))
            except Exception:
                # Best-effort fallback; stop-token detection will likely fail.
                return str(token_id)


def build_labels(
    input_ids_batch: torch.Tensor,
    conversations: Sequence[Sequence[Dict[str, Any]]],
    processor,
) -> torch.Tensor:
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

            # handle tokenizers that can produce different tokens for text with leading
            # whitespace when tokenized standalone vs in-context
            if answer_start < 0 and assistant_text != assistant_text.lstrip():
                assistant_tokens = tokenizer(
                    assistant_text.lstrip(),
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0].to(encoded.device)
                answer_start, answer_end = _find_pattern_indices(encoded, assistant_tokens, search_start_index)

            if answer_end < len(encoded):
                next_token_id = int(encoded[answer_end].item())
                next_token_str = _decode_single_token(tokenizer, next_token_id)
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
    tokenizer = getattr(processor, "tokenizer", processor)
    texts = [tokenizer.apply_chat_template(conversation, tokenize=False) for conversation in conversations]

    # Prepare audio inputs as (array, sampling_rate) tuples for the remote processor
    audio_inputs = []
    for audio in audios:
        if isinstance(audio, dict):
            audio_inputs.append((audio["array"], audio["sampling_rate"]))
        elif isinstance(audio, (list, tuple)) and len(audio) == 2:
            audio_inputs.append(tuple(audio))
        else:
            audio_inputs.append(audio)

    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )

    # The remote Phi4MM processor sets input_mode as a tensor.
    # Only set it as fallback if the processor didn't provide it.
    if "input_mode" not in batch:
        has_audio = "input_audio_embeds" in batch and batch["input_audio_embeds"].numel() > 0
        has_image = "input_image_embeds" in batch and batch["input_image_embeds"].numel() > 0
        if has_image and has_audio:
            batch["input_mode"] = 3
        elif has_image:
            batch["input_mode"] = 1
        elif has_audio:
            batch["input_mode"] = 2
        else:
            batch["input_mode"] = 0

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

    # Import at call-time to support environments/tests that inject the module
    # after this file is initially imported.
    try:
        from qwen_omni_utils import process_mm_info as _process_mm_info
    except ImportError as exc:
        raise ImportError(
            "qwen_omni_utils is required for qwen3_omni_collate_fn. Install it with: pip install qwen-omni-utils"
        ) from exc

    conversations = [example["conversation"] for example in examples]
    texts = [
        processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        for conversation in conversations
    ]

    all_audios = []
    all_images = []
    all_videos = []
    for conversation in conversations:
        audios, images, videos = _process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
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
        "add_special_tokens": False,
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


def _expand_image_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand single image placeholder tokens to the correct number based on grid_thws.

    For PP, this ensures the sequence length is fixed BEFORE the model forward pass,
    eliminating dynamic sequence expansion inside the model.

    Assumes 1 image per sample (1 placeholder per sequence).

    Args:
        input_ids: (seq_len,) tensor with 1 media_token_id placeholder
        attention_mask: (seq_len,) tensor
        grid_thws: (1, 3) tensor with [t, h, w] for the single image
        media_token_id: Token ID of the image placeholder
        merge_kernel_size: Vision tower's patch merge kernel, default (2, 2)

    Returns:
        expanded_input_ids: Input IDs with placeholder expanded to N tokens
        expanded_attention_mask: Attention mask expanded accordingly
    """
    merge_h, merge_w = merge_kernel_size

    # Calculate number of image tokens: (h // merge_h) * (w // merge_w)
    t, h, w = grid_thws[0].tolist()
    num_image_tokens = (h // merge_h) * (w // merge_w)

    # Find the placeholder position
    placeholder_positions = (input_ids == media_token_id).nonzero(as_tuple=True)[0]
    if len(placeholder_positions) == 0:
        # No placeholder found, return as-is
        return input_ids, attention_mask

    # For 1 image per sample, there should be exactly 1 placeholder
    placeholder_pos = placeholder_positions[0].item()

    # Build expanded tensors
    before = input_ids[:placeholder_pos]
    after = input_ids[placeholder_pos + 1 :]

    # Expand: replace 1 placeholder with num_image_tokens placeholders
    expanded_placeholder = torch.full((num_image_tokens,), media_token_id, dtype=input_ids.dtype)
    expanded_input_ids = torch.cat([before, expanded_placeholder, after])

    # Expand attention mask similarly
    before_mask = attention_mask[:placeholder_pos]
    after_mask = attention_mask[placeholder_pos + 1 :]
    expanded_mask_tokens = torch.ones(num_image_tokens, dtype=attention_mask.dtype)
    expanded_attention_mask = torch.cat([before_mask, expanded_mask_tokens, after_mask])

    return expanded_input_ids, expanded_attention_mask


def kimi_k25_vl_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for Kimi K2.5 VL processors with pre-expanded image tokens.

    For pipeline parallelism, this function:
    1. Processes each sample to get input_ids with 1 placeholder per image
    2. Pre-expands the placeholder to N tokens (N = (h//2)*(w//2) from grid_thws)
    3. Pads all sequences to fixed max_length
    This ensures the model forward pass doesn't change sequence length dynamically.
    """
    conversations = [example["conversation"] for example in examples]

    # Get media token ID
    media_token_id = getattr(processor, "media_placeholder_token_id", None)
    if media_token_id is None and hasattr(processor, "tokenizer"):
        media_token_id = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    if media_token_id is None:
        media_token_id = 163605  # Default for Kimi K2.5

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0

    # Process each sample individually
    all_expanded = []
    all_pixel_values = []
    all_grid_thws = []

    for i, conversation in enumerate(conversations):
        # Collect medias for this conversation
        medias: List[Dict[str, Any]] = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        medias.append({"type": "image", "image": item.get("image")})

        text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)

        processor_kwargs = {
            "text": text,
            "return_tensors": "pt",
        }
        if medias:
            processor_kwargs["medias"] = medias

        sample_batch = processor(**processor_kwargs)

        input_ids = sample_batch["input_ids"][0]
        attention_mask = sample_batch["attention_mask"][0]

        # Pre-expand image tokens if we have grid_thws
        if "grid_thws" in sample_batch and sample_batch["grid_thws"] is not None:
            grid_thws = sample_batch["grid_thws"]

            input_ids, attention_mask = _expand_image_tokens(input_ids, attention_mask, grid_thws, media_token_id)
            all_grid_thws.append(grid_thws)

        if "pixel_values" in sample_batch:
            all_pixel_values.append(sample_batch["pixel_values"])

        all_expanded.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    # Determine target length for padding
    expanded_lens = [b["input_ids"].shape[0] for b in all_expanded]
    batch_max = max(expanded_lens)

    if max_length is not None:
        target_len = max_length
    else:
        target_len = batch_max

    # Pad/truncate to target_len
    padded_input_ids = []
    padded_attention_mask = []

    for batch in all_expanded:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        seq_len = input_ids.shape[0]

        if seq_len < target_len:
            # Pad
            pad_len = target_len - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif seq_len > target_len:
            # Truncate
            input_ids = input_ids[:target_len]
            attention_mask = attention_mask[:target_len]

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }

    if all_pixel_values:
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
    if all_grid_thws:
        # Use image_grid_hws for compatibility with finetune recipe VLM chunking
        result["grid_thws"] = torch.cat(all_grid_thws, dim=0)
        # Also add as image_grid_hws for PP chunking in finetune.py
        result["image_grid_hws"] = result["grid_thws"][:, 1:]  # [N, 3] -> [N, 2] (drop temporal dim, keep H,W)

    # Build labels
    labels = build_labels(
        result["input_ids"],
        conversations,
        processor,
    )
    result["labels"] = labels[:, 1:]

    # Shift inputs (remove last token for autoregressive training)
    input_shape = result["input_ids"].shape
    for key, value in list(result.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            result[key] = value[:, :-1]

    return result


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


def _ensure_rgb(conversations):
    """Convert any PIL images in conversations to RGB to handle RGBA/grayscale inputs."""
    for conv in conversations:
        for turn in conv:
            content = turn.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("image"), PILImage.Image):
                        item["image"] = item["image"].convert("RGB")
    return conversations


def default_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Default collate function for multimodal VLM datasets."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = _ensure_rgb([example["conversation"] for example in examples])
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


def neat_packed_vlm_collater(
    batch: list[dict],
    padding_idx: int = 0,
    max_length: int | None = None,
    attn_implementation: str = "sdpa",
) -> dict:
    """Collater for neat-packed VLM sequences.

    Packs arrive with **variable lengths** (no pre-padding).  This collater:

    1. Pads all text tensors to a common length.
    2. Converts the indexed ``attention_mask`` to the appropriate format:
       - ``flash_attention_2``: keeps the indexed ``[B, S]`` mask (values
         1, 2, … for documents, 0 for padding).  The monkey-patched
         ``_get_unpad_data`` converts this to ``cu_seqlens`` for
         ``flash_attn_varlen_func``.
       - ``sdpa`` / ``eager``: converts to a 4D block-causal bool mask.
    3. Concatenates media tensors across the batch dimension.

    **No autoregressive shift** — it was already applied during packing.

    Args:
        batch: List of packed sample dicts from ``PackedDatasetWrapper``.
        padding_idx: Token ID for padding ``input_ids`` (default 0).
        max_length: If set, pad every batch to this fixed length.
            If ``None`` (default), pad to the longest pack in the batch.
            A fixed length avoids recompilation with ``torch.compile``
            and ensures uniform tensor shapes across steps.
        attn_implementation: Attention backend (``"flash_attention_2"``,
            ``"sdpa"``, or ``"eager"``).

    Returns:
        Dict with batched tensors ready for model forward.
    """
    if not batch:
        return {}

    LABEL_PAD = -100
    use_flash = attn_implementation == "flash_attention_2"

    # Determine pad target: fixed max_length or batch-dynamic
    batch_max = max(
        x["input_ids"].shape[-1] if isinstance(x["input_ids"], torch.Tensor) else len(x["input_ids"]) for x in batch
    )
    max_len = max_length if max_length is not None else batch_max

    def _pad_1d(tensor, pad_value, target_len):
        """Pad a 1D tensor to target_len."""
        t = torch.as_tensor(tensor)
        pad_len = target_len - t.shape[0]
        if pad_len > 0:
            return torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)])
        return t

    # Pad and stack text tensors
    input_ids = torch.stack([_pad_1d(x["input_ids"], padding_idx, max_len) for x in batch])
    labels = torch.stack([_pad_1d(x["labels"], LABEL_PAD, max_len) for x in batch])
    attention_mask = torch.stack([_pad_1d(x["attention_mask"], 0, max_len) for x in batch])

    if use_flash:
        # Keep indexed [B, S] mask for flash_attn_varlen_func.
        # The patched _get_unpad_data will extract per-document cu_seqlens.
        attention_mask_out = attention_mask
    else:
        from nemo_automodel.components.datasets.utils import _indexed_mask_to_4d_block_causal

        attention_mask_out = _indexed_mask_to_4d_block_causal(attention_mask)

    # Handle position_ids: 1D [seq_len] or 3D mRoPE [3, seq_len]
    pos_sample = torch.as_tensor(batch[0]["position_ids"])
    if pos_sample.ndim == 2:
        # mRoPE: [3, seq_len] → pad to [3, max_len], stack to [3, B, max_len]
        def _pad_mrope(pos, target_len):
            t = torch.as_tensor(pos)  # [3, seq_len]
            pad_len = target_len - t.shape[1]
            if pad_len > 0:
                return torch.cat([t, torch.zeros(3, pad_len, dtype=t.dtype)], dim=1)
            return t

        position_ids = torch.stack([_pad_mrope(x["position_ids"], max_len) for x in batch], dim=1)
    else:
        # Standard 1D: [seq_len] → pad to [max_len], stack to [B, max_len]
        position_ids = torch.stack([_pad_1d(x["position_ids"], 0, max_len) for x in batch])

    result: Dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask_out,
    }

    # Store indexed attention mask for loss functions that need per-sample
    # boundaries (e.g. SqrtCrossEntropy).  The indexed mask [B, S] uses
    # values 1,2,3,... per original sample and 0 for padding.  For SDPA the
    # ``attention_mask_out`` is already converted to 4D, so keep a copy.
    if attention_mask.max() > 1:
        result["_packed_seq_ids"] = attention_mask

    # Concatenate media tensors across batch (variable count, no padding needed)
    for key in ("pixel_values", "pixel_values_videos"):
        tensors = [x[key] for x in batch if key in x and x[key] is not None]
        if tensors:
            result[key] = torch.cat(tensors, dim=0).to(torch.bfloat16)

    for key in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        tensors = [x[key] for x in batch if key in x and x[key] is not None]
        if tensors:
            result[key] = torch.cat(tensors, dim=0)

    # Per-pack media counts
    image_counts = [int(x.get("n_images", 0)) for x in batch]
    video_counts = [int(x.get("n_videos", 0)) for x in batch]
    if any(c > 0 for c in image_counts):
        result["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        result["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return result


_nemotron_omni_collate_call_count = 0  # DEBUG: activation dump
_COLLATE_DUMP_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemofw/users/huiyingl/nemotronomni/activation_dumps"  # DEBUG: activation dump


def nemotron_omni_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for NemotronOmni (NemotronH_Nano_VL_V2) model.

    This model uses InternVL-style image tagging where images are split into
    dynamic tiles and <image> tokens in the input are replaced with vision
    embeddings during forward pass.

    The processor (NemotronNanoVLV2Processor) handles:
    - Image preprocessing (dynamic tiling, normalization)
    - Token expansion (<image> -> <img> + N*<image> + </img>)
    - Tokenization

    This collate function:
    1. Extracts conversations and images from examples
    2. Applies chat template and processes through the processor
    3. Builds image_flags tensor (1 for real images, 0 for padding in batch)
    4. Creates labels masking non-assistant tokens with -100

    Args:
        examples: List of dataset examples with 'conversation' key
        processor: NemotronNanoVLV2Processor instance
        max_length: Optional max sequence length for truncation

    Returns:
        Dict with input_ids, attention_mask, pixel_values, image_flags, labels
    """
    conversations = _ensure_rgb([example["conversation"] for example in examples])
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", "<image>")

    # Collect images and build text prompts
    all_images: List[Any] = []
    texts: List[str] = []

    for conversation in conversations:
        # Extract images and convert multimodal content to text with <image> tokens.
        # The NemotronOmni chat template does not natively handle {'type': 'image'}
        # content items, so we replace them with the <image> text token and collect
        # the PIL images separately for the processor.
        conv_images = []
        text_conversation = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                # Multimodal message: flatten image items to <image> token and
                # concatenate text items.
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            img = item.get("image")
                            if img is not None:
                                conv_images.append(img)
                                text_parts.append(image_token)
                        elif item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                text_conversation.append({
                    "role": message["role"],
                    "content": "".join(text_parts),
                })
            else:
                text_conversation.append(message)

        all_images.append(conv_images)

        # Apply chat template to get text with <image> tokens
        text = tokenizer.apply_chat_template(text_conversation, tokenize=False)
        texts.append(text)

    # Process each sample individually to handle variable image counts.
    # Track per-sample pixel_values so we can trim after truncation.
    all_input_ids = []
    all_attention_masks = []
    sample_pixel_values: Dict[int, torch.Tensor] = {}  # sample_idx -> pixel_values

    for i, (text, images) in enumerate(zip(texts, all_images)):
        processor_kwargs = {
            "text": text,
            "return_tensors": "pt",
            "padding": False,
        }
        if images:
            processor_kwargs["images"] = images

        batch_i = processor(**processor_kwargs)
        all_input_ids.append(batch_i["input_ids"][0])
        all_attention_masks.append(batch_i["attention_mask"][0])

        if "pixel_values" in batch_i and batch_i["pixel_values"] is not None:
            sample_pixel_values[i] = batch_i["pixel_values"]

    # Determine pad token and target length
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    seq_lengths = [ids.shape[0] for ids in all_input_ids]
    target_len = max_length if max_length is not None else max(seq_lengths)

    # Pad/truncate sequences
    padded_input_ids = []
    padded_attention_masks = []
    for input_ids, attn_mask in zip(all_input_ids, all_attention_masks):
        seq_len = input_ids.shape[0]
        if seq_len < target_len:
            pad_len = target_len - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)])
        elif seq_len > target_len:
            input_ids = input_ids[:target_len]
            attn_mask = attn_mask[:target_len]
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attn_mask)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
    }

    # After truncation, align pixel_values and <image> tokens per sample.
    # Each tile generates num_image_token <image> tokens. If truncation removed some
    # tokens, we: (1) keep only complete tiles, (2) replace leftover partial-tile
    # <image> tokens with pad_token_id so the counts match exactly.
    img_context_token_id = 18  # from model config
    num_image_token = 256  # (force_image_size/patch_size)^2 * downsample_ratio^2 = (512/16)^2 * 0.25
    adjusted_pixel_values = []
    for i in range(len(padded_input_ids)):
        if i in sample_pixel_values:
            remaining_img_tokens = (padded_input_ids[i] == img_context_token_id).sum().item()
            surviving_tiles = remaining_img_tokens // num_image_token
            expected_tokens = surviving_tiles * num_image_token
            pv = sample_pixel_values[i]
            if surviving_tiles > 0:
                adjusted_pixel_values.append(pv[:surviving_tiles])
            # Replace excess partial-tile <image> tokens with pad
            if expected_tokens < remaining_img_tokens:
                ids = padded_input_ids[i]
                img_positions = (ids == img_context_token_id).nonzero(as_tuple=True)[0]
                # Keep first expected_tokens, replace the rest
                excess_positions = img_positions[expected_tokens:]
                ids[excess_positions] = pad_token_id
                padded_input_ids[i] = ids

    # Re-stack after potential modifications
    result["input_ids"] = torch.stack(padded_input_ids)

    if adjusted_pixel_values:
        pixel_values = torch.cat(adjusted_pixel_values, dim=0)
        result["pixel_values"] = pixel_values.to(torch.bfloat16)

        # image_flags: 1 for each real image tile, shape [total_tiles, 1]
        num_tiles = pixel_values.shape[0]
        result["image_flags"] = torch.ones(num_tiles, 1, dtype=torch.long)

    # Build labels (mask non-assistant tokens with -100)
    labels = build_labels(
        result["input_ids"],
        conversations,
        processor,
    )
    result["labels"] = labels[:, 1:]

    # Shift inputs (remove last token for autoregressive training)
    input_shape = result["input_ids"].shape
    for key, value in list(result.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            result[key] = value[:, :-1]

    # DEBUG: activation dump - save first collated batch on rank 0
    global _nemotron_omni_collate_call_count  # DEBUG: activation dump
    import os as _os  # DEBUG: activation dump
    try:  # DEBUG: activation dump
        import torch.distributed as _dist  # DEBUG: activation dump
        _is_rank0 = (not _dist.is_initialized()) or (_dist.get_rank() == 0)  # DEBUG: activation dump
    except Exception:  # DEBUG: activation dump
        _is_rank0 = True  # DEBUG: activation dump
    if _is_rank0 and _nemotron_omni_collate_call_count == 0:  # DEBUG: activation dump
        collate_dump = {}  # DEBUG: activation dump
        for k, v in result.items():  # DEBUG: activation dump
            if isinstance(v, torch.Tensor):  # DEBUG: activation dump
                collate_dump[k] = v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()  # DEBUG: activation dump
            else:  # DEBUG: activation dump
                collate_dump[k] = v  # DEBUG: activation dump
        _dump_path = _os.path.join(_COLLATE_DUMP_DIR, "collate_output.pt")  # DEBUG: activation dump
        torch.save(collate_dump, _dump_path)  # DEBUG: activation dump
        print(f"[ACTIVATION DUMP] Saved collate_output.pt with keys: {list(collate_dump.keys())}")  # DEBUG: activation dump
        for k, v in collate_dump.items():  # DEBUG: activation dump
            if isinstance(v, torch.Tensor):  # DEBUG: activation dump
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")  # DEBUG: activation dump
    _nemotron_omni_collate_call_count += 1  # DEBUG: activation dump

    return result


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "KimiVLProcessor": kimi_vl_collate_fn,
    "KimiK25Processor": kimi_k25_vl_collate_fn,
    "NemotronParseProcessor": nemotron_parse_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_omni_collate_fn,
    "default": default_collate_fn,
}
