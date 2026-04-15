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

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset
import torch

from nemo_automodel.components.datasets.vlm.collate_fns import (
    HAVE_QWEN_VL_UTILS,
    MISSING_QWEN_VL_UTILS_MSG,
    _count_media_per_sample,
    _ensure_rgb,
    _inject_thinking_prefix_tokens,
    mask_fake_vision_tokens_batch,
)

logger = logging.getLogger(__name__)


def _resolve_gemma4_markers(tokenizer):
    """Derive the assistant-turn marker IDs and end-of-turn ID from a live
    chat-template application.

    This avoids relying on ``convert_tokens_to_ids`` or ``encode``, both of
    which can return wrong results for Gemma4 special tokens because the
    tokenizer stores them under internal names that don't match their surface
    strings (e.g. ``<start_of_turn>`` → 3 instead of 105).

    Returns
    -------
    (assistant_marker, end_of_turn_id)
        ``assistant_marker`` is the list of token ids that immediately precede
        the assistant's content (e.g. ``[105, 4368, 107]`` for
        ``<start_of_turn>model\\n``).
        ``end_of_turn_id`` is the single token id that closes a turn (106).
    """
    def _extract_ids(result) -> List[int]:
        """Handle both plain list and BatchEncoding (UserDict, not dict) returns."""
        try:
            return list(result["input_ids"])
        except (KeyError, TypeError):
            return list(result)

    sentinel = "XSENTINELMARKERX"
    dummy = [{"role": "user", "content": "u"}, {"role": "assistant", "content": sentinel}]
    all_ids: List[int] = _extract_ids(
        tokenizer.apply_chat_template(dummy, tokenize=True, add_generation_prompt=False)
    )

    sentinel_ids: List[int] = tokenizer.encode(sentinel, add_special_tokens=False)

    # Locate the sentinel inside all_ids
    for i in range(len(all_ids) - len(sentinel_ids) + 1):
        if all_ids[i : i + len(sentinel_ids)] == sentinel_ids:
            # Token immediately after sentinel should be <end_of_turn>
            end_idx = i + len(sentinel_ids)
            if end_idx >= len(all_ids):
                raise ValueError("No token found after sentinel – cannot determine end_of_turn_id.")
            end_of_turn_id: int = all_ids[end_idx]

            # Derive the user-turn length so we can slice out the assistant marker
            user_ids: List[int] = _extract_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "u"}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            assistant_marker: List[int] = all_ids[len(user_ids) : i]

            if not assistant_marker:
                raise ValueError("Assistant marker is empty – template layout unexpected.")

            logger.debug(
                "_resolve_gemma4_markers: marker=%s end_of_turn_id=%d",
                assistant_marker,
                end_of_turn_id,
            )
            return assistant_marker, end_of_turn_id

    raise ValueError(
        f"Sentinel '{sentinel}' (ids={sentinel_ids}) not found in template output {all_ids}."
    )


def _build_gemma4_labels_from_template(
    input_ids_batch: torch.Tensor,
    processor,
) -> torch.Tensor:
    """Build training labels for Gemma4 by scanning token IDs for role markers.

    Gemma4's chat template wraps every assistant turn as:
        <start_of_turn>model\\n … content … <end_of_turn>

    We locate these markers directly in ``input_ids`` (no re-tokenisation),
    which avoids the BPE context-sensitivity bugs of the generic
    ``build_labels`` fallback.

    Labels are set to the real token ids for the **content + <end_of_turn>**
    region; all other positions are ``-100``.
    """
    tokenizer = getattr(processor, "tokenizer", processor)

    # ------------------------------------------------------------------
    # Resolve the marker token ids once per batch.
    # <start_of_turn> is a special token → convert_tokens_to_ids is safe.
    # "model\n" is plain text → encode without special tokens.
    # ------------------------------------------------------------------
    try:
        assistant_marker, end_of_turn_id = _resolve_gemma4_markers(tokenizer)
    except Exception as exc:
        logger.warning(
            "_build_gemma4_labels_from_template: failed to resolve turn markers (%s). "
            "Returning all-(-100) labels for this batch.",
            exc,
        )
        return torch.full_like(input_ids_batch, -100)

    marker_len = len(assistant_marker)
    marker_tensor = torch.tensor(
        assistant_marker, dtype=input_ids_batch.dtype, device=input_ids_batch.device
    )

    labels_list: List[torch.Tensor] = []

    for encoded in input_ids_batch:
        labels = torch.full_like(encoded, -100)
        seq_len = len(encoded)
        i = 0
        num_assistant_turns = 0

        while i <= seq_len - marker_len:
            if torch.equal(encoded[i : i + marker_len], marker_tensor):
                content_start = i + marker_len  # first token of assistant content

                # Scan forward to find the closing <end_of_turn>.
                content_end = content_start
                while content_end < seq_len and encoded[content_end].item() != end_of_turn_id:
                    content_end += 1

                # Include the <end_of_turn> stop token so the model learns to emit it.
                if content_end < seq_len:
                    content_end += 1

                labels[content_start:content_end] = encoded[content_start:content_end]
                num_assistant_turns += 1
                i = content_end
            else:
                i += 1

        if num_assistant_turns == 0:
            logger.warning(
                "_build_gemma4_labels_from_template: no assistant turn markers found "
                "in a sequence of length %d. Labels will be all -100 for this sample.",
                seq_len,
            )

        labels_list.append(labels)

    return torch.stack(labels_list)


def _as_list(path_or_dataset: str | Sequence[str]) -> list[str]:
    if isinstance(path_or_dataset, str):
        return [path_or_dataset]
    return [str(x) for x in path_or_dataset]


def _to_conversation(example: dict) -> dict:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected each sample to contain a `messages` list.")

    conversation = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue

        content = msg.get("content", "")
        if content is None:
            content = ""
        content = str(content)

        conversation.append(
            {
                "role": role,
                "content": [{"type": "text", "text": content}],
            }
        )

    if not conversation:
        raise ValueError("Conversation is empty after message normalization.")

    return {"conversation": conversation}


def _is_valid_sample(example: dict) -> bool:
    """Return True only if the sample has at least one non-empty assistant turn."""
    msgs = example.get("messages")
    if not isinstance(msgs, list):
        return False
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and str(msg.get("content") or "").strip():
            return True
    return False


def make_parquet_messages_dataset(
    path_or_dataset,
    split: str = "train",
    shuffle_seed: int | None = 42,
    limit_dataset_samples: int | None = None,
    **kwargs,
):
    """Load local parquet files with a `messages` column and convert them to VLM conversations.

    The returned dataset matches the schema expected by
    `nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn`
    and `gemma4_prefix_collate_fn`, i.e. each sample is a dict with a
    `conversation` field containing OpenAI-style role/content messages rendered
    as text-only multimodal parts.

    Samples without a valid (non-empty) assistant turn are filtered out before
    conversion to avoid nan loss during training.
    """

    data_files = _as_list(path_or_dataset)
    missing = [p for p in data_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet file(s): {missing}")

    dataset = load_dataset("parquet", data_files=data_files, split=split)

    # Filter out samples with no assistant turn or empty assistant response
    before = len(dataset)
    dataset = dataset.filter(_is_valid_sample)
    dropped = before - len(dataset)
    if dropped:
        import logging
        logging.getLogger(__name__).warning(
            f"Filtered {dropped}/{before} samples with missing or empty assistant turns."
        )

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset.map(_to_conversation, remove_columns=dataset.column_names)


def gemma4_prefix_truncating_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Gemma4 collate that truncates overlong text samples instead of dropping them."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = _ensure_rgb([example["conversation"] for example in examples])

    processor_kwargs = {
        "tokenize": True,
        "padding": "max_length" if max_length is not None else True,
        "truncation": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if max_length is not None:
        processor_kwargs["max_length"] = max_length

    batch = processor.apply_chat_template(conversations, **processor_kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    batch = _inject_thinking_prefix_tokens(batch, tokenizer)

    if max_length is not None and batch["input_ids"].size(1) > max_length:
        for key in list(batch.keys()):
            value = batch[key]
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.size(1) > max_length and key != "pixel_values":
                batch[key] = value[:, :max_length]

    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in batch:
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)

    labels = _build_gemma4_labels_from_template(batch["input_ids"], processor)
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in list(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor) and value.shape == input_shape and key != "labels":
            batch[key] = value[:, :-1]

    # Drop samples whose assistant response was fully truncated (all labels == -100).
    # This prevents nan loss / zero grad_norm when max_length cuts off the response.
    # Guard: only filter if at least one valid sample remains; otherwise keep the
    # whole batch as-is (a single nan step is safer than an empty-tensor crash).
    valid = (batch["labels"] != -100).any(dim=-1)  # [B]
    if valid.any() and not valid.all():
        for key in list(batch.keys()):
            val = batch[key]
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == valid.shape[0]:
                batch[key] = val[valid]
        examples = [ex for ex, ok in zip(examples, valid.tolist()) if ok]

    fake_indices = [i for i, example in enumerate(examples) if example.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(count > 0 for count in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(count > 0 for count in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch
