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
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import torch
from PIL import Image as PILImage

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

logger = logging.getLogger(__name__)

# Default vision-tower patch merge kernel used by `_expand_image_tokens` and any
# caller that needs to predict expanded image-token counts. Keep this as the
# single source of truth so the two computations cannot silently drift apart.
_DEFAULT_MERGE_KERNEL: Tuple[int, int] = (2, 2)

# ---------------------------------------------------------------------------
# Fake image fallback for FSDP / DeepSpeed Zero3
# ---------------------------------------------------------------------------
# Fake images are injected per-sample at __getitem__ time (see datasets.py).
# The helpers live in fake_image.py and are imported here for use in collate
# functions that need to mask vision tokens for samples that were injected.
# ---------------------------------------------------------------------------
from nemo_automodel.components.datasets.vlm.fake_image import (  # noqa: F401
    _FAKE_IMAGE,
    _batch_has_media,
    inject_fake_image_into_conversation,
    mask_fake_vision_tokens_batch,
)
from nemo_automodel.components.datasets.vlm.samplers import _smart_resize_image
from nemo_automodel.components.datasets.vlm.utils import default_stop_tokens

# ---------------------------------------------------------------------------
# Patch BaseVideoProcessor.fetch_videos to use decord (decord2) instead of
# torchcodec.  This is applied at import time so all video processors that
# inherit from BaseVideoProcessor benefit automatically.
# ---------------------------------------------------------------------------
# def _fetch_videos_decord(self, video_url_or_urls, sample_indices_fn=None):
#     if isinstance(video_url_or_urls, list):
#         return list(zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
#     return load_video(video_url_or_urls, backend="decord", sample_indices_fn=sample_indices_fn)


# BaseVideoProcessor.fetch_videos = _fetch_videos_decord


def make_robust_collate(dataset, collate_fn, max_retries=10):
    """Wrap *collate_fn* so that on failure the entire batch is re-sampled.

    Args:
        dataset: The dataset to re-sample from on failure.
        collate_fn: The collate function to wrap.
        max_retries: Maximum number of retry attempts.
    """

    def wrapper(examples):
        last_error = None
        for attempt in range(max_retries):
            try:
                return collate_fn(examples)
            except Exception as e:
                last_error = e
                logger.warning(f"Collate failed (attempt {attempt + 1}/{max_retries}): {e}. Re-sampling batch.")
                examples = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(len(examples))]
        raise RuntimeError(f"Collate failed after {max_retries} retries. Last error: {last_error}")

    return wrapper


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


# ---------------------------------------------------------------------------
# Template-based label builder  (robust replacement for pattern-matching)
# ---------------------------------------------------------------------------
# Chat templates delimit roles with special tokens whose IDs are fixed.
# By scanning ``input_ids`` for the marker sequence
#   <|im_start|>  +  assistant  +  \n
# we can locate every assistant turn without re-tokenizing the text.
# This avoids the BPE context-sensitivity bugs of the old approach.
# ---------------------------------------------------------------------------


def _get_assistant_marker(tokenizer) -> Optional[List[int]]:
    """Return the token-id sequence that introduces an assistant turn.

    For Qwen-family models the marker is ``[<|im_start|>, assistant, \\n]``.
    Returns ``None`` when the tokenizer does not use this convention.
    """
    try:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if im_start is None or im_start == getattr(tokenizer, "unk_token_id", None):
            return None
        role_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
        if not role_ids:
            return None
        return [im_start] + role_ids
    except Exception:
        return None


def _get_stop_token_id(tokenizer) -> Optional[int]:
    """Return the token id of the turn-ending marker (``<|im_end|>``)."""
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            return tid
    except Exception:
        pass
    return None


# Processor types whose chat template uses ``<|im_start|>``/``<|im_end|>``
# markers.  For these we use the fast ``_get_assistant_marker`` /
# ``_get_stop_token_id`` helpers (no dummy-conversation overhead).
_IMSTART_TEMPLATE_PROCESSORS = frozenset(
    {
        "Qwen2VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
        "Qwen3VLMoeProcessor",
        "Qwen3OmniMoeProcessor",
    }
)


def _derive_turn_markers(tokenizer) -> Tuple[List[int], int]:
    """Derive the assistant-turn start marker and end-of-turn token id from the
    tokenizer's own chat template.

    The function applies a minimal dummy conversation that contains a known
    sentinel string as the assistant reply, then locates the sentinel in the
    resulting token sequence.  Everything between the end of the user turn and
    the start of the sentinel becomes the **assistant marker**; the first token
    *after* the sentinel becomes the **end-of-turn id**.

    This approach is robust to BPE context-sensitivity and works for any model
    whose template wraps assistant turns with fixed token sequences — e.g.
    Gemma4's ``<start_of_turn>model\\n`` … ``<end_of_turn>``.

    .. note::
        ``apply_chat_template`` may return a :class:`~transformers.BatchEncoding`
        (a ``UserDict`` subclass, **not** a plain :class:`dict`), so
        ``isinstance(result, dict)`` is ``False``.  We access ``result["input_ids"]``
        directly, which works for both ``BatchEncoding`` and plain ``dict`` / ``list``.

    Returns
    -------
    tuple[list[int], int]
        ``(assistant_marker, end_of_turn_id)``

    Raises
    ------
    ValueError
        If the sentinel cannot be located in the template output or if the
        resulting marker is empty.
    """

    def _extract_ids(result) -> List[int]:
        try:
            return list(result["input_ids"])
        except (KeyError, TypeError):
            return list(result)

    sentinel = "XSENTINELMARKERX"
    all_ids = _extract_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": sentinel}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    user_ids = _extract_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    sentinel_ids = tokenizer.encode(sentinel, add_special_tokens=False)

    for i in range(len(all_ids) - len(sentinel_ids) + 1):
        if all_ids[i : i + len(sentinel_ids)] == sentinel_ids:
            end_idx = i + len(sentinel_ids)
            if end_idx >= len(all_ids):
                raise ValueError(f"No token found after sentinel in template output {all_ids}.")
            end_of_turn_id: int = all_ids[end_idx]
            assistant_marker: List[int] = all_ids[len(user_ids) : i]
            if not assistant_marker:
                raise ValueError(
                    f"Assistant marker is empty (user_len={len(user_ids)}, sentinel_pos={i}). Full sequence: {all_ids}"
                )
            return assistant_marker, end_of_turn_id

    raise ValueError(f"Sentinel '{sentinel}' (ids={sentinel_ids}) not found in template output {all_ids}.")


def _build_labels_from_markers(
    input_ids_batch: torch.Tensor,
    assistant_marker: List[int],
    stop_id: int,
) -> torch.Tensor:
    """Scan ``input_ids`` for ``assistant_marker`` … ``stop_id`` and build labels.

    For each sequence in the batch, every token between the end of an
    assistant marker and the corresponding ``stop_id`` (inclusive) is copied
    into the labels tensor; all other positions are set to ``-100``.

    Parameters
    ----------
    input_ids_batch:
        Shape ``(B, L)``.
    assistant_marker:
        Token-id sequence that opens an assistant turn (e.g.
        ``[<|im_start|>, assistant_id, newline_id]`` for Qwen or
        ``[<start_of_turn>, model_id, newline_id]`` for Gemma4).
    stop_id:
        Single token id that closes a turn (e.g. ``<|im_end|>`` or
        ``<end_of_turn>``).
    """
    marker_len = len(assistant_marker)
    marker_tensor = torch.tensor(assistant_marker, dtype=input_ids_batch.dtype, device=input_ids_batch.device)

    labels_list: List[torch.Tensor] = []

    for encoded in input_ids_batch:
        labels = torch.full_like(encoded, -100)
        seq_len = len(encoded)
        i = 0

        while i <= seq_len - marker_len:
            if torch.equal(encoded[i : i + marker_len], marker_tensor):
                content_start = i + marker_len  # first token of assistant content

                # Scan forward to find the closing stop token.
                content_end = content_start
                while content_end < seq_len and encoded[content_end].item() != stop_id:
                    content_end += 1

                # Include the stop token in labels so the model learns to emit it.
                if content_end < seq_len:
                    content_end += 1

                labels[content_start:content_end] = encoded[content_start:content_end]
                i = content_end
            else:
                i += 1

        labels_list.append(labels)

    return torch.stack(labels_list)


def build_labels_from_template(
    input_ids_batch: torch.Tensor,
    conversations: Sequence[Sequence[Dict[str, Any]]],
    processor,
) -> torch.Tensor:
    """Build training labels by scanning ``input_ids`` for chat-template role markers.

    Instead of re-tokenizing assistant text and searching for it (fragile due
    to BPE context sensitivity), this function locates the structural markers
    that the chat template inserts around each assistant turn and sets labels
    only for the content region.

    Two strategies are attempted in order:

    1. **Fast path** (``_IMSTART_TEMPLATE_PROCESSORS``): for Qwen-family models
       whose tokenizers expose ``<|im_start|>`` / ``<|im_end|>`` via
       :func:`convert_tokens_to_ids`, the marker ids are resolved directly
       without applying any dummy conversation.

    2. **General path** (``_derive_turn_markers``): for all other processors
       (e.g. Gemma4), the assistant-turn markers are derived automatically by
       applying a minimal dummy conversation that contains a sentinel string.
       This handles models whose tokenizers do not reliably expose special-token
       ids via ``convert_tokens_to_ids`` or ``encode``.

    If both strategies fail, the function falls back to the legacy
    :func:`build_labels` (BPE pattern-matching), which logs a warning because
    it is sensitive to tokenisation context and may produce ``num_label_tokens=0``
    / nan loss on some samples.
    """
    processor_type = type(processor).__name__
    tokenizer = getattr(processor, "tokenizer", processor)

    # ------------------------------------------------------------------
    # Fast path: Qwen-family processors with <|im_start|>/<|im_end|>.
    # ------------------------------------------------------------------
    if processor_type in _IMSTART_TEMPLATE_PROCESSORS:
        assistant_marker = _get_assistant_marker(tokenizer)
        stop_id = _get_stop_token_id(tokenizer)
        if assistant_marker is not None and stop_id is not None:
            return _build_labels_from_markers(input_ids_batch, assistant_marker, stop_id)
        logger.warning(
            "Processor %s is listed in _IMSTART_TEMPLATE_PROCESSORS but the tokenizer "
            "does not expose <|im_start|>/<|im_end|>. Trying template-derived markers.",
            processor_type,
        )

    # ------------------------------------------------------------------
    # General path: derive markers from the chat template via sentinel.
    # Handles Gemma4 and any future model automatically.
    # ------------------------------------------------------------------
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            assistant_marker, stop_id = _derive_turn_markers(tokenizer)
            return _build_labels_from_markers(input_ids_batch, assistant_marker, stop_id)
        except Exception as exc:
            logger.warning(
                "Processor %s: could not derive turn markers from chat template (%s). "
                "Falling back to BPE pattern-match labels, which may produce nan loss.",
                processor_type,
                exc,
            )

    # ------------------------------------------------------------------
    # Last-resort fallback: BPE pattern-matching (fragile).
    # ------------------------------------------------------------------
    return build_labels(input_ids_batch, conversations, processor)


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

    labels = build_labels_from_template(
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


def _extract_media_from_conversations(conversations):
    """Extract image and video inputs from conversation content elements.

    Images are returned as-is (PIL Image or path string) for the image processor.
    Videos are returned as path strings so the video processor can read and sample
    them using its own ``fps`` / ``max_frames`` configuration.

    Returns:
        tuple: (images list | None, videos list | None)
    """
    images = []
    videos = []
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                typ = ele.get("type")
                if typ == "image" and "image" in ele:
                    images.append(ele["image"])
                elif typ == "video" and "video" in ele:
                    videos.append(ele["video"])
    return images or None, videos or None


def _count_media_per_sample(conversations):
    """Count images and videos per sample from conversation structure.

    Returns two lists of length ``len(conversations)`` giving the number of
    image and video items in each conversation, respectively.
    """
    image_counts = []
    video_counts = []
    for conv in conversations:
        n_img = n_vid = 0
        for msg in conv:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            n_img += 1
                        elif item.get("type") == "video":
                            n_vid += 1
        image_counts.append(n_img)
        video_counts.append(n_vid)
    return image_counts, video_counts


def qwen2_5_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    conversations = [example["conversation"] for example in examples]

    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]

    images, videos = _extract_media_from_conversations(conversations)

    batch = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )
    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    # Per-sample media counts for PP chunking
    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

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

    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    # Per-sample media counts for PP chunking
    image_counts = [len(imgs) if imgs else 0 for imgs in all_images]
    video_counts = [len(vids) if vids else 0 for vids in all_videos]
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch


def kimi_vl_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for KimiVL processors."""
    conversations = [example["conversation"] for example in examples]

    # Drop overlong samples before processing
    if max_length is not None:
        conversations, kept = _drop_overlong_samples(conversations, processor, max_length)
        examples = [examples[i] for i in kept]

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
        processor_kwargs["truncation"] = False  # Pre-filtering guarantees samples fit
    if images:
        processor_kwargs["images"] = images

    batch = processor(**processor_kwargs)

    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    # Per-sample media counts for PP chunking
    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch


def _expand_image_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int,
    merge_kernel_size: Tuple[int, int] = _DEFAULT_MERGE_KERNEL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand image placeholder tokens to the correct patch counts based on grid_thws.

    For PP, this ensures the sequence length is fixed BEFORE the model forward pass,
    eliminating dynamic sequence expansion inside the model.

    Supports both single-image and multi-image samples.  Each placeholder token is
    expanded to ``(h // merge_h) * (w // merge_w)`` tokens using the corresponding
    entry in *grid_thws*.  The number of placeholders must match ``grid_thws.shape[0]``.

    Args:
        input_ids: (seq_len,) tensor containing one ``media_token_id`` per image.
        attention_mask: (seq_len,) tensor aligned with *input_ids*.
        grid_thws: (N, 3) tensor with ``[t, h, w]`` for each of the N images.
        media_token_id: Token ID used as the image placeholder.
        merge_kernel_size: Vision tower's patch merge kernel, default (2, 2).

    Returns:
        expanded_input_ids: Input IDs with each placeholder expanded to its patch count.
        expanded_attention_mask: Attention mask expanded accordingly.

    Raises:
        ValueError: When the number of placeholders does not match ``grid_thws.shape[0]``.
    """
    merge_h, merge_w = merge_kernel_size

    placeholder_positions = (input_ids == media_token_id).nonzero(as_tuple=True)[0]
    if len(placeholder_positions) == 0:
        return input_ids, attention_mask

    n_placeholders = len(placeholder_positions)
    n_images = grid_thws.shape[0]
    if n_placeholders != n_images:
        raise ValueError(
            f"_expand_image_tokens: found {n_placeholders} placeholder(s) in input_ids "
            f"but grid_thws has {n_images} row(s). They must match."
        )

    # Rebuild input_ids and attention_mask by expanding each placeholder in order.
    # We iterate through the placeholder positions; between consecutive placeholders
    # we copy the original tokens unchanged.
    pieces_ids: List[torch.Tensor] = []
    pieces_mask: List[torch.Tensor] = []
    cursor = 0
    for placeholder_pos, (_, h, w) in zip(placeholder_positions.tolist(), grid_thws.tolist()):
        n_tokens = (int(h) // merge_h) * (int(w) // merge_w)
        pieces_ids.append(input_ids[cursor:placeholder_pos])
        pieces_mask.append(attention_mask[cursor:placeholder_pos])
        pieces_ids.append(torch.full((n_tokens,), media_token_id, dtype=input_ids.dtype))
        pieces_mask.append(torch.ones(n_tokens, dtype=attention_mask.dtype))
        cursor = placeholder_pos + 1

    # Append tokens after the last placeholder
    pieces_ids.append(input_ids[cursor:])
    pieces_mask.append(attention_mask[cursor:])

    return torch.cat(pieces_ids), torch.cat(pieces_mask)


def kimi_k25_vl_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
    drop_overlong: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collate function for Kimi K2.5 VL processors with pre-expanded image tokens.

    For pipeline parallelism, this function:
    1. Processes each sample to get input_ids with 1 placeholder per image
    2. Pre-expands the placeholder to N tokens (N = (h//2)*(w//2) from grid_thws)
    3. Pads all sequences to fixed max_length
    This ensures the model forward pass doesn't change sequence length dynamically.
    """
    conversations = [example["conversation"] for example in examples]

    # Optionally drop overlong samples before processing
    if max_length is not None and drop_overlong:
        conversations, _kept = _drop_overlong_samples(conversations, processor, max_length)
        examples = [examples[i] for i in _kept]

    # Get media token ID
    media_token_id = getattr(processor, "media_placeholder_token_id", None)
    if media_token_id is None and hasattr(processor, "tokenizer"):
        media_token_id = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    if media_token_id is None:
        media_token_id = 163605  # Default for Kimi K2.5

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0

    # Process each sample individually, dropping any that exceed max_length
    # after token expansion.
    kept_conversations = []
    kept_examples = []
    all_expanded = []
    all_pixel_values = []
    all_grid_thws = []
    # Per-sample image counts, kept in lockstep with all_expanded so that
    # n_images_per_sample length matches batch_size downstream. Samples that
    # are text-only or whose image region was orphaned by truncation get 0.
    per_sample_image_count: List[int] = []

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
        if max_length is not None and not drop_overlong:
            processor_kwargs["truncation"] = True
            processor_kwargs["max_length"] = max_length

        sample_batch = processor(**processor_kwargs)

        input_ids = sample_batch["input_ids"][0]
        attention_mask = sample_batch["attention_mask"][0]

        # Pre-expand image tokens if we have grid_thws
        grid_thws = None
        if "grid_thws" in sample_batch and sample_batch["grid_thws"] is not None:
            grid_thws = sample_batch["grid_thws"]
            input_ids, attention_mask = _expand_image_tokens(input_ids, attention_mask, grid_thws, media_token_id)

        # Handle overlong samples after expansion
        if max_length is not None and input_ids.shape[0] > max_length:
            if drop_overlong:
                logger.warning(
                    "Dropping expanded sample with %d tokens (max_length=%d).",
                    input_ids.shape[0],
                    max_length,
                )
                continue
            else:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]

        kept_conversations.append(conversation)
        kept_examples.append(examples[i])

        # Only include image data if all expanded image tokens survived truncation.
        # Partial truncation into image regions would cause a mismatch in the model forward.
        sample_image_count = 0
        if grid_thws is not None:
            merge_h, merge_w = _DEFAULT_MERGE_KERNEL
            expected_image_tokens = sum(int((h // merge_h) * (w // merge_w)) for _, h, w in grid_thws.tolist())
            actual_image_tokens = (input_ids == media_token_id).sum().item()
            if actual_image_tokens == expected_image_tokens:
                all_grid_thws.append(grid_thws)
                sample_image_count = int(grid_thws.shape[0])
                if "pixel_values" in sample_batch:
                    all_pixel_values.append(sample_batch["pixel_values"])
            else:
                # Replace orphaned image tokens so the model doesn't look for missing pixel data
                input_ids = input_ids.masked_fill(input_ids == media_token_id, pad_token_id)
        elif "pixel_values" in sample_batch:
            all_pixel_values.append(sample_batch["pixel_values"])

        all_expanded.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
        per_sample_image_count.append(sample_image_count)

    if not all_expanded:
        raise ValueError(
            f"All samples in batch exceed max_length={max_length} after expansion. "
            "Consider increasing max_length or filtering your dataset."
        )
    conversations = kept_conversations

    # Determine target length for padding
    expanded_lens = [b["input_ids"].shape[0] for b in all_expanded]
    batch_max = max(expanded_lens)

    if max_length is not None:
        target_len = max_length
    else:
        target_len = batch_max

    # Pad or truncate to target_len
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
        # Per-sample image counts for PP chunking. Length must equal batch_size,
        # so include zeros for text-only samples and for samples whose image
        # region was orphaned by truncation.
        result["n_images_per_sample"] = torch.tensor(per_sample_image_count, dtype=torch.long)

    # Build labels
    labels = build_labels_from_template(
        result["input_ids"],
        conversations,
        processor,
    )
    result["labels"] = labels[:, 1:]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(kept_examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(result, processor, fake_indices)

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

    labels = build_labels_from_template(
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

    # Per-sample image counts for PP chunking (max 1 image per sample)
    image_counts = [1 if img is not None else 0 for img in images]
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)

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


def _extract_image_config(processor):
    """Extract image processing config from processor for token estimation."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    patch_size = getattr(ip, "patch_size", 14)
    merge_size = getattr(ip, "merge_size", 2)
    # Qwen2VL/Qwen3VL store min/max_pixels as direct attributes;
    # fall back to ip.size dict with both Qwen-style and HF-style keys.
    size = getattr(ip, "size", {}) or {}
    min_pixels = getattr(ip, "min_pixels", None) or size.get("min_pixels") or size.get("shortest_edge") or 56 * 56
    max_pixels = (
        getattr(ip, "max_pixels", None) or size.get("max_pixels") or size.get("longest_edge") or 14 * 14 * 4 * 1280
    )
    return {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "factor": patch_size * merge_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }


def _estimate_media_tokens(conversation, processor):
    """Estimate expanded media token count from image/video dimensions.

    Returns total extra tokens beyond the single-placeholder-per-media count
    that tokenization produces.  Only images with known dimensions (PIL Image
    objects or loadable paths) are estimated; unknown media items contribute 0
    extra tokens (the placeholder is still counted in the base tokenization).
    """
    image_cfg = _extract_image_config(processor)
    if image_cfg is None:
        return 0

    extra = 0
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image" or "image" not in item:
                continue

            img = item["image"]
            try:
                if isinstance(img, PILImage.Image):
                    width, height = img.size
                elif isinstance(img, str):
                    with PILImage.open(img) as im:
                        width, height = im.size
                else:
                    continue
            except Exception:
                continue

            resized_h, resized_w = _smart_resize_image(
                height,
                width,
                factor=image_cfg["factor"],
                min_pixels=image_cfg["min_pixels"],
                max_pixels=image_cfg["max_pixels"],
            )
            merge_length = image_cfg["merge_size"] ** 2
            image_seq_len = (
                (resized_h // image_cfg["patch_size"]) * (resized_w // image_cfg["patch_size"]) // merge_length
            )
            extra += image_seq_len - 1  # -1: placeholder already counted in base tokenization

    return extra


def _drop_overlong_samples(conversations, processor, max_length):
    """Drop conversations whose estimated token count exceeds *max_length*.

    Returns ``(filtered_conversations, kept_indices)`` where *kept_indices*
    are the original positions that survived filtering.  Raises ``ValueError``
    when every sample in the batch is dropped (caught by ``robust_collate``
    which re-samples).
    """
    if max_length is None:
        return conversations, list(range(len(conversations)))

    tokenizer = getattr(processor, "tokenizer", processor)
    filtered = []
    kept_indices = []

    for i, conv in enumerate(conversations):
        try:
            text = processor.apply_chat_template([conv], tokenize=False)
            if isinstance(text, list):
                text = text[0]
            base_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            extra_tokens = _estimate_media_tokens(conv, processor)
            total = base_tokens + extra_tokens
            if total > max_length:
                logger.warning(
                    "Dropping sample with estimated %d tokens (max_length=%d).",
                    total,
                    max_length,
                )
                continue
        except Exception:
            pass  # estimation failed → keep the sample
        filtered.append(conv)
        kept_indices.append(i)

    if not filtered:
        raise ValueError(
            f"All {len(conversations)} samples in batch exceed max_length={max_length}. "
            "Consider increasing max_length or filtering your dataset."
        )

    return filtered, kept_indices


def default_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
    drop_overlong: bool = False,
    _post_tokenize_hook=None,
) -> Dict[str, torch.Tensor]:
    """Default collate function for multimodal VLM datasets.

    Args:
        _post_tokenize_hook: Optional callable ``(batch, processor) -> batch``
            invoked right after ``apply_chat_template`` and before
            ``build_labels``.  Used by model-specific collate wrappers
            (e.g. Gemma4 thinking-channel injection) to transform the
            tokenized batch and the prefix tokens without duplicating the rest of the pipeline.
    """
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = _ensure_rgb([example["conversation"] for example in examples])

    # Optionally drop overlong samples before processing
    if max_length is not None and drop_overlong:
        conversations, kept = _drop_overlong_samples(conversations, processor, max_length)
        examples = [examples[i] for i in kept]

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
        if drop_overlong:
            processor_kwargs["truncation"] = False  # Pre-filtering guarantees samples fit
    batch = processor.apply_chat_template(conversations, **processor_kwargs)

    if _post_tokenize_hook is not None:
        batch = _post_tokenize_hook(batch, processor)

    # NOTE: Do NOT generate fallback position_ids here. Models with mrope
    # (e.g. Qwen3-VL) need 3D position_ids [3, batch, seq_len] generated by
    # get_rope_index(input_ids, image_grid_thw, video_grid_thw) inside the
    # model forward. Passing simple sequential position_ids would bypass that
    # and degrade mrope to 1D positional encoding.

    # Convert pixel values to bfloat16 (images and/or videos)
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in batch:
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)

    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in list(batch.keys()):
        v = batch[key]
        if isinstance(v, torch.Tensor) and v.shape == input_shape and key != "labels":
            batch[key] = v[:, :-1]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    # Per-sample media counts for PP chunking
    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch


def pad_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for pre-tokenized samples (from :class:`PreTokenizedDatasetWrapper`).

    Each *example* is expected to carry at least ``input_ids``, ``attention_mask``,
    and ``labels`` as 1-D tensors, plus optional media tensors (``pixel_values``,
    ``image_grid_thw``, ``pixel_values_videos``, ``video_grid_thw``).

    Fake image injection and vision-token masking are handled per-sample in
    :class:`PreTokenizedDatasetWrapper.__getitem__`, so this function only
    pads, stacks, and concatenates.

    The function:

    1. Pads all sequence tensors to the same length (either *max_length* or the
       longest sequence in the batch).
    2. Concatenates media tensors across the batch.
    3. Applies the standard autoregressive shift (``labels = labels[:, 1:]``,
       inputs truncated by one token).
    """
    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------
    seq_lengths = [ex["input_ids"].shape[0] for ex in examples]
    pad_to = max_length if max_length is not None else max(seq_lengths)

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for ex in examples:
        ids = ex["input_ids"]
        mask = ex["attention_mask"]
        labs = ex["labels"]
        pad_len = pad_to - ids.shape[0]

        if pad_len > 0:
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)]))
            padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
            padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)]))
        else:
            padded_input_ids.append(ids[:pad_to])
            padded_attention_mask.append(mask[:pad_to])
            padded_labels.append(labs[:pad_to])

    batch: Dict[str, torch.Tensor] = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }

    # Pad sequence-length tensors that mirror input_ids (e.g. mm_token_type_ids)
    for seq_key in ("mm_token_type_ids",):
        if any(seq_key in ex for ex in examples):
            padded = []
            for ex in examples:
                t = ex.get(seq_key)
                if t is None:
                    padded.append(torch.zeros(pad_to, dtype=torch.long))
                else:
                    t = t[0] if t.dim() == 2 else t
                    p = pad_to - t.shape[0]
                    padded.append(torch.cat([t, torch.zeros(p, dtype=t.dtype)]) if p > 0 else t[:pad_to])
            batch[seq_key] = torch.stack(padded)

    # ------------------------------------------------------------------
    # Autoregressive shift: labels[t] predicts input_ids[t+1]
    # ------------------------------------------------------------------
    labels_tensor = torch.stack(padded_labels)
    batch["labels"] = labels_tensor[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    # ------------------------------------------------------------------
    # Concatenate media tensors across samples & compute per-sample counts
    # ------------------------------------------------------------------
    for key in ("pixel_values", "pixel_values_videos"):
        tensors = [ex[key] for ex in examples if key in ex and ex[key] is not None]
        if tensors:
            batch[key] = torch.cat(tensors, dim=0).to(torch.bfloat16)

    # Per-sample image counts from image_grid_thw shapes (before concat)
    image_grid_per_sample = [
        ex["image_grid_thw"] for ex in examples if "image_grid_thw" in ex and ex["image_grid_thw"] is not None
    ]
    if image_grid_per_sample:
        # Each tensor is [n_images_in_sample, 3]; build per-sample counts for all samples
        image_counts = []
        for ex in examples:
            if "image_grid_thw" in ex and ex["image_grid_thw"] is not None:
                image_counts.append(ex["image_grid_thw"].shape[0])
            else:
                image_counts.append(0)
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)

    # Per-sample video counts from video_grid_thw shapes (before concat)
    video_grid_per_sample = [
        ex["video_grid_thw"] for ex in examples if "video_grid_thw" in ex and ex["video_grid_thw"] is not None
    ]
    if video_grid_per_sample:
        video_counts = []
        for ex in examples:
            if "video_grid_thw" in ex and ex["video_grid_thw"] is not None:
                video_counts.append(ex["video_grid_thw"].shape[0])
            else:
                video_counts.append(0)
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    for key in ("image_grid_thw", "video_grid_thw", "image_position_ids"):
        tensors = [ex[key] for ex in examples if key in ex and ex[key] is not None]
        if tensors:
            batch[key] = torch.cat(tensors, dim=0)
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

    def _get_mm_token_type_ids(item):
        v = item.get("mm_token_type_ids")
        return v if v is not None else torch.zeros(0, dtype=torch.long)

    mm_token_type_ids = torch.stack([_pad_1d(_get_mm_token_type_ids(x), 0, max_len) for x in batch])

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
        "mm_token_type_ids": mm_token_type_ids,
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

    for key in ("image_grid_thw", "image_position_ids", "video_grid_thw", "second_per_grid_ts"):
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


def nemotron_omni_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
    max_video_frames: int = 8,
) -> Dict[str, torch.Tensor]:
    """Collate for NemotronOmni (image / video / audio).

    Defers ``<image>``/``<video>``/``<audio>`` placeholder expansion to the processor;
    the collate only gathers media into processor kwargs, pads, builds labels, and
    stacks tensors.
    """
    import numpy as np
    from transformers.video_utils import VideoMetadata

    from nemo_automodel.components.datasets.vlm.utils import _read_video_frames

    conversations = _ensure_rgb([example["conversation"] for example in examples])
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", "<image>")
    video_token = getattr(processor, "video_token", "<video>")

    sample_audio: Dict[int, np.ndarray] = {}
    target_sr = getattr(processor, "audio_sampling_rate", 16000)
    for idx, example in enumerate(examples):
        audio = example.get("audio")
        if audio is None:
            continue
        if isinstance(audio, tuple) and len(audio) == 2:
            waveform, sr = audio
            waveform = np.asarray(waveform, dtype=np.float32)
            if sr != target_sr:
                n_out = int(round(len(waveform) * target_sr / sr))
                waveform = np.interp(
                    np.linspace(0, len(waveform), n_out, endpoint=False),
                    np.arange(len(waveform)),
                    waveform,
                ).astype(np.float32)
        else:
            waveform = np.asarray(audio, dtype=np.float32)
        sample_audio[idx] = waveform

    all_images: List[List[Any]] = []
    all_videos: List[Optional[Tuple[List[Any], Optional[VideoMetadata]]]] = []
    texts: List[str] = []

    for conversation in conversations:
        conv_images: List[Any] = []
        conv_video: Optional[Tuple[List[Any], Optional[VideoMetadata]]] = None
        text_conversation = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type")
                    if t == "image":
                        img = item.get("image")
                        if img is not None:
                            conv_images.append(img)
                            text_parts.append(image_token)
                    elif t == "video":
                        vid = item.get("video")
                        if vid is None:
                            continue
                        if isinstance(vid, str):
                            import decord

                            decord.bridge.set_bridge("native")
                            total = len(decord.VideoReader(vid))
                            # Even count required by ``video_temporal_patch_dim=2``.
                            n = min(max_video_frames, total)
                            if n % 2 != 0:
                                n = max(2, n - 1)
                            frame_idx = torch.linspace(0, total - 1, n).round().long().tolist()
                            frames, video_fps, indices = _read_video_frames(
                                vid,
                                processor=processor,
                                frame_indices=frame_idx,
                                return_metadata=True,
                            )
                            metadata = VideoMetadata(
                                total_num_frames=len(frames),
                                fps=video_fps,
                                frames_indices=[int(i) for i in indices],
                            )
                        elif isinstance(vid, list):
                            frames = vid
                            metadata = item.get("metadata")
                        else:
                            raise ValueError(f"Unsupported video type: {type(vid)}")
                        if conv_video is not None:
                            raise NotImplementedError("Only 1 video per sample is supported")
                        conv_video = (frames, metadata)
                        text_parts.append(video_token)
                    elif t == "text":
                        text_parts.append(item.get("text", ""))
                text_conversation.append(
                    {
                        "role": message["role"],
                        "content": "".join(text_parts),
                    }
                )
            else:
                text_conversation.append(message)

        all_images.append(conv_images)
        all_videos.append(conv_video)
        text = tokenizer.apply_chat_template(text_conversation, tokenize=False)
        texts.append(text)

    all_input_ids: List[torch.Tensor] = []
    all_attention_masks: List[torch.Tensor] = []
    sample_pixel_values: Dict[int, Any] = {}
    sample_pixel_values_videos: Dict[int, Any] = {}
    sample_sound_clips: Dict[int, Any] = {}

    for i, (text, images, video) in enumerate(zip(texts, all_images, all_videos)):
        kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        if video is not None:
            frames, metadata = video
            kwargs["videos"] = frames
            if metadata is not None:
                kwargs["videos_kwargs"] = {"video_metadata": metadata}
        if i in sample_audio:
            kwargs["audio"] = [sample_audio[i]]

        batch_i = processor(**kwargs)
        all_input_ids.append(batch_i["input_ids"][0])
        all_attention_masks.append(batch_i["attention_mask"][0])

        if batch_i.get("pixel_values") is not None:
            sample_pixel_values[i] = batch_i["pixel_values"]
        if batch_i.get("pixel_values_videos") is not None:
            sample_pixel_values_videos[i] = batch_i["pixel_values_videos"]
        if batch_i.get("sound_clips") is not None:
            sample_sound_clips[i] = batch_i["sound_clips"]

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    seq_lengths = [ids.shape[0] for ids in all_input_ids]
    target_len = max_length if max_length is not None else max(seq_lengths)

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

    result: Dict[str, Any] = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
    }

    def _flatten_pv(per_sample_dict: Dict[int, Any]) -> List[torch.Tensor]:
        flat: List[torch.Tensor] = []
        for i in range(len(padded_input_ids)):
            pv = per_sample_dict.get(i)
            if pv is None:
                continue
            if isinstance(pv, torch.Tensor):
                if pv.dim() == 4:
                    for k in range(pv.shape[0]):
                        flat.append(pv[k])
                else:
                    flat.append(pv)
            else:
                flat.extend(pv)
        return flat

    flat_images = _flatten_pv(sample_pixel_values)
    if flat_images:
        if all(t.shape == flat_images[0].shape for t in flat_images):
            result["pixel_values"] = torch.stack(flat_images).to(torch.bfloat16)
        else:
            # Variable shapes — extract_feature handles list input.
            result["pixel_values"] = [t.to(torch.bfloat16) for t in flat_images]
        result["image_flags"] = torch.ones(len(flat_images), 1, dtype=torch.long)

    flat_video_frames = _flatten_pv(sample_pixel_values_videos)
    if flat_video_frames:
        if all(t.shape == flat_video_frames[0].shape for t in flat_video_frames):
            result["pixel_values_videos"] = torch.stack(flat_video_frames).to(torch.bfloat16)
        else:
            result["pixel_values_videos"] = [t.to(torch.bfloat16) for t in flat_video_frames]

    if sample_sound_clips:
        all_waves_np: List[np.ndarray] = []
        for i in range(len(padded_input_ids)):
            clips = sample_sound_clips.get(i)
            if clips is None:
                continue
            wave = clips[0] if isinstance(clips, list) else clips
            if isinstance(wave, torch.Tensor):
                wave = wave.cpu().numpy()
            wave = np.asarray(wave, dtype=np.float32).squeeze()
            all_waves_np.append(wave)
        if all_waves_np:
            fe = getattr(processor, "_sound_feature_extractor", None)
            if fe is None:
                from transformers import ParakeetFeatureExtractor

                fe = ParakeetFeatureExtractor(sampling_rate=target_sr, feature_size=128)
                try:
                    processor._sound_feature_extractor = fe
                except Exception:
                    pass
            audio_inputs = fe(all_waves_np, sampling_rate=target_sr, return_tensors="pt")
            result["sound_features"] = audio_inputs["input_features"].to(torch.float32)
            if "attention_mask" in audio_inputs:
                result["sound_attention_mask"] = audio_inputs["attention_mask"].to(torch.long)

    labels = build_labels(result["input_ids"], conversations, processor)
    result["labels"] = labels[:, 1:]

    input_shape = result["input_ids"].shape
    for key, value in list(result.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            result[key] = value[:, :-1]

    return result


# ---------------------------------------------------------------------------
# Gemma4 thinking-channel prefix injection
# ---------------------------------------------------------------------------
_GEMMA4_MODEL_TURN = "<|turn>model\n"
_GEMMA4_THINKING_PREFIX = "<|channel>thought\n<channel|>"


def _inject_thinking_prefix_tokens(
    batch: Dict[str, torch.Tensor],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """Insert ``<|channel>thought\\n<channel|>`` tokens after every ``<|turn>model\\n`` marker.

    Modifies ``input_ids``, ``attention_mask``, and ``mm_token_type_ids``
    (if present).  Additionally, any other 2-D integer tensor whose second
    dimension matches ``input_ids`` is extended with zeros so that sequence
    lengths stay consistent (this ismore of future-proofing)
    """
    marker_ids = tokenizer.encode(_GEMMA4_MODEL_TURN, add_special_tokens=False)
    prefix_ids = tokenizer.encode(_GEMMA4_THINKING_PREFIX, add_special_tokens=False)

    if not prefix_ids or not marker_ids:
        return batch

    marker_len = len(marker_ids)
    prefix_len = len(prefix_ids)
    marker_t = torch.tensor(marker_ids, dtype=batch["input_ids"].dtype)
    prefix_t = torch.tensor(prefix_ids, dtype=batch["input_ids"].dtype)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    seq_keys = ["input_ids", "attention_mask"]
    fill_defaults: Dict[str, int] = {"input_ids": pad_id, "attention_mask": 0}
    inject_defaults: Dict[str, Any] = {"input_ids": prefix_t, "attention_mask": 1}

    if "mm_token_type_ids" in batch:
        seq_keys.append("mm_token_type_ids")
        fill_defaults["mm_token_type_ids"] = 0
        inject_defaults["mm_token_type_ids"] = 0

    B = batch["input_ids"].size(0)
    new_seqs: Dict[str, List[torch.Tensor]] = {k: [] for k in seq_keys}

    for b in range(B):
        seq = batch["input_ids"][b]
        dev = seq.device

        insert_after: List[int] = []
        i = 0
        while i <= len(seq) - marker_len:
            if torch.all(seq[i : i + marker_len] == marker_t.to(dev)):
                insert_after.append(i + marker_len)
                i += marker_len
            else:
                i += 1

        if not insert_after:
            for k in seq_keys:
                new_seqs[k].append(batch[k][b])
            continue

        for k in seq_keys:
            src = batch[k][b]
            chunks: List[torch.Tensor] = []
            prev = 0
            for pos in insert_after:
                chunks.append(src[prev:pos])
                if k == "input_ids":
                    chunks.append(prefix_t.to(dev))
                else:
                    val = inject_defaults[k]
                    chunks.append(torch.full((prefix_len,), val, dtype=src.dtype, device=dev))
                prev = pos
            chunks.append(src[prev:])
            new_seqs[k].append(torch.cat(chunks))

    max_new_len = max(s.size(0) for s in new_seqs["input_ids"])
    for k in seq_keys:
        fill = fill_defaults[k]
        padded = torch.full((B, max_new_len), fill, dtype=batch[k].dtype, device=batch[k].device)
        for i, t in enumerate(new_seqs[k]):
            L = t.size(0)
            padded[i, :L] = t[:L]
        batch[k] = padded

    return batch


def gemma4_inject_thinking_prefix(
    batch: Dict[str, torch.Tensor],
    processor,
) -> Dict[str, torch.Tensor]:
    """Inject Gemma4's thinking-channel prefix after every assistant turn marker.

    Gemma4 31B / 26B-A4B MoE instruction-tuned models always emit a thinking-
    channel prefix before the actual response.  When this prefix is absent from
    training sequences the model predicts ``<|channel>`` but the label says
    answer text, inflating initial loss to ~9.  Injecting the prefix (masked
    as -100 in labels) lets the model see its expected pattern and brings
    initial loss down to ~3.

    Safe no-op for non-Gemma4 tokenizers.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    return _inject_thinking_prefix_tokens(batch, tokenizer)


def gemma4_prefix_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for Gemma4 models with thinking-channel prefix.

    Wraps ``default_collate_fn`` and injects ``<|channel>thought\\n<channel|>``
    after every ``<|turn>model\\n`` marker before labels are built.  The injected
    tokens are automatically masked to -100 by ``build_labels_from_template``
    (which only unmasks tokens inside assistant turns), so the model sees its
    expected thinking prefix without being penalised for it.
    """

    def _inject(batch, proc):
        batch = gemma4_inject_thinking_prefix(batch, proc)
        if max_length is not None and batch["input_ids"].size(1) > max_length:
            for key in list(batch.keys()):
                v = batch[key]
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) > max_length and key != "pixel_values":
                    batch[key] = v[:, :max_length]
        return batch

    return default_collate_fn(examples, processor, max_length, _post_tokenize_hook=_inject)


def llava_onevision_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
) -> Dict[str, torch.Tensor]:
    """Collate function for LLaVA-OneVision-1.5 processors.

    Handles image and video inputs using the Qwen-style chat template and
    qwen_vl_utils for vision info processing.
    """
    conversations = [example["conversation"] for example in examples]

    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]

    images, videos = _extract_media_from_conversations(conversations)

    batch = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )

    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    # Mask fake vision tokens for samples that had fake images injected at dataset level.
    fake_indices = [i for i, ex in enumerate(examples) if ex.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    # Per-sample media counts for PP chunking
    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(c > 0 for c in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(c > 0 for c in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "KimiVLProcessor": kimi_vl_collate_fn,
    "KimiK25Processor": kimi_k25_vl_collate_fn,
    "NemotronParseProcessor": nemotron_parse_collate_fn,
    "NemotronH_Nano_Omni_Reasoning_V3Processor": nemotron_omni_collate_fn,
    "LlavaOneVisionProcessor": llava_onevision_collate_fn,
    "default": default_collate_fn,
}
