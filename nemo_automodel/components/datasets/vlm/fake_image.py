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

"""Fake image injection helpers for FSDP / DeepSpeed Zero3.

When a batch contains no images or videos, the visual encoder is not called
during the model forward pass.  In FSDP / DeepSpeed Zero3 every parameter
must participate in the collective all-gather / reduce-scatter; skipping the
visual encoder causes the training to hang.

The fix mirrors LLaMA-Factory's approach: inject a tiny (56x56) white image
into pure-text samples.  The corresponding vision tokens get
``attention_mask = 0`` so they are invisible to attention and
``labels = -100`` (automatic, because the fake image lives in a *user*
message, never an assistant turn).  This guarantees model correctness while
keeping the visual encoder active.
"""

import copy
import logging

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Constant 56x56 white PIL image used as the fake placeholder.
_FAKE_IMAGE = PILImage.new("RGB", (56, 56), (255, 255, 255))


def _conversation_has_media(conversation):
    """Return True if *conversation* (a single list of messages) contains an image or video."""
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image", "video"):
                    return True
                # Also detect items that carry an "image" or "video" key
                # without an explicit "type" field (common in some datasets).
                if "image" in item or "video" in item:
                    return True
    return False


def _batch_has_media(conversations):
    """Return True if any conversation in *conversations* contains an image or video."""
    for conv in conversations:
        if _conversation_has_media(conv):
            return True
    return False


def inject_fake_image_into_conversation(conversation):
    """Inject a fake image into a single conversation's first user message.

    Returns a deep-copied conversation so the original is never mutated.
    """
    conversation = copy.deepcopy(conversation)
    for message in conversation:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                content.insert(0, {"type": "image", "image": _FAKE_IMAGE})
            elif isinstance(content, str):
                message["content"] = [
                    {"type": "image", "image": _FAKE_IMAGE},
                    {"type": "text", "text": content},
                ]
            else:
                message["content"] = [{"type": "image", "image": _FAKE_IMAGE}]
            return conversation
    # No user message found - prepend one with just the fake image.
    conversation.insert(
        0,
        {
            "role": "user",
            "content": [{"type": "image", "image": _FAKE_IMAGE}],
        },
    )
    return conversation


# Attribute names that VLM processors / configs use to expose vision token IDs.
# Different model families use different names; we scan all of them.
_VISION_TOKEN_ID_ATTRS = (
    "image_token_id",  # Qwen2/3-VL, generic HF VLMs (also Gemma3 alias)
    "video_token_id",  # Qwen2/3-VL
    "image_token_index",  # LLaVA family, Pixtral/Mistral3, Gemma3
    "video_token_index",  # LLaVA-OneVision / NeXT-Video
    "media_placeholder_token_id",  # Kimi K2.5, KimiVL
    "vision_start_token_id",  # Qwen
    "vision_end_token_id",  # Qwen
    "vision_token_id",  # Qwen
    "image_break_token_id",  # Pixtral / Mistral3
    "image_end_token_id",  # Pixtral / Mistral3
    "boi_token_index",  # Gemma3 (begin-of-image)
    "eoi_token_index",  # Gemma3 (end-of-image)
    "boi_token_id",  # Gemma3 alias
    "eoi_token_id",  # Gemma3 alias
)

# Explicit vision token strings used across VLM families.
_VISION_TOKEN_STRINGS = (
    "<|vision_start|>",
    "<|vision_end|>",  # Qwen
    "<|image_pad|>",
    "<|video_pad|>",  # Qwen
    "<|media_start|>",
    "<|media_content|>",  # Kimi K2.5 / KimiVL
    "<|media_end|>",
    "<|media_pad|>",  # Kimi K2.5 / KimiVL
    "<image>",
    "<video>",  # LLaVA family
    "<|image|>",
    "<|video|>",  # Some HF chat templates
    "[IMG]",
    "[IMG_END]",
    "[IMG_BREAK]",  # Pixtral / Mistral3
)

# Substrings used to fuzzy-match additional vision tokens from the tokenizer's
# added_tokens_decoder.  Kept short enough to avoid colliding with non-vision
# tokens like ``<|imagine|>`` (no such real token, but be defensive).
_VISION_TOKEN_KEYWORDS = ("image", "video", "media", "vision", "img_pad", "vid_pad")


def _scan_attrs(source, attr_names):
    """Yield integer token IDs found via ``getattr`` on *source*."""
    if source is None:
        return
    for attr in attr_names:
        tid = getattr(source, attr, None)
        if isinstance(tid, int):
            yield tid


def _get_vision_token_ids(processor):
    """Collect vision token IDs from a processor / tokenizer / config.

    Walks three sources to be robust across VLM families:
    1. Known attribute names on the processor *and* its ``config`` (Gemma4,
       LLaVA put the IDs on the config rather than the processor).
    2. A curated list of vision token strings looked up via
       ``tokenizer.convert_tokens_to_ids``.
    3. A keyword-based fuzzy scan of ``tokenizer.added_tokens_decoder`` so
       custom or future VLMs are picked up automatically.
    """
    vision_token_ids: set[int] = set()

    config = getattr(processor, "config", None)
    for tid in _scan_attrs(processor, _VISION_TOKEN_ID_ATTRS):
        vision_token_ids.add(tid)
    for tid in _scan_attrs(config, _VISION_TOKEN_ID_ATTRS):
        vision_token_ids.add(tid)

    tokenizer = getattr(processor, "tokenizer", processor)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for token in _VISION_TOKEN_STRINGS:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(tid, int) and tid != unk_id:
            vision_token_ids.add(tid)

    added = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added, dict):
        for tid, tok in added.items():
            try:
                tok_str = getattr(tok, "content", str(tok)).lower()
            except Exception:
                continue
            if any(kw in tok_str for kw in _VISION_TOKEN_KEYWORDS):
                try:
                    vision_token_ids.add(int(tid))
                except (TypeError, ValueError):
                    continue

    return vision_token_ids


_warned_unknown_processors: set[str] = set()


def _warn_no_vision_tokens(processor) -> None:
    """Log a one-time warning when a processor exposes no recognizable vision tokens.

    Without this warning a fake-image injection silently leaves the vision
    tokens visible to attention, which can degrade training quality without
    any other observable symptom.
    """
    key = type(processor).__name__
    if key in _warned_unknown_processors:
        return
    _warned_unknown_processors.add(key)
    logger.warning(
        "fake_image: could not detect any vision token IDs from processor %s. "
        "Fake-image vision tokens will NOT be masked from attention. "
        "If this model uses a non-standard vision token, extend "
        "_VISION_TOKEN_ID_ATTRS / _VISION_TOKEN_STRINGS in fake_image.py.",
        key,
    )


def mask_fake_vision_tokens_single(sample_dict, processor):
    """Mask vision tokens in a single pre-tokenized sample (1D tensors).

    Sets ``attention_mask = 0`` for every vision token in *sample_dict*.
    This is used at ``__getitem__`` time for pre-tokenized datasets.
    """
    if "attention_mask" not in sample_dict:
        return
    vision_token_ids = _get_vision_token_ids(processor)
    if not vision_token_ids:
        _warn_no_vision_tokens(processor)
        return

    input_ids = sample_dict["input_ids"]
    for tid in vision_token_ids:
        mask = input_ids == tid
        sample_dict["attention_mask"][mask] = 0


def mask_fake_vision_tokens_batch(batch, processor, sample_indices):
    """Mask vision tokens in specified batch samples (2D tensors).

    Sets ``attention_mask = 0`` for every vision token in the given
    *sample_indices* of the batch.
    """
    if "attention_mask" not in batch or not sample_indices:
        return
    vision_token_ids = _get_vision_token_ids(processor)
    if not vision_token_ids:
        _warn_no_vision_tokens(processor)
        return

    for idx in sample_indices:
        input_ids_i = batch["input_ids"][idx]
        for tid in vision_token_ids:
            mask = input_ids_i == tid
            batch["attention_mask"][idx][mask] = 0
