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

from PIL import Image as PILImage

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


def _get_vision_token_ids(processor):
    """Collect vision token IDs from a processor/tokenizer."""
    vision_token_ids = set()

    for attr in ("image_token_id", "video_token_id"):
        tid = getattr(processor, attr, None)
        if tid is not None:
            vision_token_ids.add(tid)

    tokenizer = getattr(processor, "tokenizer", processor)
    for token in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", None):
                vision_token_ids.add(tid)
        except Exception:
            pass

    return vision_token_ids


def mask_fake_vision_tokens_single(sample_dict, processor):
    """Mask vision tokens in a single pre-tokenized sample (1D tensors).

    Sets ``attention_mask = 0`` for every vision token in *sample_dict*.
    This is used at ``__getitem__`` time for pre-tokenized datasets.
    """
    vision_token_ids = _get_vision_token_ids(processor)
    if not vision_token_ids or "attention_mask" not in sample_dict:
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
    vision_token_ids = _get_vision_token_ids(processor)
    if not vision_token_ids or "attention_mask" not in batch or not sample_indices:
        return

    for idx in sample_indices:
        input_ids_i = batch["input_ids"][idx]
        for tid in vision_token_ids:
            mask = input_ids_i == tid
            batch["attention_mask"][idx][mask] = 0
