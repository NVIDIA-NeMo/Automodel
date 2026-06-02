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
"""Modality-neutral label/marker utilities shared across dataset collate functions.

These helpers build training labels (assistant-turn masking) from chat-template
token markers. They are deliberately free of any image/audio/video specifics so
that both the VLM and audio collate modules can import them without creating a
cross-package dependency between those two modalities.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def default_stop_tokens(processor) -> Iterable[str]:
    """Return default generation stop tokens for a processor tokenizer."""
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    candidates = [
        "<end_of_turn>",
        "<|im_end|>",
        "<|eot_id|>",
    ]
    if eos_token is not None:
        candidates.append(eos_token)
    return tuple(candidates)


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
        "Qwen2_5OmniProcessor",
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
