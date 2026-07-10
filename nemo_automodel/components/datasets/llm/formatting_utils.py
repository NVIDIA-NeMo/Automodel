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

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def _resolve_chat_template(chat_template: Optional[str]) -> Optional[str]:
    """Resolve a chat template string that may be a file path.

    If *chat_template* points to an existing file, its contents are returned.
    If opening it as a file fails and the string contains Jinja-like characters
    (``{``, ``}``, or newlines) it is treated as a literal template.  Otherwise
    a :class:`ValueError` is raised so the caller knows the path was invalid.

    Args:
        chat_template: A Jinja template string or path to a template file.

    Returns:
        The resolved template string, or ``None`` when the input is ``None``.
    """
    if chat_template is None:
        return None

    if "{%" in chat_template or "{{" in chat_template:
        return chat_template

    p = Path(chat_template)
    if p.exists():
        content = p.read_text(encoding="utf-8")
        try:
            content = json.loads(content)["chat_template"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return content
    return chat_template


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")


def _tokenized_chat_length(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, str]],
    *,
    tools: Optional[List[Dict]] = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: Optional[int] = None,
) -> int:
    """Return the tokenized chat length for a message prefix without padding."""
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=truncation,
        max_length=seq_length,
    )
    return len(tokenized_chat.get("input_ids", []))


def _tokenize_chat(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict]] = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: Optional[int] = None,
) -> List[int]:
    """Tokenize chat messages without padding and return input ids."""
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=truncation,
        max_length=seq_length,
    )
    return tokenized_chat.get("input_ids", [])


def _maybe_shift_mask_for_left_padding(
    mask: List[int],
    tokenizer: "PreTrainedTokenizer",
    attention_mask: Optional[List[int]],
) -> List[int]:
    """Shift a token-level mask right when the tokenizer uses left padding.

    ``_build_multiturn_assistant_mask`` and ``_build_reasoning_mask`` compute
    span indices from **unpadded** (left-aligned) tokenizations.  When the
    tokenizer pads on the left, actual content is right-aligned in
    ``input_ids``, so the mask must be shifted right by the padding offset to
    keep positions aligned.

    For right-padding tokenizers (the majority) this is a no-op.
    """
    if getattr(tokenizer, "padding_side", "right") != "left":
        return mask
    if attention_mask is None:
        return mask
    pad_len = len(mask) - sum(attention_mask)
    if pad_len <= 0:
        return mask
    return [0] * pad_len + mask[: len(mask) - pad_len]


def _is_consistent_render_prefix(prefix_ids: "list[int]", reference_ids: "list[int]") -> bool:
    """Check that a prefix render matches the start of the full-conversation render.

    Locating spans by prefix lengths is only correct when
    ``render(messages[:k])`` reproduces the first tokens of ``render(messages)``.
    The final prefix token is excluded from the comparison: many templates close
    the rendered text with a token (an EOS or a trailing newline) that is
    replaced or merged once the next message is appended, and that one-token
    boundary fuzziness is already inherent to prefix-length arithmetic.
    """
    if not prefix_ids:
        return True
    if len(prefix_ids) > len(reference_ids):
        return False
    return prefix_ids[:-1] == reference_ids[: len(prefix_ids) - 1]


def _build_multiturn_assistant_mask(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, Any]],
    input_ids: List[int],
    *,
    tools: Optional[List[Dict]] = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: Optional[int] = None,
    unpadded_full_ids: "list[int] | None" = None,
) -> List[int]:
    """Build a fallback loss mask that supervises every assistant turn.

    Each assistant span is located by tokenizing the conversation prefixes
    before and after the turn, which is O(turns) ``apply_chat_template`` calls.
    Two reductions keep that from re-doing work:

    - ``unpadded_full_ids`` is the caller's already-known unpadded tokenization
      of the whole conversation. When the dialogue ends on an assistant turn
      its closing boundary is the full conversation, so passing it skips
      re-tokenizing the entire prefix (the single most expensive call in the
      loop). When omitted, the full conversation is tokenized once here.
    - Prefix lengths are memoized so a boundary shared by adjacent turns (a
      turn's end and the next turn's start) is tokenized at most once.

    Every tokenized prefix is validated against ``unpadded_full_ids`` (see
    :func:`_is_consistent_render_prefix`) and a :class:`ValueError` is raised
    on mismatch: a rewriting template leaves no way to place assistant spans
    correctly, so failing loudly is the only alternative to silently
    mislabeling supervised tokens.
    """
    assistant_mask = [0] * len(input_ids)
    found_assistant = False

    if unpadded_full_ids is None:
        unpadded_full_ids = _tokenize_chat(
            tokenizer,
            formatted_text,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )
    length_cache: Dict[int, int] = {len(formatted_text): len(unpadded_full_ids)}

    def prefix_length(k: int) -> int:
        if k not in length_cache:
            prefix_ids = _tokenize_chat(
                tokenizer,
                formatted_text[:k],
                tools=tools,
                truncation=truncation,
                seq_length=seq_length,
            )
            if not _is_consistent_render_prefix(prefix_ids, unpadded_full_ids):
                raise ValueError(
                    f"Cannot build an answer-only loss mask from conversation prefixes: rendering "
                    f"the first {k} message(s) alone does not reproduce a prefix of the fully "
                    f"rendered conversation. The chat template rewrites earlier turns based on "
                    f"later ones (for example, Qwen3's template drops <think> blocks from "
                    f"assistant turns that precede the last user turn), so prefix-length "
                    f"arithmetic would mislabel supervised tokens. Provide a chat template that "
                    f"wraps assistant turns in {{% generation %}}...{{% endgeneration %}} so the "
                    f"tokenizer returns the assistant mask directly."
                )
            length_cache[k] = len(prefix_ids)
        return length_cache[k]

    for idx, message in enumerate(formatted_text):
        if message["role"] != "assistant":
            continue

        found_assistant = True
        start = prefix_length(idx)
        end = prefix_length(idx + 1)
        for pos in range(min(start, len(assistant_mask)), min(end, len(assistant_mask))):
            assistant_mask[pos] = 1

    if not found_assistant:
        raise AssertionError("At least one assistant message is required when answer_only_loss_mask=True")

    return assistant_mask


def _masked_reasoning_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a message with reasoning_content removed."""
    masked = dict(message)
    masked["reasoning_content"] = ""
    return masked


def _find_reasoning_span(full_segment: List[int], masked_segment: List[int]) -> Optional[tuple[int, int]]:
    """Locate the contiguous token span attributable to reasoning content."""
    prefix_len = 0
    while (
        prefix_len < min(len(full_segment), len(masked_segment))
        and full_segment[prefix_len] == masked_segment[prefix_len]
    ):
        prefix_len += 1

    suffix_len = 0
    max_suffix = min(len(full_segment) - prefix_len, len(masked_segment) - prefix_len)
    while suffix_len < max_suffix and full_segment[-(suffix_len + 1)] == masked_segment[-(suffix_len + 1)]:
        suffix_len += 1

    reasoning_start = prefix_len
    reasoning_end = len(full_segment) - suffix_len
    if reasoning_end <= reasoning_start:
        return None

    return reasoning_start, reasoning_end


def _build_reasoning_mask(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, Any]],
    input_ids: List[int],
    *,
    tools: Optional[List[Dict]] = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: Optional[int] = None,
    unpadded_full_ids: "list[int] | None" = None,
) -> List[int]:
    """Build a token mask for reasoning_content spans inside assistant turns.

    Spans are located from conversation-prefix renders, so each prefix is
    validated against ``unpadded_full_ids`` (the unpadded tokenization of the
    whole conversation, tokenized here when omitted; see
    :func:`_is_consistent_render_prefix`). Turns whose prefix render diverges
    from the full render are skipped with a warning rather than an error,
    unlike :func:`_build_multiturn_assistant_mask`: a rewriting template does
    not render that turn's reasoning content in the full conversation at all,
    so there is no span to mask, whereas any prefix-based location would punch
    the hole at the wrong tokens.
    """
    reasoning_mask = [0] * len(input_ids)

    for idx, message in enumerate(formatted_text):
        if message.get("role") != "assistant" or not message.get("reasoning_content"):
            continue

        if unpadded_full_ids is None:
            unpadded_full_ids = _tokenize_chat(
                tokenizer,
                formatted_text,
                tools=tools,
                truncation=truncation,
                seq_length=seq_length,
            )

        prefix_ids = _tokenize_chat(
            tokenizer,
            formatted_text[:idx],
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )
        full_ids = _tokenize_chat(
            tokenizer,
            formatted_text[: idx + 1],
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )
        if not (
            _is_consistent_render_prefix(prefix_ids, unpadded_full_ids)
            and _is_consistent_render_prefix(full_ids, unpadded_full_ids)
        ):
            logger.warning(
                "Skipping reasoning_content masking for assistant message %s: the chat template "
                "renders the conversation prefix differently inside the full conversation, so the "
                "reasoning span cannot be located (and is likely not rendered in the full "
                "conversation at all).",
                idx,
            )
            continue

        masked_ids = _tokenize_chat(
            tokenizer,
            formatted_text[:idx] + [_masked_reasoning_message(message)],
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )

        start = len(prefix_ids)
        full_segment = full_ids[start:]
        masked_segment = masked_ids[start:]
        span = _find_reasoning_span(full_segment, masked_segment)
        if span is None:
            logger.warning(
                "Could not isolate reasoning_content tokens for assistant message %s. "
                "Leave `mask_reasoning_content=False` or ensure the chat template renders "
                "reasoning_content in a distinct block.",
                idx,
            )
            continue

        reasoning_start, reasoning_end = span
        for pos in range(
            min(start + reasoning_start, len(reasoning_mask)), min(start + reasoning_end, len(reasoning_mask))
        ):
            reasoning_mask[pos] = 1

    return reasoning_mask


def _mask_labels_to_last_turn(mask: List[int], ignore_index: int = -100) -> List[int]:
    """Restrict supervision to the final assistant turn (``mask_history``).

    Operates on any per-token sequence where ``ignore_index`` marks
    unsupervised positions: a label list (``ignore_index=-100``) or a 0/1
    assistant mask (``ignore_index=0``). Each assistant turn renders as a
    single contiguous supervised span, so this keeps only the last such run
    and rewrites every earlier supervised position to ``ignore_index``.

    Apply this to the assistant mask **before** any reasoning_content holes are
    punched into it; running it on already-holed labels would treat the
    reasoning gap as a turn boundary and drop in-turn content before the hole.

    Args:
        mask: per-token labels or 0/1 mask (``ignore_index`` marks unsupervised).
        ignore_index: the value marking unsupervised positions.

    Returns:
        The same list, mutated so only the final supervised run is kept.
    """
    last = -1
    for i in range(len(mask) - 1, -1, -1):
        if mask[i] != ignore_index:
            last = i
            break
    if last < 0:
        return mask
    start = last
    while start - 1 >= 0 and mask[start - 1] != ignore_index:
        start -= 1
    for i in range(start):
        mask[i] = ignore_index
    return mask


@torch.no_grad()
def _get_right_trailing_pad_mask(
    sequence: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Boolean mask identifying right-trailing padding positions.

    When *pad_token_id != eos_token_id*, it is simply ``sequence == pad_token_id``.

    When the two IDs collide, a plain equality check would also match real EOS
    tokens inside the content.  In that case the function locates the trailing
    contiguous run of the shared token and treats all positions **after the
    first one** in that run as padding.  The first token in the trailing run is
    the real EOS and is kept unmasked so the model still learns to predict
    end-of-sequence.

    Args:
        sequence: 1-D token id tensor.
        pad_token_id: The token id used for padding.
        eos_token_id: The token id used for end-of-sequence.  When equal to
            *pad_token_id* the positional trailing-run logic is used.

    Returns:
        Boolean tensor (same shape as *sequence*) where ``True`` = padding.
    """
    if pad_token_id != eos_token_id:
        return sequence == pad_token_id

    mask = torch.zeros(sequence.shape, dtype=torch.bool, device=sequence.device)
    non_pad_positions = (sequence != pad_token_id).nonzero(as_tuple=True)[0]
    if non_pad_positions.numel() > 0:
        last_content_idx = non_pad_positions[-1].item()
        # last_content_idx + 1 → real EOS (keep), last_content_idx + 2 → padding
        mask[last_content_idx + 2 :] = True
    else:
        # Entire sequence is the pad/eos token; keep the first as real EOS.
        mask[1:] = True
    return mask


def _pad_to_seq_length(sample, pad_token_id, seq_length):
    """Pad a sample to a specific sequence length."""
    n = seq_length - len(sample)
    if n == 0:
        return sample
    return sample + [pad_token_id] * n


_warned_add_pad_token = set()


def _add_pad_token(tokenizer):
    """Add pad token to tokenizer if not present."""
    pad_token_id = None
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if not "no_pad_id" in _warned_add_pad_token:
            _warned_add_pad_token.add("no_pad_id")
            logger.warning(
                "Tokenizer has no pad_token_id; falling back to eos_token_id (%s). "
                "This may cause issues if downstream code masks padding by token ID.",
                tokenizer.eos_token_id,
            )
    else:
        pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if (
        pad_token_id
        and pad_token_id == getattr(tokenizer, "eos_token_id", None)
        and not "pad_eq_eos" in _warned_add_pad_token
    ):
        _warned_add_pad_token.add("pad_eq_eos")
        logger.warning(
            "pad_token_id (%s) == eos_token_id (%s) for tokenizer '%s'. "
            "Ensure loss masking uses positional logic rather than token-ID comparison.",
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            getattr(tokenizer, "name_or_path", "unknown"),
        )
    return pad_token_id


def _has_chat_template(tokenizer: "PreTrainedTokenizer") -> bool:
    """
    Check if the tokenizer supports a chat template.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        True if the tokenizer supports a chat template, False otherwise.
    """
    return getattr(tokenizer, "chat_template", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def _package_tokenized_example(
    tokenizer,
    input_ids,
    assistant_masks,
    eos_token_id,
    pad_token_id,
    seq_length,
    truncation="do_not_truncate",
    padding="do_not_pad",
    unshifted=False,
):
    """
    Package a tokenized example with proper masking and padding.

    Args:
        tokenizer: The tokenizer to use.
        input_ids: The tokenized input ids.
        assistant_masks: Boolean mask indicating which tokens are assistant/answer tokens (1) vs prompt tokens (0).
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.
        truncation: Optional truncation strategy.
        padding: Optional padding strategy.
        unshifted: If True, return unshifted format for dLLM training
            (``input_ids`` at full length with ``loss_mask`` instead of
            shifted ``input_ids``/``labels``).
    Returns:
        A dictionary with input_ids, labels, and attention_mask.
        When *unshifted* is True, ``labels`` is replaced by ``loss_mask``.
    """
    if unshifted:
        # --- Unshifted dLLM format ---
        # No shift: input_ids stays at full length, loss_mask = assistant_masks.
        loss_mask = [int(bool(m)) for m in assistant_masks]

        # Compute content length (skip trailing pad tokens).
        content_length = len(input_ids)
        if pad_token_id is not None and content_length > 0:
            end = content_length
            while end > 0 and input_ids[end - 1] == pad_token_id:
                end -= 1
            if pad_token_id == eos_token_id:
                content_length = min(end + 1, content_length)
            else:
                content_length = end
        attention_mask = [1] * content_length + [0] * (len(input_ids) - content_length)

        if isinstance(seq_length, int) and padding in ("max_length",):
            input_ids = _pad_to_seq_length(input_ids, pad_token_id, seq_length)
            loss_mask = _pad_to_seq_length(loss_mask, 0, seq_length)

        attention_mask += [0] * (len(input_ids) - len(attention_mask))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "___PAD_TOKEN_IDS___": {
                "input_ids": pad_token_id,
                "loss_mask": 0,
                "attention_mask": 0,
            },
        }

    # --- Standard shifted NTP format ---
    labels = input_ids.copy()
    # Compute content length on the original input_ids (before the next-token
    # shift) so that pre-padded and non-padded inputs produce identical
    # attention masks.  The shift removes one token; when the input is padded
    # that token is a pad, but when unpadded it is the last real token.
    # Computing on the original and subtracting 1 gives the same result in
    # both cases.
    content_length = len(input_ids)
    if pad_token_id is not None and content_length > 0:
        end = content_length
        while end > 0 and input_ids[end - 1] == pad_token_id:
            end -= 1
        if pad_token_id == eos_token_id:
            content_length = min(end + 1, content_length)
        else:
            content_length = end
    input_ids = input_ids[:-1]
    content_length = max(0, min(content_length - 1, len(input_ids)))
    attention_mask = [1] * content_length + [0] * (len(input_ids) - content_length)
    # Labels: mask out prompt tokens
    labels[:] = [label if bool(m) else -100 for label, m in zip(labels, assistant_masks)]
    # remove BOS
    labels = labels[1:]
    if not _has_chat_template(tokenizer) and truncation is None:
        assert labels[-1] == eos_token_id, f"labels[-1]={labels[-1]} != eos_token_id={eos_token_id}"
        assert input_ids[-1] != eos_token_id, f"input_ids[-1]={input_ids[-1]} == eos_token_id={eos_token_id}"
    assert len(input_ids) == len(labels), f"len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"

    # Only pad to a fixed length for "max_length".  For "longest" / True the
    # collator pads to the longest sample in the batch, so the dataset must
    # return variable-length sequences (same as "do_not_pad").
    if isinstance(seq_length, int) and padding in ("max_length",):
        input_ids = _pad_to_seq_length(input_ids, pad_token_id, seq_length)
        labels = _pad_to_seq_length(labels, -100, seq_length)

    # the attention mask can also be extended in the collator with zeros.
    attention_mask += [0] * (len(labels) - len(attention_mask))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "___PAD_TOKEN_IDS___": {
            "input_ids": pad_token_id,
            "labels": -100,
            "attention_mask": 0,
        },
    }


def format_prompt_completion(
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    answer: str,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
    answer_only_loss_mask: bool = True,
    unshifted: bool = False,
) -> Dict[str, List[int]]:
    """
    Format a prompt-completion style example (without chat template).

    Args:
        tokenizer: The tokenizer to use.
        prompt: The prompt string (e.g. context + question).
        answer: The answer string.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.

    Returns:
        A dictionary with the formatted example.
    """
    full_text = prompt + answer

    # Tokenize separately to locate answer start
    if answer_only_loss_mask:
        # don't add eos token here. NOTE: this is only for calculating the length of the prompt.
        # we are not modifying the prompt to be returned here.
        prompt_ids = [tokenizer.bos_token_id] if getattr(tokenizer, "add_bos_token", False) else []
        prompt_ids += tokenizer(prompt, add_special_tokens=False)["input_ids"]
        len_prompt_ids = len(prompt_ids)
    else:
        len_prompt_ids = 0
    # transformers 5.5.0 still honored `padding_side: "right"` baked into the
    # tokenizer's saved tokenizer_config.json, but 5.8.1 ignores that field and
    # uses the LlamaTokenizer class default ("left"). Hardcode "right" here so
    # pad positions land at the end (the label-masking / attention-mask logic
    # below assumes right padding).
    _saved_padding_side = getattr(tokenizer, "padding_side", None)
    if _saved_padding_side is not None:
        tokenizer.padding_side = "right"
    try:
        tokenized = tokenizer(
            full_text,
            padding=padding,
            truncation=truncation,
            max_length=seq_length,
        )
    finally:
        if _saved_padding_side is not None:
            tokenizer.padding_side = _saved_padding_side
    input_ids = tokenized["input_ids"]

    # Create assistant_masks: 0 for prompt tokens, 1 for answer tokens
    assistant_masks = [0] * len_prompt_ids + [1] * (len(input_ids) - len_prompt_ids)

    # Zero out the loss mask at padding positions using the tokenizer's
    # own attention_mask so pad tokens are never treated as supervised.
    tokenizer_attn_mask = tokenized.get("attention_mask")
    if tokenizer_attn_mask is not None:
        for i in range(min(len(assistant_masks), len(tokenizer_attn_mask))):
            if not tokenizer_attn_mask[i]:
                assistant_masks[i] = 0

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=assistant_masks,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        truncation=truncation,
        padding=padding,
        unshifted=unshifted,
    )


def format_chat_template(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, Any]],
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
    tools: Optional[List[Dict]] = None,
    answer_only_loss_mask: bool = True,
    mask_reasoning_content: bool = False,
    train_on_last_turn_only: bool = False,
    unshifted: bool = False,
) -> Dict[str, List[int]]:
    """
    Format a chat template style example.

    Args:
        tokenizer: The tokenizer to use.
        formatted_text: The formatted text, with role tags embedded in the content.
        eos_token_id: The end-of-sequence token id.
        pad_token_id: The padding token id.
        seq_length: Optional sequence length for padding.
        tools: Optional list of tool definitions for function calling.
        answer_only_loss_mask: Whether to compute the loss mask only on the answer tokens.
        mask_reasoning_content: Whether to exclude rendered reasoning_content tokens from loss.
        train_on_last_turn_only: Whether to supervise only the final assistant turn,
            masking every earlier assistant turn (``mask_history``). Applied to the
            assistant mask before reasoning_content is masked out.

    Returns:
        A dictionary with the formatted example.
    """
    # Ensure we have a usable chat template
    if not _has_chat_template(tokenizer):
        raise ValueError("Tokenizer lacks a usable chat template (chat_template/apply_chat_template)")

    # Resolve the template string — some tokenizers store multiple templates as a dict
    # (keyed by name, e.g. "default", "tool_use"). We need the raw string for regex checks.
    chat_template_str = tokenizer.chat_template
    if isinstance(chat_template_str, dict):
        chat_template_str = chat_template_str.get("default", next(iter(chat_template_str.values())))

    template_has_generation_kwd = GENERATION_REGEX.search(chat_template_str) is not None
    template_mentions_reasoning_content = "reasoning_content" in chat_template_str
    has_reasoning_content = any(
        message.get("role") == "assistant" and bool(message.get("reasoning_content")) for message in formatted_text
    )

    if has_reasoning_content and not template_mentions_reasoning_content:
        logger.warning(
            "Assistant messages include `reasoning_content`, but the active chat template does not reference "
            "`reasoning_content`. Those reasoning traces may be dropped from training data."
        )

    tokenized_chat = tokenizer.apply_chat_template(
        formatted_text,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=template_has_generation_kwd,
        padding=padding,
        truncation=truncation,
        max_length=seq_length,
    )

    input_ids = tokenized_chat.get("input_ids")

    def _unpadded_full_ids() -> "list[int] | None":
        """Unpadded full-conversation ids from this tokenization, already known
        without another tokenizer call; the mask builders use them to skip
        re-tokenizing the full prefix and to validate prefix renders. None
        (recompute in the builder) when no attention_mask is available."""
        attn = tokenized_chat.get("attention_mask")
        if attn is None:
            return None
        return [token for token, keep in zip(input_ids, attn) if keep]

    if template_has_generation_kwd:
        mask = tokenized_chat["assistant_masks"]
    elif not template_has_generation_kwd and answer_only_loss_mask:
        mask = _build_multiturn_assistant_mask(
            tokenizer,
            formatted_text,
            input_ids,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
            unpadded_full_ids=_unpadded_full_ids(),
        )
        # _build_multiturn_assistant_mask computes indices from unpadded
        # lengths — shift for left-padding tokenizers.
        mask = _maybe_shift_mask_for_left_padding(mask, tokenizer, tokenized_chat.get("attention_mask"))
    else:
        mask = [1] * len(input_ids)

    # Zero out the loss mask at padding positions using the tokenizer's
    # own attention_mask so pad tokens are never treated as supervised.
    tokenizer_attn_mask = tokenized_chat.get("attention_mask")
    if tokenizer_attn_mask is not None:
        for i in range(min(len(mask), len(tokenizer_attn_mask))):
            if not tokenizer_attn_mask[i]:
                mask[i] = 0

    # Restrict to the last assistant turn before reasoning is masked, so the
    # contiguous-run heuristic sees a hole-free mask (one run per turn).
    if train_on_last_turn_only:
        _mask_labels_to_last_turn(mask, ignore_index=0)

    if mask_reasoning_content and has_reasoning_content:
        reasoning_mask = _build_reasoning_mask(
            tokenizer,
            formatted_text,
            input_ids,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
            unpadded_full_ids=_unpadded_full_ids(),
        )
        # _build_reasoning_mask also computes from unpadded lengths.
        reasoning_mask = _maybe_shift_mask_for_left_padding(
            reasoning_mask, tokenizer, tokenized_chat.get("attention_mask")
        )
        mask = [assistant if not reasoning else 0 for assistant, reasoning in zip(mask, reasoning_mask)]

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        truncation=truncation,
        padding=padding,
        unshifted=unshifted,
    )
