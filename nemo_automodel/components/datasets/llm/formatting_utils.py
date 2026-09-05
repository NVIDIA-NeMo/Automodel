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
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch

logger = logging.getLogger(__name__)


def _resolve_chat_template(chat_template: str | None) -> str | None:
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


def _tokenize_chat(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, Any]],
    *,
    tools: List[Dict] | None = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: int | None = None,
    **template_kwargs: Any,
) -> List[int]:
    """Tokenize chat messages without padding and return input ids.

    ``template_kwargs`` (for example ``add_generation_prompt`` or
    ``enable_thinking``) are forwarded to ``apply_chat_template`` as given, so the
    default path calls the tokenizer exactly as before.
    """
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=truncation,
        max_length=seq_length,
        **template_kwargs,
    )
    return tokenized_chat.get("input_ids", [])


def _tokenized_chat_length(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, str]],
    *,
    tools: List[Dict] | None = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: int | None = None,
) -> int:
    """Return the tokenized chat length for a message prefix without padding."""
    return len(_tokenize_chat(tokenizer, messages, tools=tools, truncation=truncation, seq_length=seq_length))


def _maybe_shift_mask_for_left_padding(
    mask: List[int],
    tokenizer: "PreTrainedTokenizer",
    attention_mask: List[int] | None,
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


def _is_consistent_render_prefix(
    prefix_ids: "list[int]", reference_ids: "list[int]", *, trailing_token_id: "int | None" = None
) -> bool:
    """Check that a prefix render matches the start of the full-conversation render.

    Locating spans by prefix lengths is only correct when
    ``render(messages[:k])`` reproduces the first tokens of ``render(messages)``.
    The comparison is exact unless a multi-token prefix ends with a known
    standalone terminator that is replaced when the next message is appended.
    """
    if len(prefix_ids) > len(reference_ids):
        return False
    if prefix_ids == reference_ids[: len(prefix_ids)]:
        return True
    return (
        len(prefix_ids) > 1
        and trailing_token_id is not None
        and prefix_ids[-1] == trailing_token_id
        and prefix_ids[:-1] == reference_ids[: len(prefix_ids) - 1]
    )


def _build_multiturn_assistant_mask(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: List[Dict[str, Any]],
    input_ids: List[int],
    *,
    tools: List[Dict] | None = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: int | None = None,
    unpadded_full_ids: "list[int] | None" = None,
    prefix_cache: "dict[int, list[int]] | None" = None,
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
    - Each prefix length is memoized so a boundary shared by adjacent turns (a
      turn's end and the next turn's start) is tokenized at most once. When the
      caller passes a ``prefix_cache`` dict (``k`` -> ids of
      ``formatted_text[:k]``) the rendered ids are kept in it as well, so
      :func:`_build_generation_prompt_mask` can reuse these renders instead of
      repeating them; without one only the lengths are retained, since holding
      every prefix render would cost O(turns x sequence length) per sample.

    Every tokenized prefix is validated against ``unpadded_full_ids`` (see
    :func:`_is_consistent_render_prefix`). A known trailing EOS may differ when
    the prefix is rendered alone. Any other mismatch raises :class:`ValueError`
    because prefix arithmetic cannot place the spans safely.
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
    if prefix_cache is not None:
        prefix_cache.setdefault(len(formatted_text), unpadded_full_ids)
        length_cache.update((k, len(ids)) for k, ids in prefix_cache.items())

    def prefix_length(k: int) -> int:
        if k not in length_cache:
            prefix_ids = _tokenize_chat(
                tokenizer,
                formatted_text[:k],
                tools=tools,
                truncation=truncation,
                seq_length=seq_length,
            )
            if not _is_consistent_render_prefix(
                prefix_ids, unpadded_full_ids, trailing_token_id=getattr(tokenizer, "eos_token_id", None)
            ):
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
            if prefix_cache is not None:
                prefix_cache[k] = prefix_ids
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


def _truncation_window_offset(
    tokenizer: "PreTrainedTokenizer", unpadded_full_ids: list[int], reference_full_ids: list[int], what: str
) -> int:
    """Return where the retained (truncated) tokens start inside the untruncated render.

    The retained window must be a contiguous prefix (right truncation) or suffix
    (left truncation) of ``reference_full_ids``. A short window can match both
    ends of a render that starts and ends with the same tokens, so the side the
    tokenizer actually truncates on (``tokenizer.truncation_side``) is tried first.
    """
    prefix_offset, suffix_offset = 0, len(reference_full_ids) - len(unpadded_full_ids)
    candidates = [prefix_offset, suffix_offset]
    if getattr(tokenizer, "truncation_side", "right") == "left":
        candidates.reverse()
    for offset in candidates:
        if unpadded_full_ids == reference_full_ids[offset : offset + len(unpadded_full_ids)]:
            return offset
    raise ValueError(
        f"Cannot mask {what} after truncation because the retained tokens are not a contiguous "
        "prefix or suffix of the untruncated conversation render."
    )


_SENTINEL_TEXTS = ("", "0", "1")
"""Replacement texts for :func:`_generation_prefix_bound`. The empty text renders the turn
with no generated text at all, which is what fixes the content boundary; the two others are
compared token by token, so what matters is that they tokenize differently, not how they read."""


def _perturbed_assistant_message(message: dict[str, Any], sentinel: str) -> dict[str, Any]:
    """Return a copy of an assistant message whose model-generated text is replaced by ``sentinel``.

    Every field the model would produce is swapped: ``content`` (string or text
    parts) and any reasoning field. ``tool_calls`` are dropped rather than
    rewritten, because their serialization starts with boilerplate
    (``{"name": "``) the model nevertheless generates; dropping them makes the
    render diverge at the call marker instead.
    """
    perturbed = {k: v for k, v in message.items() if k != "tool_calls"}
    content = message.get("content")
    if isinstance(content, list):
        perturbed["content"] = [
            {**part, "text": sentinel} if isinstance(part, dict) and "text" in part else part for part in content
        ]
    else:
        perturbed["content"] = sentinel
    for key in ("reasoning_content", "reasoning"):
        if message.get(key):
            perturbed[key] = sentinel
    return perturbed


def _generation_prefix_bound(
    tokenizer: "PreTrainedTokenizer",
    prefix: list[dict[str, Any]],
    message: dict[str, Any],
    conversation_ids: list[int],
    turn: list[int],
    **render_kwargs: Any,
) -> int:
    """Return how many leading tokens of ``turn`` provably do not depend on ``message``.

    The turn is rendered again with its generated text swapped for each of
    :data:`_SENTINEL_TEXTS`, and only tokens every render shares with the real
    turn, at the same position, may be masked as the generation prompt. The
    content boundary comes from the render with the generated text removed
    entirely: whatever the template emits before the first token that render
    lacks is present with no message text at all, so it cannot be owned by the
    content. That render alone is not enough, because a template's closing
    token can coincide with the real text's first token, so two non-empty
    sentinels must also agree up to the bound and diverge from each other
    afterwards; checking token ids rather than characters keeps normalization
    or an UNK mapping from folding a sentinel onto the real text. A prefix
    token the tokenizer emits for every non-empty value (a word-start marker,
    say) is absent from the empty render and therefore never masked. When the
    non-empty sentinel renders do not diverge, or any render already differs
    inside the conversation prefix, nothing is proven and the bound is ``0``
    (nothing is masked).

    Removing the text can also merge the template's own characters into one
    token: Qwen3-Thinking renders a reasoning turn as ``<think>\\n`` + text,
    and with no text the ``\\n`` fuses with the ``\\n</think>`` that follows
    into a single ``\\n\\n`` token, so the empty render diverges from the real
    turn one token before the text starts. The bound stops there and that
    ``\\n`` keeps its supervision. Extending it would need a proof on the
    rendered text that the real token lies inside the template's characters;
    a prefix relation between tokenizer-internal token strings is not one
    (byte-level tokens can share an internal prefix while their decoded text
    differs), so the comparison is on token ids only and fails closed.

    Token ids alone also cannot tell a template's closing token from a
    content token that maps to the same id (through normalization or an UNK
    mapping): with a header-less turn whose closing ``<eot>`` and content
    marker ``<word_start>`` share an id, the empty render ``[X]`` is a prefix
    of the real turn ``[X, text..., X]`` although its only token is the
    closing. The empty render is therefore also aligned from its end: it is
    ``template prefix + closing`` and each sentinel render is ``template
    prefix + text + closing``, so their common suffix is at least the
    closing, and no token inside it may be masked. Everything a token could
    be is excluded; the two alignments never overlap.
    """
    tails: list[list[int]] = []
    for sentinel in _SENTINEL_TEXTS:
        ids = _tokenize_chat(tokenizer, prefix + [_perturbed_assistant_message(message, sentinel)], **render_kwargs)
        if ids[: len(conversation_ids)] != conversation_ids:
            return 0
        tails.append(ids[len(conversation_ids) :])
    empty, sentinels = tails[0], tails[1:]
    if sentinels[0] == sentinels[1]:
        return 0
    bound = min(_common_prefix_length(tail, turn) for tail in sentinels)
    bound = min(bound, _common_prefix_length(*sentinels))
    bound = min(bound, _common_prefix_length(empty, turn))
    closing = max(_common_suffix_length(empty, tail) for tail in sentinels)
    return max(0, min(bound, len(empty) - closing))


def _common_suffix_length(left: list[int], right: list[int]) -> int:
    """Return the number of trailing elements ``left`` and ``right`` share."""
    n = 0
    while n < min(len(left), len(right)) and left[-(n + 1)] == right[-(n + 1)]:
        n += 1
    return n


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return the number of leading elements ``left`` and ``right`` share."""
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def _find_reasoning_span(full_segment: List[int], masked_segment: List[int]) -> tuple[int, int] | None:
    """Locate the contiguous token span attributable to reasoning content."""
    prefix_len = _common_prefix_length(full_segment, masked_segment)

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
    tools: List[Dict] | None = None,
    truncation: Union[str, bool] = "do_not_truncate",
    seq_length: int | None = None,
    unpadded_full_ids: "list[int] | None" = None,
) -> List[int]:
    """Build a token mask for reasoning_content spans inside assistant turns.

    Each span is isolated by comparing the full conversation render with a
    second full render where only that message's ``reasoning_content`` is
    cleared. This remains correct when the template rewrites earlier turns
    based on later messages. If clearing the field does not change the render,
    that turn's reasoning is not present and no tokens need to be masked.
    """
    reasoning_mask = [0] * len(input_ids)

    if unpadded_full_ids is None:
        unpadded_full_ids = _tokenize_chat(
            tokenizer,
            formatted_text,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )

    # Truncation only changes the render when it actually cut something; a sample
    # that fits keeps identical truncated and untruncated renders.
    truncation_enabled = truncation not in (False, None, "do_not_truncate") and (
        seq_length is None or len(unpadded_full_ids) >= seq_length
    )
    reference_full_ids = unpadded_full_ids
    reference_offset = 0
    if truncation_enabled:
        reference_full_ids = _tokenize_chat(
            tokenizer,
            formatted_text,
            tools=tools,
            truncation=False,
            seq_length=None,
        )
        reference_offset = _truncation_window_offset(
            tokenizer, unpadded_full_ids, reference_full_ids, "reasoning_content"
        )

    for idx, message in enumerate(formatted_text):
        if message.get("role") != "assistant" or not message.get("reasoning_content"):
            continue

        masked_messages = formatted_text[:idx] + [_masked_reasoning_message(message)] + formatted_text[idx + 1 :]
        masked_ids = _tokenize_chat(
            tokenizer,
            masked_messages,
            tools=tools,
            truncation=False if truncation_enabled else truncation,
            seq_length=None if truncation_enabled else seq_length,
        )

        span = _find_reasoning_span(reference_full_ids, masked_ids)
        if span is None:
            logger.warning(
                "Could not isolate reasoning_content tokens for assistant message %s. The chat template may not "
                "render that field, or it may not render it in a distinct block.",
                idx,
            )
            continue

        reasoning_start, reasoning_end = span
        reasoning_start = max(0, reasoning_start - reference_offset)
        reasoning_end = min(len(unpadded_full_ids), reasoning_end - reference_offset)
        for pos in range(reasoning_start, reasoning_end):
            reasoning_mask[pos] = 1

    return reasoning_mask


_warned_generation_prompt: set[str] = set()


def _match_generation_prompt(block: list[int], turn: list[int], eos_token_id: int | None) -> tuple[int, bool]:
    """Match the tokens a generation prompt adds against the start of a rendered turn.

    Returns ``(matched, complete)``: how many leading tokens of ``turn`` the
    prompt reproduces, and whether the whole prompt was reproduced. Two anchors
    are tried and the better one (complete first, then longer) wins:

    - the prompt's own first token, when it equals ``turn[0]``;
    - the last place ``turn[0]`` occurs inside the prompt. A template appends
      its generation prompt at the very end of the render, so whatever precedes
      that occurrence (the ``/nothink`` GLM appends to the user turn, a rewritten
      system block) belongs to the conversation prefix, not to the prompt. This
      anchor is skipped when its remainder contains ``eos_token_id``: a
      generation prompt never closes a turn, so that occurrence is rendered
      history and should not outscore the prompt's own header.

    This is a scoring rule, not the safety net: templates close turns with
    tokens other than ``eos_token_id`` and continuation turns have no header,
    so a wrong anchor can still reproduce real content here. The caller caps
    the result at the turn's message-independent prefix (see
    :func:`_build_generation_prompt_mask`), which is what keeps content out.
    """
    if not block or not turn:
        return 0, False
    candidates = []
    if block[0] == turn[0]:
        candidates.append(0)
    anchor = next((i for i in reversed(range(1, len(block))) if block[i] == turn[0]), None)
    if anchor is not None and (eos_token_id is None or eos_token_id not in block[anchor:]):
        candidates.append(anchor)
    best = (False, 0)
    for anchor in candidates:
        suffix = block[anchor:]
        matched = _common_prefix_length(suffix, turn)
        best = max(best, (matched == len(suffix), matched))
    return best[1], best[0]


def _build_generation_prompt_mask(
    tokenizer: "PreTrainedTokenizer",
    formatted_text: list[dict[str, Any]],
    input_ids: list[int],
    *,
    tools: list[dict] | None = None,
    truncation: str | bool = "do_not_truncate",
    seq_length: int | None = None,
    unpadded_full_ids: list[int] | None = None,
    prefix_cache: dict[int, list[int]] | None = None,
) -> list[int]:
    """Mark the tokens of each assistant turn that the generation prompt supplies.

    At inference the chat template, not the model, emits the assistant role
    header plus whatever it inserts ahead of the first generated token. For a
    turn without reasoning that is typically an empty reasoning block such as
    ``<think></think>`` (Nemotron), ``<think>\\n\\n</think>\\n\\n`` (Qwen3) or
    ``</think>`` (DeepSeek V3.1); for a thinking turn it may be an opening
    ``<think>\\n``. The model never produces these tokens, so supervising them
    only teaches template boilerplate, and an empty reasoning block also
    teaches an immediate ``<think>`` -> ``</think>`` transition.

    The span is located without knowing any tag string. For every assistant
    turn, ``messages[:idx]`` is rendered with ``add_generation_prompt=True`` in
    both thinking modes (``enable_thinking`` True and False), and the tokens
    each prompt adds are matched against the start of the turn in the full
    render (see :func:`_match_generation_prompt`). The mode whose prompt
    reproduces the turn best wins (a complete match first, then the longer
    one), because the data alone cannot tell the modes apart: a thinking turn
    may carry its reasoning in ``reasoning_content`` or inline in ``content``,
    and a template may or may not honor ``enable_thinking`` at all.

    The match alone cannot prove it stopped short of the model's own text: a
    template may rewrite history when it renders the generation prompt (Gemma
    prepends a thinking system block, GLM appends ``/nothink``), and a turn
    that continues a previous one (after a tool response) has no header, so
    the prompt's added tokens can contain earlier turns whose text repeats the
    current one. Every anchor is therefore capped by a structural bound
    (:func:`_generation_prefix_bound`): the turn is rendered again with its
    generated text removed and with it replaced by two different sentinels, and
    only tokens every render shares, the part of the turn that provably does
    not depend on the message, may be marked. The empty render fixes the
    content boundary (a token the tokenizer emits for any value is absent from
    it), the check is on token ids, so a tokenizer that normalizes a sentinel
    onto the real text's first token cannot widen it, and it fails closed
    (marks nothing) when the sentinel renders do not diverge. Assistant content
    can never be reached, whichever anchor won.

    Truncation is handled like :func:`_build_reasoning_mask`: spans are located
    in the untruncated render and mapped back through the retained window,
    which must be a contiguous prefix or suffix of it. A turn whose prefix
    render cannot be aligned with the full render is left untouched (warned
    once per process), as is a leading assistant turn whose template cannot
    render an empty conversation; any other render error propagates. Positions are computed from unpadded
    (left-aligned) ids, like :func:`_build_multiturn_assistant_mask`, whose
    prefix renders are reused through ``prefix_cache``. Up to five extra
    ``apply_chat_template`` renders per assistant turn are the price, which is
    why this is opt-in.
    """
    generation_prompt_mask = [0] * len(input_ids)

    if unpadded_full_ids is None:
        unpadded_full_ids = _tokenize_chat(
            tokenizer,
            formatted_text,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
        )
    if prefix_cache is None:
        prefix_cache = {}

    # Truncation only changes the render when it actually cut something; a sample
    # that fits keeps identical truncated and untruncated renders.
    truncation_enabled = truncation not in (False, None, "do_not_truncate") and (
        seq_length is None or len(unpadded_full_ids) >= seq_length
    )
    reference_full_ids = unpadded_full_ids
    reference_offset = 0
    render_kwargs: dict[str, Any] = dict(tools=tools, truncation=truncation, seq_length=seq_length)
    if truncation_enabled:
        reference_full_ids = _tokenize_chat(tokenizer, formatted_text, tools=tools, truncation=False, seq_length=None)
        reference_offset = _truncation_window_offset(
            tokenizer, unpadded_full_ids, reference_full_ids, "the generation prompt"
        )
        render_kwargs = dict(tools=tools, truncation=False, seq_length=None)
        # The shared cache holds truncated prefix renders; this builder needs untruncated ones.
        prefix_cache = {}
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def skip(reason: str, idx: int) -> None:
        if reason not in _warned_generation_prompt:
            _warned_generation_prompt.add(reason)
            logger.warning(
                "Could not %s for assistant message %s; its template-supplied tokens stay in the loss. "
                "This warning is shown once.",
                reason,
                idx,
            )

    for idx, message in enumerate(formatted_text):
        if message.get("role") != "assistant":
            continue

        prefix = formatted_text[:idx]
        base_ids = prefix_cache.get(idx)
        if base_ids is None and idx == 0:
            # A leading assistant turn has an empty prefix, which templates that read
            # messages[0] cannot render (with or without the generation prompt).
            try:
                base_ids = _tokenize_chat(tokenizer, prefix, **render_kwargs)
                _tokenize_chat(tokenizer, prefix, add_generation_prompt=True, **render_kwargs)
            except Exception:
                skip("render the generation prompt of a leading assistant turn", idx)
                continue
        if base_ids is None:
            base_ids = _tokenize_chat(tokenizer, prefix, **render_kwargs)
        if not _is_consistent_render_prefix(base_ids, reference_full_ids, trailing_token_id=eos_token_id):
            skip("align the generation prompt", idx)
            continue
        prefix_cache[idx] = base_ids  # validated, so the multiturn builder may reuse it
        # len(base_ids), or one less when the prefix render ends on a trailing
        # terminator that the full render replaces.
        start = _common_prefix_length(base_ids, reference_full_ids)
        turn = reference_full_ids[start:]

        best = (False, 0)
        previous_ids: list[int] | None = None
        for enable_thinking in (True, False):
            generation_ids = _tokenize_chat(
                tokenizer, prefix, add_generation_prompt=True, enable_thinking=enable_thinking, **render_kwargs
            )
            if generation_ids == previous_ids:  # the template ignores enable_thinking
                continue
            previous_ids = generation_ids
            # Everything the generation prompt adds to (or changes in) the plain prefix render.
            block = generation_ids[_common_prefix_length(base_ids, generation_ids) :]
            matched, complete = _match_generation_prompt(block, turn, eos_token_id)
            best = max(best, (complete, matched))
        if best[1] == 0:
            continue

        # The tokens of the turn that provably do not depend on the message, i.e. the
        # template's own prefix: no anchor may mark anything past them.
        bound = _generation_prefix_bound(tokenizer, prefix, message, reference_full_ids[:start], turn, **render_kwargs)

        first = start - reference_offset
        for pos in range(max(first, 0), min(first + min(best[1], bound), len(generation_prompt_mask))):
            generation_prompt_mask[pos] = 1

    return generation_prompt_mask


def _subtract_mask(mask: list[int], removed: list[int]) -> list[int]:
    """Zero every position of ``mask`` that ``removed`` marks."""
    return [keep if not drop else 0 for keep, drop in zip(mask, removed)]


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
    seq_length: int | None = None,
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
    seq_length: int | None = None,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "do_not_truncate",
    tools: List[Dict] | None = None,
    answer_only_loss_mask: bool = True,
    mask_reasoning_content: bool = False,
    train_on_last_turn_only: bool = False,
    unshifted: bool = False,
    mask_generation_prompt: bool = False,
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
        mask_generation_prompt: Whether to exclude from the loss the tokens of each
            assistant turn that the chat template's generation prompt supplies at
            inference: the role header and any template-inserted empty reasoning
            block (for example ``<think></think>``). See
            :func:`_build_generation_prompt_mask`.

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

    # Unpadded full-conversation ids from this tokenization, already known
    # without another tokenizer call; the mask builders use them to skip
    # re-tokenizing the full prefix and to validate prefix renders. Computed
    # lazily (the common generation-kwd path never needs it) but memoized so
    # the multiturn and reasoning-mask builders share one computation instead
    # of each rebuilding it. None (recompute in the builder) when no
    # attention_mask is available.
    _unpadded_full_ids_memo: "dict[str, list[int] | None]" = {}
    # Prefix renders (k -> ids of formatted_text[:k]) shared by the mask builders. Only
    # the generation-prompt builder reuses the ids; without it the multiturn builder
    # keeps prefix lengths alone rather than every render.
    prefix_cache: dict[int, list[int]] | None = {} if mask_generation_prompt else None

    def unpadded_full_ids() -> "list[int] | None":
        if "value" not in _unpadded_full_ids_memo:
            attn = tokenized_chat.get("attention_mask")
            _unpadded_full_ids_memo["value"] = (
                [token for token, keep in zip(input_ids, attn) if keep] if attn is not None else None
            )
        return _unpadded_full_ids_memo["value"]

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
            unpadded_full_ids=unpadded_full_ids(),
            prefix_cache=prefix_cache,
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

    # Drop the template-supplied prefix of every assistant turn (role header and
    # any empty reasoning block). Independent of the reasoning mask below: that one
    # removes rendered reasoning text, this one removes what the generation prompt
    # would emit around it.
    if mask_generation_prompt:
        generation_prompt_mask = _build_generation_prompt_mask(
            tokenizer,
            formatted_text,
            input_ids,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
            unpadded_full_ids=unpadded_full_ids(),
            prefix_cache=prefix_cache,
        )
        # Computed from unpadded lengths, like the builders above.
        generation_prompt_mask = _maybe_shift_mask_for_left_padding(
            generation_prompt_mask, tokenizer, tokenized_chat.get("attention_mask")
        )
        mask = _subtract_mask(mask, generation_prompt_mask)

    if mask_reasoning_content and has_reasoning_content:
        reasoning_mask = _build_reasoning_mask(
            tokenizer,
            formatted_text,
            input_ids,
            tools=tools,
            truncation=truncation,
            seq_length=seq_length,
            unpadded_full_ids=unpadded_full_ids(),
        )
        # _build_reasoning_mask also computes from unpadded lengths.
        reasoning_mask = _maybe_shift_mask_for_left_padding(
            reasoning_mask, tokenizer, tokenized_chat.get("attention_mask")
        )
        mask = _subtract_mask(mask, reasoning_mask)

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
