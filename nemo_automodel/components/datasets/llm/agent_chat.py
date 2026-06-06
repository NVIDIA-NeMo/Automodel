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
"""Multi-turn agent SFT dataset adapter.

Loads function-calling chat datasets where each example contains tool
definitions and a multi-turn message list that includes tool calls and
tool responses, then renders them through the tokenizer's chat template
with ``answer_only_loss_mask=True`` so that ``user`` and ``tool`` tokens
are excluded from the loss.

Two input schemas are accepted:

1. Swift / chatml ``messages`` schema (used by
   ``AI-ModelScope/function-calling-chatml``)::

       {
           "tools": "[{...openai tool schema...}]",
           "messages": [
               {"role": "user",          "content": "..."},
               {"role": "tool_call",     "content": "{\\"name\\": ..., \\"arguments\\": ...}"},
               {"role": "tool_response", "content": "..."},
               {"role": "assistant",     "content": "..."}
           ]
       }

2. ShareGPT ``conversations`` schema (used by
   ``llamafactory/glaive_toolcall_en`` and similar)::

       {
           "tools": "[...]",
           "conversations": [
               {"from": "human",         "value": "..."},
               {"from": "function_call", "value": "..."},
               {"from": "observation",   "value": "..."},
               {"from": "gpt",           "value": "..."}
           ]
       }

Consecutive ``tool_call`` entries are merged into a single assistant
message with parallel ``tool_calls``; the following ``tool_response``
entries are paired with those calls in order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _tokenized_chat_length,
    format_chat_template,
)

logger = logging.getLogger(__name__)


_VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool_call", "tool_response", "tool"}

# ShareGPT ``from`` -> chatml ``role`` mapping.
_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "function_call": "tool_call",
    "tool_call": "tool_call",
    "observation": "tool_response",
    "tool_response": "tool_response",
    "tool": "tool_response",
}


def _json_load_if_str(value: Any) -> Any:
    """Parse ``value`` as JSON if it is a string, otherwise return as-is."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _sharegpt_to_chatml(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert ShareGPT ``{from, value}`` turns to chatml ``{role, content}``."""
    out: List[Dict[str, Any]] = []
    for turn in conversations:
        src_role = turn.get("from")
        if src_role not in _SHAREGPT_ROLE_MAP:
            raise ValueError(f"Unsupported sharegpt role: {src_role!r}")
        chatml_turn = {"role": _SHAREGPT_ROLE_MAP[src_role], "content": turn.get("value", "")}
        # Carry an explicit reasoning/thinking field through if the export
        # stores it alongside the turn (e.g. ``reasoning_content``).
        if turn.get("reasoning_content"):
            chatml_turn["reasoning_content"] = turn["reasoning_content"]
        out.append(chatml_turn)
    return out


def _convert_messages(
    messages: List[Dict[str, Any]],
    example_id: Optional[Union[int, str]] = None,
    drop_history_reasoning_content: bool = False,
) -> List[Dict[str, Any]]:
    """Convert chatml-style agent messages to OpenAI chat-completions format.

    Consecutive ``tool_call`` entries collapse into one assistant message
    with parallel ``tool_calls``. When the preceding emitted turn is an
    assistant message without tool_calls (e.g. an assistant ``content``
    that reasons before calling tools), the tool_calls attach to that
    message instead of creating a second consecutive assistant turn —
    this preserves the natural single-turn shape the model produces at
    inference. ``tool_response`` (or ``tool``) entries that follow are
    paired with those tool_call ids in order.

    Args:
        messages: chatml-style turns with roles in ``_VALID_MESSAGE_ROLES``.
        example_id: Optional identifier used to derive unique tool_call ids.
        drop_history_reasoning_content: If True, strip ``reasoning_content``
            from every assistant turn except the final one, so historical
            thinking traces are not rendered into the prompt. This matches
            inference, where the model never sees its own prior-turn thinking.

    Returns:
        A list of OpenAI-format messages suitable for ``apply_chat_template``.
    """
    out: List[Dict[str, Any]] = []
    pending_call_ids: List[str] = []
    call_counter = 0
    id_prefix = f"call_{example_id}" if example_id is not None else "call"

    i = 0
    while i < len(messages):
        role = messages[i].get("role")
        if role not in _VALID_MESSAGE_ROLES:
            raise ValueError(f"Unsupported role in agent messages: {role!r}")

        if role == "tool_call":
            tool_calls: List[Dict[str, Any]] = []
            pending_call_ids = []
            while i < len(messages) and messages[i].get("role") == "tool_call":
                call = _json_load_if_str(messages[i].get("content") or "{}")
                name = call.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"tool_call missing `name`: {messages[i]!r}")
                args = call.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                cid = f"{id_prefix}_{call_counter}"
                call_counter += 1
                pending_call_ids.append(cid)
                tool_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )
                i += 1
            if out and out[-1].get("role") == "assistant" and "tool_calls" not in out[-1]:
                # Attach tool_calls to the prior assistant text turn so the
                # rendered chat keeps a single assistant message (matching
                # what the model emits at inference: think/text + tool_call).
                out[-1]["tool_calls"] = tool_calls
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        elif role in ("tool_response", "tool"):
            local_idx = 0
            while i < len(messages) and messages[i].get("role") in ("tool_response", "tool"):
                if local_idx < len(pending_call_ids):
                    tool_call_id = pending_call_ids[local_idx]
                else:
                    tool_call_id = f"{id_prefix}_response_{call_counter}_{local_idx}"
                content = messages[i].get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                out.append({"role": "tool", "content": content, "tool_call_id": tool_call_id})
                i += 1
                local_idx += 1
        else:
            # Reset pending_call_ids so a later orphan tool_response does not
            # silently reuse stale IDs from the previous tool_call group.
            pending_call_ids = []
            content = messages[i].get("content", "")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            msg: Dict[str, Any] = {"role": role, "content": content}
            # Preserve an assistant turn's reasoning/thinking trace so it
            # survives into the rendered chat (chat templates that reference
            # ``reasoning_content`` will emit it; otherwise it is dropped with
            # a warning by ``format_chat_template``). Carried through here so a
            # following tool_call group can merge onto this same turn.
            if role == "assistant":
                reasoning = messages[i].get("reasoning_content")
                if reasoning:
                    msg["reasoning_content"] = reasoning if isinstance(reasoning, str) else str(reasoning)
            out.append(msg)
            i += 1

    if drop_history_reasoning_content:
        last_assistant = max(
            (idx for idx, m in enumerate(out) if m.get("role") == "assistant"),
            default=-1,
        )
        for idx, m in enumerate(out):
            if idx != last_assistant and m.get("role") == "assistant":
                m.pop("reasoning_content", None)

    return out


def _truncate_messages_to_fit(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    seq_length: int,
) -> List[Dict[str, Any]]:
    """Drop the oldest conversation exchanges until the dialogue fits ``seq_length``.

    Unlike token-level ``truncation`` (which clips from the tokenizer's
    ``truncation_side`` and so typically drops the final assistant answer — the
    supervised target), this drops whole oldest *exchanges* while keeping any
    leading ``system`` message, the tool definitions, and the final exchange
    (the user turn that produces the last assistant answer). Tool-call/response
    pairing among the survivors is preserved because exchanges are cut only at
    ``user`` boundaries.

    Args:
        tokenizer: tokenizer with a chat template.
        messages: OpenAI-format messages from :func:`_convert_messages`.
        tools: tool definitions rendered alongside the messages.
        seq_length: the token budget the rendered dialogue must fit.

    Returns:
        ``messages`` unchanged when it already fits or has no ``user`` boundary
        to cut at; otherwise the largest suffix of exchanges (leading ``system``
        messages prepended) that fits. When even ``system`` + the final exchange
        overflows, that minimal suffix is returned and the caller's token-level
        ``truncation`` (if enabled) clips the remainder.
    """
    if _tokenized_chat_length(tokenizer, messages, tools=tools) <= seq_length:
        return messages

    n_system = 0
    while n_system < len(messages) and messages[n_system].get("role") == "system":
        n_system += 1
    system = messages[:n_system]
    history = messages[n_system:]

    # Each ``user`` message starts a new exchange. Re-add exchanges newest-first
    # (drop oldest) and keep the largest suffix that fits; the rendered length is
    # monotonic in the number of exchanges kept.
    boundaries = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not boundaries:
        return messages

    for b in boundaries:
        candidate = system + history[b:]
        if _tokenized_chat_length(tokenizer, candidate, tools=tools) <= seq_length:
            return candidate

    # Even the final exchange alone overflows; return it and let token-level
    # truncation (if enabled) clip the remainder.
    return system + history[boundaries[-1] :]


def _format_example(
    example: Dict[str, Any],
    tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = False,
    truncation: Union[str, bool] = False,
    mask_reasoning_content: bool = False,
    train_on_last_turn_only: bool = False,
    drop_history_reasoning_content: bool = False,
    truncate_history: bool = False,
) -> Dict[str, List[int]]:
    """Render one agent example into tokenized ``input_ids`` / ``labels``.

    Thin wrapper that re-raises any parsing/rendering failure as a ``ValueError``
    tagged with the example id. Rows are rendered lazily inside the dataloader,
    so without this a single malformed row surfaces as an opaque
    ``JSONDecodeError``/``AssertionError`` deep in the stack — with no hint as to
    which row caused it — and aborts the whole training run.
    """
    try:
        return _format_example_impl(
            example,
            tokenizer,
            eos_token_id,
            pad_token_id,
            seq_length=seq_length,
            padding=padding,
            truncation=truncation,
            mask_reasoning_content=mask_reasoning_content,
            train_on_last_turn_only=train_on_last_turn_only,
            drop_history_reasoning_content=drop_history_reasoning_content,
            truncate_history=truncate_history,
        )
    except Exception as e:
        raise ValueError(f"Failed to format agent SFT example (id={example.get('id')!r}): {e}") from e


def _format_example_impl(
    example: Dict[str, Any],
    tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = False,
    truncation: Union[str, bool] = False,
    mask_reasoning_content: bool = False,
    train_on_last_turn_only: bool = False,
    drop_history_reasoning_content: bool = False,
    truncate_history: bool = False,
) -> Dict[str, List[int]]:
    """Render one agent example into tokenized ``input_ids`` / ``labels``."""
    raw_tools = example.get("tools")
    if isinstance(raw_tools, str) and not raw_tools.strip():
        raw_tools = None
    tools = _json_load_if_str(raw_tools)
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"`tools` must be a list or JSON-encoded list, got {type(tools).__name__}")
    if tools is not None and len(tools) == 0:
        tools = None

    raw_messages = example.get("messages")
    if raw_messages is None:
        sharegpt = example.get("conversations")
        if sharegpt is None:
            raise ValueError("Example missing both `messages` and `conversations` fields")
        if not isinstance(sharegpt, list):
            raise ValueError(f"`conversations` must be a list, got {type(sharegpt).__name__}")
        raw_messages = _sharegpt_to_chatml(sharegpt)
    elif not isinstance(raw_messages, list):
        raise ValueError(f"`messages` must be a list, got {type(raw_messages).__name__}")

    formatted = _convert_messages(
        raw_messages,
        example_id=example.get("id"),
        drop_history_reasoning_content=drop_history_reasoning_content,
    )

    if truncate_history and seq_length is not None:
        formatted = _truncate_messages_to_fit(tokenizer, formatted, tools, seq_length)

    tokenized = format_chat_template(
        tokenizer=tokenizer,
        formatted_text=formatted,
        tools=tools,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        answer_only_loss_mask=True,
        mask_reasoning_content=mask_reasoning_content,
        train_on_last_turn_only=train_on_last_turn_only,
    )
    # Truncation (or over-aggressive last-turn masking) can leave a sample with no
    # supervised tokens at all — every label is ``ignore_index`` (-100). A single
    # such sample is harmless: the loss normalizes by the batch's supervised-token
    # count, so it just contributes nothing (a NaN only arises if a *whole* step is
    # empty, which signals a misconfigured seq_length/truncation). Rather than
    # hard-failing the run on one example, warn (with the id) and let it through — it
    # is already effectively skipped from the gradient, and the map-style
    # dataset/collator cannot drop a row in place at this stage.
    labels = tokenized.get("labels")
    if labels is not None and all(label == -100 for label in labels):
        logger.warning(
            "Agent SFT example (id=%r) has no supervised tokens (all labels masked); it "
            "contributes nothing to the loss. This usually means truncation dropped every "
            "assistant turn — increase seq_length or disable truncation.",
            example.get("id"),
        )
    return tokenized


def _extract_eval_samples_from_example(
    example: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract one eval sample per assistant tool-call position.

    For each assistant turn in ``example`` that issues tool_calls, emit
    a sample whose ``prompt_messages`` are all messages strictly before
    that turn and whose ``gt_tool_calls`` are the tool_calls from that
    turn. This lets the evaluator measure tool-call accuracy at every
    position the model is expected to act, not just the first one.

    Tool-call arguments are normalized from JSON-encoded strings (as
    produced by :func:`_convert_messages`) back to dicts so callers can
    compare against parser output directly.

    Args:
        example: a raw row from the agent SFT dataset (chatml ``messages``
            or ShareGPT ``conversations`` schema, with a ``tools`` field).

    Returns:
        A list of eval samples, each with keys ``prompt_messages``,
        ``tools``, ``gt_tool_calls``, ``example_id``, ``turn_index``.
        Returns ``[]`` if the example contains no tool-call turns.
    """
    raw_tools = example.get("tools")
    if isinstance(raw_tools, str) and not raw_tools.strip():
        raw_tools = None
    tools = _json_load_if_str(raw_tools)
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"`tools` must be a list or JSON-encoded list, got {type(tools).__name__}")
    if tools is not None and len(tools) == 0:
        tools = None

    raw_messages = example.get("messages")
    if raw_messages is None:
        sharegpt = example.get("conversations")
        if sharegpt is None:
            return []
        if not isinstance(sharegpt, list):
            raise ValueError(f"`conversations` must be a list, got {type(sharegpt).__name__}")
        raw_messages = _sharegpt_to_chatml(sharegpt)
    elif not isinstance(raw_messages, list):
        raise ValueError(f"`messages` must be a list, got {type(raw_messages).__name__}")

    example_id = example.get("id")
    formatted = _convert_messages(raw_messages, example_id=example_id)

    samples: List[Dict[str, Any]] = []
    turn_index = 0
    for idx, msg in enumerate(formatted):
        tool_calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if not tool_calls:
            continue

        gt_tool_calls: List[Dict[str, Any]] = []
        for call in tool_calls:
            fn = call.get("function") or {}
            args_raw = fn.get("arguments", "")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            gt_tool_calls.append({"name": fn.get("name"), "arguments": args})

        # ``prompt_messages`` is everything strictly before this assistant
        # turn. If that turn also carries text content, the evaluator will
        # ask the model to generate from the point right before this turn,
        # which is the natural inference contract.
        prompt_messages = formatted[:idx]
        samples.append(
            {
                "prompt_messages": prompt_messages,
                "tools": tools,
                "gt_tool_calls": gt_tool_calls,
                "example_id": example_id,
                "turn_index": turn_index,
            }
        )
        turn_index += 1

    return samples


def make_agent_chat_eval_samples(
    *,
    dataset_name: Optional[str] = None,
    path: Optional[Union[str, List[str]]] = None,
    split: str = "train",
    limit_dataset_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a flat list of tool-call eval samples from an agent SFT dataset.

    Each dialogue is expanded into one sample per assistant tool-call
    position via :func:`_extract_eval_samples_from_example`. The result
    is a plain list (not a HuggingFace ``Dataset``) because evaluation
    iterates linearly and needs the raw structured fields, not tokenized
    tensors.

    Exactly one of ``dataset_name`` or ``path`` must be provided.

    Args:
        dataset_name: HF Hub dataset id, e.g. ``llamafactory/glaive_toolcall_en``.
        path: Local JSON/JSONL file path or list of paths.
        split: Dataset split (only used with ``dataset_name``).
        limit_dataset_samples: If set, read only the first N dialogues
            before expansion. Useful to bound evaluation cost.
        max_eval_samples: If set, cap the total expanded sample count.

    Returns:
        A list of dicts with keys ``prompt_messages``, ``tools``,
        ``gt_tool_calls``, ``example_id``, ``turn_index``.
    """
    if (dataset_name is None) == (path is None):
        raise ValueError("Exactly one of `dataset_name` or `path` must be provided")

    if dataset_name is not None:
        if limit_dataset_samples is not None:
            assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
            if "[" not in split:
                split = f"{split}[:{limit_dataset_samples}]"
            else:
                logger.warning(
                    "Dataset split %s already contains slice, skipping limit_dataset_samples",
                    split,
                )
        dataset = load_dataset(dataset_name, split=split)
    else:
        data_files = [str(p) for p in (path if isinstance(path, list) else [path])]
        dataset = load_dataset("json", data_files=data_files, split="train")
        if limit_dataset_samples is not None:
            assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
            dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    samples: List[Dict[str, Any]] = []
    for example in dataset:
        samples.extend(_extract_eval_samples_from_example(example))
        if max_eval_samples is not None and len(samples) >= max_eval_samples:
            samples = samples[:max_eval_samples]
            break
    return samples


def make_agent_chat_dataset(
    tokenizer,
    *,
    dataset_name: Optional[str] = None,
    path: Optional[Union[str, List[str]]] = None,
    split: str = "train",
    seq_length: Optional[int] = None,
    limit_dataset_samples: Optional[int] = None,
    padding: Union[str, bool] = False,
    truncation: Union[str, bool] = False,
    mask_reasoning_content: bool = False,
    train_on_last_turn_only: bool = False,
    drop_history_reasoning_content: bool = False,
    truncate_history: bool = False,
) -> LazyMappedDataset:
    """Load a multi-turn function-calling SFT dataset.

    Exactly one of ``dataset_name`` (HuggingFace Hub id) or ``path`` (local
    JSON/JSONL file or list of files) must be provided. The loaded examples
    are lazily rendered through the tokenizer's chat template; tool/user
    tokens are excluded from the loss via ``answer_only_loss_mask=True``.

    Args:
        tokenizer: HuggingFace tokenizer with a chat template.
        dataset_name: HF Hub dataset id, e.g. ``llamafactory/glaive_toolcall_en``.
        path: Local JSON/JSONL file path or list of paths.
        split: Dataset split (only used with ``dataset_name``).
        seq_length: Optional max sequence length for the tokenizer.
        limit_dataset_samples: If set, keep only the first N examples.
        padding: Padding strategy forwarded to the tokenizer.
        truncation: Truncation strategy forwarded to the tokenizer.
        mask_reasoning_content: If True, exclude assistant ``reasoning_content``
            (thinking) tokens from the loss while still rendering them into the
            prompt. Requires a chat template that emits ``reasoning_content``.
            Defaults to False, which trains on reasoning tokens like any other
            assistant content.
        train_on_last_turn_only: If True, supervise only the final assistant
            turn of each dialogue (``mask_history``); all earlier assistant
            turns are excluded from the loss. Defaults to False, which
            supervises every assistant turn.
        drop_history_reasoning_content: If True, strip ``reasoning_content``
            from all but the final assistant turn so historical thinking is not
            rendered into the prompt (matching inference, where prior-turn
            thinking is not visible). Orthogonal to ``mask_reasoning_content``:
            this controls whether history thinking appears in the prompt at all,
            the latter controls whether rendered thinking contributes to loss.
            Defaults to False, which keeps every turn's reasoning_content.
        truncate_history: If True and ``seq_length`` is set, drop the oldest
            conversation exchanges (keeping any leading system message, the tool
            definitions, and the final exchange) until the dialogue fits
            ``seq_length``. Unlike token-level ``truncation``, which clips from
            the tokenizer side and usually drops the final assistant answer,
            this preserves the supervised target. Defaults to False.

    Returns:
        A ``LazyMappedDataset`` yielding dicts with ``input_ids``, ``labels``
        and ``attention_mask`` ready for the trainer collator.
    """
    if (dataset_name is None) == (path is None):
        raise ValueError("Exactly one of `dataset_name` or `path` must be provided")

    # ``seq_length`` is forwarded to ``apply_chat_template(max_length=...)``, which
    # the tokenizer only honors when truncation is enabled (to cap the length) or
    # when ``padding="max_length"`` (to pad up to it). With the defaults
    # (``truncation=False``, ``padding=False``) ``max_length`` is ignored, so
    # ``seq_length`` silently has no effect and over-long dialogues pass through
    # uncapped. Warn rather than silently dropping tokens by flipping truncation on.
    truncation_active = bool(truncation) and truncation != "do_not_truncate"
    if seq_length is not None and not truncation_active and padding != "max_length":
        logger.warning(
            "`seq_length=%s` has no effect: truncation is disabled and `padding` is not "
            "'max_length', so the tokenizer ignores `max_length` and dialogues are not "
            "capped. Set `truncation=True` to cap long dialogues, or `padding='max_length'` "
            "to pad to `seq_length`.",
            seq_length,
        )

    if dataset_name is not None:
        if limit_dataset_samples is not None:
            assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
            if "[" not in split:
                split = f"{split}[:{limit_dataset_samples}]"
            else:
                logger.warning("Dataset split %s already contains slice, skipping limit_dataset_samples", split)
        dataset = load_dataset(dataset_name, split=split)
    else:
        data_files = [str(p) for p in (path if isinstance(path, list) else [path])]
        dataset = load_dataset("json", data_files=data_files, split="train")
        if limit_dataset_samples is not None:
            assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
            dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id

    fmt_fn = lambda x: _format_example(  # noqa: E731
        x,
        tokenizer,
        eos_token_id,
        pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        mask_reasoning_content=mask_reasoning_content,
        train_on_last_turn_only=train_on_last_turn_only,
        drop_history_reasoning_content=drop_history_reasoning_content,
        truncate_history=truncate_history,
    )
    return LazyMappedDataset(dataset, fmt_fn)


@dataclass
class AgentChatConfig:
    """Construction-time configuration for the agent chat dataset (tokenizer is a build arg).

    Exactly one of ``dataset_name`` or ``path`` must be set when calling :func:`build`.
    """

    dataset_name: Optional[str] = None
    """HF Hub dataset id, e.g. ``llamafactory/glaive_toolcall_en``."""
    path: Optional[Union[str, List[str]]] = None
    """Local JSON/JSONL file path or list of paths."""
    split: str = "train"
    """Dataset split (only used with ``dataset_name``)."""
    seq_length: Optional[int] = None
    """Optional max sequence length for the tokenizer."""
    limit_dataset_samples: Optional[int] = None
    """If set, keep only the first N examples."""
    padding: Union[str, bool] = False
    """Padding strategy forwarded to the tokenizer."""
    truncation: Union[str, bool] = False
    """Truncation strategy forwarded to the tokenizer."""
    mask_reasoning_content: bool = False
    """If True, exclude assistant ``reasoning_content`` tokens from the loss."""
    train_on_last_turn_only: bool = False
    """If True, supervise only the final assistant turn of each dialogue."""
    drop_history_reasoning_content: bool = False
    """If True, strip ``reasoning_content`` from all but the final assistant turn."""

    def build(self, *, tokenizer) -> LazyMappedDataset:
        """Build the agent chat :class:`LazyMappedDataset` from this :class:`AgentChatConfig` and a runtime tokenizer."""
        return make_agent_chat_dataset(
            tokenizer,
            dataset_name=self.dataset_name,
            path=self.path,
            split=self.split,
            seq_length=self.seq_length,
            limit_dataset_samples=self.limit_dataset_samples,
            padding=self.padding,
            truncation=self.truncation,
            mask_reasoning_content=self.mask_reasoning_content,
            train_on_last_turn_only=self.train_on_last_turn_only,
            drop_history_reasoning_content=self.drop_history_reasoning_content,
        )
