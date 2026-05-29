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

import json
import logging
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token, format_chat_template

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

    return out


def _format_example(
    example: Dict[str, Any],
    tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    padding: Union[str, bool] = False,
    truncation: Union[str, bool] = False,
    mask_reasoning_content: bool = False,
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

    formatted = _convert_messages(raw_messages, example_id=example.get("id"))

    return format_chat_template(
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
    )


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

    Returns:
        A ``LazyMappedDataset`` yielding dicts with ``input_ids``, ``labels``
        and ``attention_mask`` ready for the trainer collator.
    """
    if (dataset_name is None) == (path is None):
        raise ValueError("Exactly one of `dataset_name` or `path` must be provided")

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
        seq_length,
        padding,
        truncation,
        mask_reasoning_content,
    )
    return LazyMappedDataset(dataset, fmt_fn)
