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

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import VerificationMode, load_dataset
from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _has_chat_template,
    _resolve_chat_template,
    format_chat_template,
)


def _is_hf_repo_id(val: str) -> bool:
    # Basic check: org/name without local path existing
    if "/" not in val:
        return False
    p = Path(val)
    return not p.exists() and all(part for part in val.split("/"))


def _as_iter(val: Union[str, Sequence[str]]) -> Iterator[str]:
    if isinstance(val, str):
        yield val
    else:
        for x in val:
            if not isinstance(x, str):
                raise ValueError("data_files entries must be strings")
            yield x


_SPLIT_SLICE_RE = re.compile(r"^(\w+)\[(\d*):(\d*)\]$")


def _parse_split_slice(split: Optional[str]):
    """Parse a split string like ``"train[1024:]"`` into ``(base_split, slice | None)``."""
    if split is None:
        return split, None
    match = _SPLIT_SLICE_RE.match(split)
    if not match:
        return split, None
    base = match.group(1)
    start = int(match.group(2)) if match.group(2) else None
    end = int(match.group(3)) if match.group(3) else None
    return base, slice(start, end)


def _load_openai_messages(
    path_or_dataset_id: Union[str, Sequence[str]],
    split: Optional[str] = None,
    name: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    skip_invalid_samples: bool = False,
):
    """Load OpenAI chat messages datasets from HF or local JSON/JSONL files.

    For HF repo IDs, we delegate to datasets.load_dataset.  When *split*
    is provided, the full base split is loaded and shuffled *before* any
    slice (e.g. ``[1024:]``) is applied so that train/val splits sample
    from a consistent random order.  When *split* is ``None`` it is passed
    through to ``load_dataset`` as-is (no default override).

    For local files, we manually parse JSONL/JSON to avoid pyarrow type
    inference issues (e.g., heterogeneous field types under `tools`).

    Args:
        path_or_dataset_id: HF dataset ID or local file path(s).
        split: Dataset split to load (e.g., "train", "train[1024:]").
        name: Dataset configuration/subset name
        shuffle_seed: Random seed for shuffling HF datasets before slicing.
            Set to ``None`` to disable shuffling.
        skip_invalid_samples: If ``True``, skip malformed JSONL lines for local
            files instead of failing fast.
    """
    if isinstance(path_or_dataset_id, str) and _is_hf_repo_id(path_or_dataset_id):
        base_split, sl = _parse_split_slice(split)

        dataset = load_dataset(
            path_or_dataset_id,
            name=name,
            split=base_split,
            streaming=False,
            verification_mode=VerificationMode.NO_CHECKS,
        )
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)

        if sl is not None:
            indices = range(*sl.indices(len(dataset)))
            dataset = dataset.select(indices)

        return dataset

    # Handle local directories and Parquet files via load_dataset.
    # This covers pre-filtered cached datasets saved as Parquet.
    if isinstance(path_or_dataset_id, str):
        p = Path(path_or_dataset_id)
        is_parquet_file = p.is_file() and p.suffix.lower() == ".parquet"
        is_dataset_dir = p.is_dir() and any(p.glob("*.parquet"))

        if is_parquet_file or is_dataset_dir:
            logging.getLogger(__name__).info("Loading local dataset from %s via load_dataset", path_or_dataset_id)
            base_split, sl = _parse_split_slice(split)

            load_path = str(p.parent) if is_parquet_file else str(p)
            # Cached Parquet datasets (from prefilter_dataset.py) are saved as a single
            # split. Default to "train" when split is unspecified or was stripped to
            # extract a slice (e.g. "train[:128]" → base_split="train", sl=slice(None,128)).
            dataset = load_dataset(
                load_path,
                name=name,
                split=base_split or "train",
                data_files=p.name if is_parquet_file else None,
                verification_mode=VerificationMode.NO_CHECKS,
            )

            if shuffle_seed is not None:
                dataset = dataset.shuffle(seed=shuffle_seed)
            if sl is not None:
                indices = range(*sl.indices(len(dataset)))
                dataset = dataset.select(indices)
            return dataset

    # Fall back to manual JSON/JSONL parsing for local files.
    files = list(_as_iter(path_or_dataset_id))
    if not files:
        raise RuntimeError("No data files provided")

    rows: List[Dict[str, Any]] = []

    def _read_file(fp: str) -> None:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".jsonl", ".ndjson"}:
            skipped_lines = 0
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if not skip_invalid_samples:
                        raise
                    skipped_lines += 1
            if skipped_lines:
                logging.getLogger(__name__).warning(
                    "Skipped %d malformed JSONL line(s) from %s (skip_invalid_samples=True)",
                    skipped_lines,
                    fp,
                )
        else:
            obj = json.loads(text)
            if isinstance(obj, list):
                rows.extend(obj)
            else:
                rows.append(obj)

    for f in files:
        _read_file(f)

    return rows


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure messages list is valid and content fields are strings for system/user/assistant.

    - Keeps tool_calling fields if present (e.g., tool calls in assistant messages, tool role messages).
    - If content is a list of parts, only keep text parts.
    """

    def _normalize_content(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(part["text"] for part in value if isinstance(part, dict) and "text" in part)
        if value is None:
            return ""
        return str(value)

    def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls, list):
            raise ValueError("assistant message `tool_calls` must be a list")

        normalized_tool_calls: List[Dict[str, Any]] = []
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise ValueError(f"assistant message `tool_calls[{idx}]` must be a dict")

            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError(f"assistant message `tool_calls[{idx}].id` must be a non-empty string")

            tool_call_type = tool_call.get("type")
            if not isinstance(tool_call_type, str) or not tool_call_type:
                raise ValueError(f"assistant message `tool_calls[{idx}].type` must be a non-empty string")

            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise ValueError(f"assistant message `tool_calls[{idx}].function` must be a dict")

            function_name = function.get("name")
            if not isinstance(function_name, str) or not function_name:
                raise ValueError(f"assistant message `tool_calls[{idx}].function.name` must be a non-empty string")

            function_arguments = function.get("arguments")
            if function_arguments is None:
                raise ValueError(f"assistant message `tool_calls[{idx}].function.arguments` is required")

            normalized_function = dict(function)
            if not isinstance(function_arguments, str):
                normalized_function["arguments"] = json.dumps(function_arguments)

            normalized_tool_call = dict(tool_call)
            normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)

        return normalized_tool_calls

    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        out = dict(m)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role in messages: {role}")

        out["content"] = _normalize_content(m.get("content"))

        if role == "assistant":
            if "reasoning_content" in m:
                reasoning_content = m.get("reasoning_content")
                if reasoning_content is None:
                    out["reasoning_content"] = ""
                else:
                    if not isinstance(reasoning_content, str):
                        raise ValueError("assistant message `reasoning_content` must be a string when provided")
                    out["reasoning_content"] = reasoning_content
            if "tool_calls" in m:
                out["tool_calls"] = _normalize_tool_calls(m.get("tool_calls"))

        if role == "tool":
            tool_call_id = m.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("tool message `tool_call_id` must be a non-empty string")

        norm.append(out)
    return norm


class ChatDataset(Dataset):
    """Dataset for OpenAI-format tool-calling chat transcripts.

    This class expects each row to contain a `messages` list in OpenAI chat format,
    potentially including tool calls and tool responses. The datasetformats the
    conversation via the tokenizer's chat template to produce `input_ids`, `labels`,
    and `attention_mask` suitable for SFT.
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, Sequence[str]],
        tokenizer,
        *,
        split: Optional[str] = None,
        name: Optional[str] = None,
        seq_length: Optional[int] = None,
        padding: Union[str, bool] = "do_not_pad",
        truncation: Union[str, bool] = "do_not_truncate",
        start_of_turn_token: Optional[str] = None,
        chat_template: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
        mask_reasoning_content: bool = False,
        unshifted: bool = False,
        skip_invalid_samples: bool = False,
    ) -> None:
        """Load OpenAI-format chat rows and tokenize via the chat template.

        Args:
            path_or_dataset_id: Hugging Face dataset id, local JSON/JSONL path(s), Parquet file, or Parquet directory.
            tokenizer: Tokenizer with chat template support (required).
            split: Dataset split or slice (e.g. ``train``, ``train[1024:]``).
            name: Optional Hub subset / config name.
            seq_length: Maximum sequence length for padding and truncation in formatting.
            padding: Padding mode for ``format_chat_template``.
            truncation: Truncation mode for ``format_chat_template``.
            start_of_turn_token: Optional token marking assistant turns for answer-only loss.
            chat_template: Optional Jinja template string overriding ``tokenizer.chat_template``.
            shuffle_seed: If set, shuffles Hub/Parquet data before applying a split slice.
            mask_reasoning_content: If ``True``, exclude rendered reasoning traces from the loss mask.
            unshifted: Passed through to ``format_chat_template``.
            skip_invalid_samples: If ``True``, skip malformed JSONL lines when reading local files (warning logs
                include skip counts). If ``False``, a bad line raises. Does not skip invalid structured rows after
                load; those still raise when a sample is accessed.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        # Enforce chat-template availability for tool-calling data
        if chat_template is not None:
            tokenizer.chat_template = _resolve_chat_template(chat_template)

        if not _has_chat_template(tokenizer):
            raise ValueError("ChatDataset requires a tokenizer with chat template support.")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation
        self.start_of_turn_token = start_of_turn_token
        self.mask_reasoning_content = mask_reasoning_content
        self.unshifted = unshifted
        self.skip_invalid_samples = skip_invalid_samples

        self.dataset = _load_openai_messages(
            path_or_dataset_id,
            split=split,
            name=name,
            shuffle_seed=shuffle_seed,
            skip_invalid_samples=skip_invalid_samples,
        )

        # Ensure pad token presence for downstream padding
        eos_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.pad_token_id = _add_pad_token(self.tokenizer) or eos_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.dataset[idx]
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each sample must contain a `messages` list in OpenAI format")

        normalized = _normalize_messages(messages)
        tools = row.get("tools")
        if tools is not None and not isinstance(tools, list):
            tools = None

        eos_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        sample = format_chat_template(
            self.tokenizer,
            normalized,
            eos_token_id,
            self.pad_token_id,
            seq_length=self.seq_length,
            padding=self.padding,
            truncation=self.truncation,
            tools=tools,
            mask_reasoning_content=self.mask_reasoning_content,
            unshifted=self.unshifted,
        )
        return sample
