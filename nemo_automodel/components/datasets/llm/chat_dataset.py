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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import VerificationMode, load_dataset
from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _has_chat_template,
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


def _load_openai_messages(
    path_or_dataset_id: Union[str, Sequence[str]], split: Optional[str] = None, name: Optional[str] = None
):
    """Load OpenAI chat messages datasets from HF or local JSON/JSONL files.

    For HF repo IDs, we delegate to datasets.load_dataset.
    For local files, we manually parse JSONL/JSON to avoid pyarrow type
    inference issues (e.g., heterogeneous field types under `tools`).

    Args:
        path_or_dataset_id: HF dataset ID or local file path(s).
        split: Dataset split to load (e.g., "train", "validation").
        name: Dataset configuration/subset name
    """
    if isinstance(path_or_dataset_id, str) and _is_hf_repo_id(path_or_dataset_id):
        return load_dataset(
            path_or_dataset_id, name=name, split=split, streaming=False, verification_mode=VerificationMode.NO_CHECKS
        )

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
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
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
    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        out = dict(m)
        if isinstance(content, list):
            out["content"] = str(content["text"])
        else:
            out["content"] = str(content)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role in messages: {role}")
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
    ) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        # Enforce chat-template availability for tool-calling data
        if chat_template is not None:
            # Allow overriding the tokenizer's template
            tokenizer.chat_template = chat_template

        if not _has_chat_template(tokenizer):
            raise ValueError("ChatDataset requires a tokenizer with chat template support.")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation
        self.start_of_turn_token = start_of_turn_token

        self.dataset = _load_openai_messages(path_or_dataset_id, split=split, name=name)

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
        )
        return sample
