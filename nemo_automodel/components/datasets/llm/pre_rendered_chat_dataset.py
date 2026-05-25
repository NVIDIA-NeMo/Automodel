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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _package_tokenized_example,
)

logger = logging.getLogger(__name__)


def _load_jsonl_rows(
    paths: Union[str, Sequence[str]],
    *,
    skip_invalid_samples: bool = False,
) -> List[Dict[str, Any]]:
    if isinstance(paths, str):
        paths = [paths]
    rows: List[Dict[str, Any]] = []
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        skipped = 0
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if not skip_invalid_samples:
                        raise
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d malformed JSONL lines from %s", skipped, fp)
    return rows


def _tokenize_row(tokenizer, messages: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Tokenize all message contents, returning (input_ids, assistant_mask)."""
    input_ids: List[int] = []
    assistant_mask: List[int] = []
    for m in messages:
        content = m.get("content", "") or ""
        if not content:
            continue
        ids = tokenizer.encode(content, add_special_tokens=False)
        input_ids.extend(ids)
        mark = 1 if m.get("role") == "assistant" else 0
        assistant_mask.extend([mark] * len(ids))
    return input_ids, assistant_mask


class PreRenderedChatDataset(Dataset):
    """SFT dataset where every ``content`` string is already template-rendered.

    Each row in the JSONL has the shape::

        {"messages": [{"role": ..., "content": "<|im_start|>...<|im_end|>\\n"}, ...]}

    The chat template has already been applied to the strings, so we MUST NOT
    re-template here. Instead we tokenize each ``content`` piece independently
    (``add_special_tokens=False``), concatenate the ids into a single sequence,
    and build an assistant-only loss mask. The packaging matches ``ChatDataset``
    so the trainer sees the same keys / shapes.

    Args:
        path_or_dataset_id: Local JSONL path or list of paths.
        tokenizer: HF tokenizer matching the rendered template (e.g. one whose
            vocab contains ``<|im_start|>``, ``<|im_end|>``, ``<think>`` ...).
        seq_length: Max sequence length. Rows longer than this are truncated;
            padding behavior is controlled by ``padding``.
        padding: ``"do_not_pad"`` | ``"max_length"`` (passed to packaging).
        truncation: ``"do_not_truncate"`` | other (passed to packaging).
        skip_invalid_samples: If True, skip malformed JSONL lines.
        num_samples_limit: If set, only take the first N records from the file
            before any further filtering.
        max_total_tokens: If set, pre-tokenize every record and drop any whose
            total token count exceeds this value or that contain no assistant
            content. The surviving records are cached so __getitem__ is O(1).
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, Sequence[str]],
        tokenizer,
        *,
        seq_length: Optional[int] = 8192,
        padding: Union[str, bool] = "do_not_pad",
        truncation: Union[str, bool] = "do_not_truncate",
        skip_invalid_samples: bool = False,
        num_samples_limit: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation

        rows = _load_jsonl_rows(
            path_or_dataset_id,
            skip_invalid_samples=skip_invalid_samples,
        )
        if num_samples_limit is not None and num_samples_limit > 0:
            rows = rows[:num_samples_limit]

        eos_token_id = getattr(tokenizer, "eos_token_id", 0)
        self.eos_token_id = eos_token_id
        self.pad_token_id = _add_pad_token(tokenizer) or eos_token_id

        self._cached: Optional[List[Tuple[List[int], List[int]]]] = None
        if max_total_tokens is not None and max_total_tokens > 0:
            cached: List[Tuple[List[int], List[int]]] = []
            dropped_too_long = 0
            dropped_no_assistant = 0
            for row in rows:
                messages = row.get("messages") or []
                if not messages:
                    dropped_no_assistant += 1
                    continue
                input_ids, assistant_mask = _tokenize_row(tokenizer, messages)
                if len(input_ids) > max_total_tokens:
                    dropped_too_long += 1
                    continue
                if sum(assistant_mask) == 0:
                    dropped_no_assistant += 1
                    continue
                cached.append((input_ids, assistant_mask))
            logger.info(
                "PreRenderedChatDataset: kept %d records (dropped %d > %d tokens, "
                "%d with no assistant content) from %d input rows",
                len(cached),
                dropped_too_long,
                max_total_tokens,
                dropped_no_assistant,
                len(rows),
            )
            if not cached:
                raise RuntimeError(
                    f"All {len(rows)} records were filtered out "
                    f"(max_total_tokens={max_total_tokens}); nothing to train on."
                )
            self._cached = cached
            self.rows = None
        else:
            self.rows = rows

    def __len__(self) -> int:
        if self._cached is not None:
            return len(self._cached)
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if self._cached is not None:
            cached_ids, cached_mask = self._cached[idx]
            input_ids = list(cached_ids)
            assistant_mask = list(cached_mask)
        else:
            messages = self.rows[idx].get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Row {idx} has no `messages` list")
            input_ids, assistant_mask = _tokenize_row(self.tokenizer, messages)

        if self.seq_length and len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            assistant_mask = assistant_mask[: self.seq_length]

        return _package_tokenized_example(
            self.tokenizer,
            input_ids,
            assistant_mask,
            self.eos_token_id,
            self.pad_token_id,
            self.seq_length,
            truncation=self.truncation,
            padding=self.padding,
            unshifted=False,
        )
