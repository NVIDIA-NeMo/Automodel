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

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from datasets import load_dataset


def _as_list(path_or_dataset: str | Sequence[str]) -> list[str]:
    if isinstance(path_or_dataset, str):
        return [path_or_dataset]
    return [str(x) for x in path_or_dataset]


def _to_conversation(example: dict) -> dict:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected each sample to contain a `messages` list.")

    conversation = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue

        content = msg.get("content", "")
        if content is None:
            content = ""
        content = str(content)

        conversation.append(
            {
                "role": role,
                "content": [{"type": "text", "text": content}],
            }
        )

    if not conversation:
        raise ValueError("Conversation is empty after message normalization.")

    return {"conversation": conversation}


def make_parquet_messages_dataset(
    path_or_dataset,
    split: str = "train",
    shuffle_seed: int | None = 42,
    limit_dataset_samples: int | None = None,
    **kwargs,
):
    """Load local parquet files with a `messages` column and convert them to VLM conversations.

    The returned dataset matches the schema expected by
    `nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn`
    and `gemma4_prefix_collate_fn`, i.e. each sample is a dict with a
    `conversation` field containing OpenAI-style role/content messages rendered
    as text-only multimodal parts.
    """

    data_files = _as_list(path_or_dataset)
    missing = [p for p in data_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet file(s): {missing}")

    dataset = load_dataset("parquet", data_files=data_files, split=split)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset.map(_to_conversation, remove_columns=dataset.column_names)
