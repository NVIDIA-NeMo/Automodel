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
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset
from enum import Enum
import re

# Supported cases:
# Format:
# - Context + question + answer
# - Question + answer
# Input types:
# - one or more paths to jsonl files
# - dataset id from huggingface.

class ColumnTypes(Enum):
    Context = "context"
    Question = "question"
    Answer = "answer"

def make_iterable(val: Union[str, List[str]]) -> Iterator[str]:
    """Utility that converts *val* into an iterator of strings.

    The helper accepts either a single string or a list of strings and
    yields its contents. This is handy when we want to treat the two cases
    uniformly downstream (e.g. when iterating over *data_files* that can be
    provided as either a single path or a collection of paths).

    Args:
        val: Either a single string or a list/tuple of strings.

    Yields:
        str: The individual strings contained in *val*.

    Raises:
        ValueError: If *val* is neither a string nor an iterable of strings.
    """
    if isinstance(val, str):
        yield val
    elif isinstance(val, (list, tuple)):
        for item in val:
            if not isinstance(item, str):
                raise ValueError("All elements must be strings")
            yield item
    else:
        raise ValueError(f"Expected str or list[str], got {type(val)}")

def _str_is_hf_repo_id(val: str) -> bool:
    """
    Check if a string is a valid huggingface dataset id.

    Args:
        val: A string to check.

    Returns:
        True if the string is a valid huggingface dataset id, False otherwise.
    """
    return val.count('/') == 1 \
        and re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$', val) is not None \
        and not Path(val).exists()


def _load_dataset(path_or_dataset_id: Union[str, List[str]], split: Optional[str] = None):
    """Load a dataset either from the Hugging Face Hub or from local JSON/JSONL files.

    If *path_or_dataset_id* resembles a HF repo ID (i.e. of the form
    ``org/dataset`` and the path does **not** exist on the local filesystem),
    we defer to ``datasets.load_dataset`` directly. Otherwise, we assume the
    argument points to one or more local JSON/JSONL files and let
    ``datasets.load_dataset`` with the *"json"* script handle the parsing.

    Args:
        path_or_dataset_id: Either a HF dataset identifier (``org/name``) or
            a path / list of paths to local ``.json`` / ``.jsonl`` files.
        split: Optional split to load when retrieving a remote dataset. This
            parameter is ignored for local files as the *json* script always
            returns a single split.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    if isinstance(path_or_dataset_id, str) and _str_is_hf_repo_id(path_or_dataset_id):
        return load_dataset(path_or_dataset_id, split=split or "train")

    data_files = list(make_iterable(path_or_dataset_id))
    if not data_files:
        raise ValueError("No data files provided")

    return load_dataset("json", data_files=data_files, split="train")

class ColumnMappedTextInstructionDataset(Dataset):
    """Generic *instruction‐tuning* dataset that maps arbitrary column names.

    The class is intentionally lightweight: it simply loads the raw samples
    (either from HF or from local JSON/JSONL files) and remaps the columns so
    that downstream components can rely on a consistent field interface.

    Optionally, if *answer_only_loss_mask* is requested, the dataset will also
    compute a *loss_mask* indicating which tokens should contribute to the
    loss (typically only those belonging to the assistant answer).
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, List[str]],
        column_mapping: Dict[str, str],
        tokenizer,
        *,
        split: Optional[str] = None,
        answer_only_loss_mask: bool = True,
        start_of_turn_token: Optional[str] = None,
    ) -> None:
        if answer_only_loss_mask and start_of_turn_token is None:
            raise ValueError(
                "start_of_turn_token must be provided when answer_only_loss_mask=True"
            )

        self.tokenizer = tokenizer

        self.dataset = _load_dataset(path_or_dataset_id, split=split)

        # Keep mapping: dest -> source (i.e. public_field -> raw_column_name)
        self.column_mapping = column_mapping

        self.answer_only_loss_mask = answer_only_loss_mask
        self.start_of_turn_token = start_of_turn_token

    def __len__(self) -> int:  # noqa: D401
        return len(self.dataset)

    def __getitem__(self, idx):  # noqa: D401
        row = self.dataset[idx]
        mapped = {dest: row[src] for dest, src in self.column_mapping.items()}
        return mapped