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
    """
    Convert a single string or list to a string iterator.

    Args:
        val: A single string or list of strings.

    Returns:
        An iterator over the strings.
    """
    if isinstance(val, str):
        yield val
    elif isinstance(val, list):
        yield from val
    else:
        raise ValueError(f"Invalid input type: {type(val)}")

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


def _load_dataset(path_or_dataset_id: Union[str, List[str]]):
    """
    Load a dataset from a single path or a list of paths.

    Args:
        path_or_dataset_id: A single path or a list of paths to jsonl files.

    Returns:
        A dataset.
    """
    if isinstance(path_or_dataset_id, str) and _str_is_hf_repo_id(path_or_dataset_id):
        return load_dataset(path_or_dataset_id)
    if isinstance(path_or_dataset_id, (str, list)):
        return Dataset.from_list(json.load(open(path)) for path in make_iterable(path_or_dataset_id))
    else:
        raise ValueError(f"Invalid input type: {type(path_or_dataset_id)}")

class ColumnMappedTextInstructionDataset(Dataset):
    def __init__(
            self,
            path_or_dataset_id: Union[str, List[str]],
            column_mapping: Dict[str, str],
            tokenizer,
            split: Optional[str] = None,
            answer_only_loss_mask: bool = True,
            start_of_turn_token: Optional[str] = None
        ):
        """
        Initialize a column mapped text dataset.

        Args:
            path_or_dataset_id: A single path or a list of paths to jsonl files.
            column_mapping: A dictionary mapping the column names to the column indices.
            tokenizer: A tokenizer.
            split: The split to load from the dataset.
            answer_only_loss_mask: Whether to use only the answer column for the loss mask.
            start_of_turn_token: The string to use as the start of turn token.

        Returns:
            Instance of the ColumnMappedTextInstructionDataset class.
        """
        self.dataset = _load_dataset(path_or_dataset_id)
        if split:
            self.dataset = self.dataset[split]
        self.column_mapping = column_mapping
        self.tokenizer = tokenizer
        if answer_only_loss_mask and start_of_turn_token_id is None:
            raise ValueError("start_of_turn_token_id is required when answer_only_loss_mask is True")
        self.answer_only_loss_mask = answer_only_loss_mask
        self.start_of_turn_token = start_of_turn_token

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: The index of the item to get.

        Returns:
            A dictionary with the mapped columns.
        """
        return {k: v[idx] for k, v in self.column_mapping.items()}