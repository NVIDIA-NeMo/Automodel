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

from torch.utils.data import Dataset
from typing import Dict, Any

# Supported cases:
# Format:
# - Context + question + answer
# - Question + answer
# Input types:
# - one or more paths to jsonl files
# - dataset id from huggingface.

class ColumnTypes
    Context = "context"
    Question = "question"
    Answer = "answer"

def listify(val: Union[str, List[str]]) -> List[str]:
    """
    Convert a single string or list of strings to a list.

    Args:
        val: A single string or list of strings.

    Returns:
        A list of strings.
    """
    if isinstance(val, str):
        return [val]
    elif isinstance(val, list):
        return val
    else:
        raise ValueError(f"Invalid input type: {type(val)}")

def load_dataset(path_or_dataset_id: Union[str, List[str]], split: Optional[str] = None):
    """
    Load a dataset from a single path or a list of paths.

    Args:
        path_or_dataset_id: A single path or a list of paths to jsonl files.
        split: The split to load from the dataset.

    Returns:
        A dataset.
    """
    if isinstance(path_or_dataset_id, (str, list)):
        return Dataset.from_list(json.load(open(path)) for path in listify(path_or_dataset_id))
    else:
        raise ValueError(f"Invalid input type: {type(path_or_dataset_id)}")

class ColumnMappedTextDataset(Dataset):
    def __init__(self, path_or_dataset_id: Union[str, List[str]], column_mapping: Dict[str, str], tokenizer, split: Optional[str] = None):
        """
        Initialize a column mapped text dataset.

        Args:
            path_or_dataset_id: A single path or a list of paths to jsonl files.
            column_mapping: A dictionary mapping the column names to the column indices.
            split: The split to load from the dataset.
        """
        self.dataset = load_dataset(path_or_dataset_id)[split]
        self.column_mapping = column_mapping
        self.tokenizer = tokenizer

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