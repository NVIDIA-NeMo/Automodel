# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, Iterator, List, Optional, Union

from torch.utils.data import IterableDataset

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
    ColumnTypes,
    _check_all_values_equal_length,
    _load_dataset,
)

logger = logging.getLogger(__name__)


class ColumnMappedTextInstructionIterableDataset(IterableDataset, ColumnMappedTextInstructionDataset):
    """Streaming iterable variant that reuses the column-mapping/tokenization logic.

    This wraps a Hugging Face streaming dataset (IterableDataset from `datasets`)
    and yields tokenized samples compatible with the non-streaming variant, while
    supporting sharding and epoch-setting for deterministic shuffles upstream.
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, List[str]],
        column_mapping: Dict[str, str],
        tokenizer,
        *,
        split: Optional[str] = None,
        name: Optional[str] = None,
        answer_only_loss_mask: bool = True,
        seq_length: Optional[int] = None,
        padding: Union[str, bool] = "do_not_pad",
        truncation: Union[str, bool] = "do_not_truncate",
        start_of_turn_token: Optional[str] = None,
        limit_dataset_samples: Optional[int] = None,
        repeat_on_exhaustion: bool = True,
        use_hf_chat_template: bool = False,
    ) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer is required")
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token", None) is None:
            if hasattr(self.tokenizer, "eos_token"):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                logger.warning("Setting tokenizer pad_token to ' '. tokenizer does not have `eos_token`.")
                self.tokenizer.pad_token = " "

        if ColumnTypes.Answer.value not in column_mapping:
            raise AssertionError(("Expected answer to be in column_mapping", column_mapping))
        if len(column_mapping) == 3:
            if ColumnTypes.Context.value not in column_mapping:
                raise AssertionError(("Expected context to be in column_mapping", column_mapping))
            if ColumnTypes.Question.value not in column_mapping:
                raise AssertionError(("Expected question to be in column_mapping", column_mapping))
        elif len(column_mapping) == 2:
            if ColumnTypes.Context.value not in column_mapping and ColumnTypes.Question.value not in column_mapping:
                raise AssertionError(("Expected context or question to be in column_mapping", column_mapping))
        else:
            raise ValueError(f"Expected 2 or 3 columns in column_mapping, got {len(column_mapping)}")

        self.column_mapping = column_mapping
        self.answer_only_loss_mask = answer_only_loss_mask
        self.start_of_turn_token = start_of_turn_token
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation
        self.use_hf_chat_template = use_hf_chat_template
        self.num_shards = getattr(self, "num_shards", 1)
        self._current_epoch_for_repeat = 0
        self.repeat_on_exhaustion = bool(repeat_on_exhaustion)

        # Always load in streaming mode
        ds = _load_dataset(path_or_dataset_id, split=split, streaming=True, name=name)
        if limit_dataset_samples is not None:
            try:
                ds = ds.take(limit_dataset_samples)
            except Exception as e:
                logger.warning("limit_dataset_samples ignored; 'take' not supported on this dataset: %s", e)

        self.dataset = ds

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        while True:
            for row in self.dataset:
                mapped = {dest: row[src] for dest, src in self.column_mapping.items() if src in row}
                # Skip rows missing required fields
                if ColumnTypes.Answer.value not in mapped:
                    continue
                tokenized = self._apply_tokenizer(mapped)  # provided by ColumnMappedTextInstructionDataset
                # Skip samples with no valid labels (aligns with non-iterable behavior)
                if not any(label != -100 for label in tokenized.get("labels", [])):
                    continue
                if not _check_all_values_equal_length(tokenized):
                    continue
                yield tokenized

            if not self.repeat_on_exhaustion:
                return
            # Wrap-around: advance epoch for deterministic reshuffle if supported and iterate again
            try:
                self._current_epoch_for_repeat += 1
                self.set_epoch(self._current_epoch_for_repeat)
            except Exception:
                pass

    def set_epoch(self, epoch: int) -> None:
        ds = getattr(self, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

    def shard(self, num_shards: int, index: int):
        ds = getattr(self, "dataset", None)
        if ds is not None and hasattr(ds, "shard"):
            self.dataset = ds.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int, seed: int):
        ds = getattr(self, "dataset", None)
        if ds is not None and hasattr(ds, "shuffle"):
            self.dataset = ds.shuffle(buffer_size=buffer_size, seed=seed)
        return self
