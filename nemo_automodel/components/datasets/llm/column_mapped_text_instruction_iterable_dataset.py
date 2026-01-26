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



def _load_streaming_dataset(
    path_or_dataset_id: Union[str, List[str]],
    split: Optional[str] = None,
    streaming: bool = False,
    name: Optional[str] = None,
    delta_storage_options: Optional[Dict[str, str]] = None,
    delta_version: Optional[int] = None,
    delta_sql_query: Optional[str] = None,
):
    """Load a dataset from HuggingFace Hub, local JSON/JSONL files, or Delta Lake tables.

    If *path_or_dataset_id* resembles a HF repo ID (i.e. of the form
    ``org/dataset`` and the path does **not** exist on the local filesystem),
    we defer to ``datasets.load_dataset`` directly. If the path is a Delta Lake
    table (prefixed with ``delta://``, ``dbfs:/``, or a directory containing
    ``_delta_log``), we load using the Delta Lake reader. Otherwise, we assume
    the argument points to one or more local JSON/JSONL files and let
    ``datasets.load_dataset`` with the *"json"* script handle the parsing.

    Args:
        path_or_dataset_id: Either a HF dataset identifier (``org/name``),
            a Delta Lake table path (``delta://path/to/table``), or
            a path / list of paths to local ``.json`` / ``.jsonl`` files.
        split: Optional split to load when retrieving a remote dataset. This
            parameter is ignored for local files and Delta Lake tables.
        streaming: Whether to stream the dataset.
        name: Optional name of the dataset configuration/subset to load
        delta_storage_options: Optional dict of storage options for Delta Lake
            cloud authentication (e.g., ``{"DATABRICKS_TOKEN": "dapi..."}``)
        delta_version: Optional specific version of the Delta table to read.
        delta_sql_query: Optional SQL query to execute against the Delta Lake source.
            This is supported when running with a SparkSession (Databricks / pyspark)
            or when using the Databricks SQL Connector. The query must return the
            columns expected by `column_mapping`.

    Returns:
        datasets.Dataset: The loaded dataset.

    Examples:
        >>> # Load from HuggingFace Hub
        >>> ds = _load_dataset("org/dataset", split="train")

        >>> # Load from local Delta Lake table
        >>> ds = _load_dataset("delta:///path/to/delta_table", streaming=True)

        >>> # Load from Databricks with authentication
        >>> ds = _load_dataset(
        ...     "delta://catalog.schema.table",
        ...     delta_storage_options={"DATABRICKS_TOKEN": "dapi..."},
        ...     streaming=True,
        ... )
    """
    # Check for Delta Lake sources first
    if isinstance(path_or_dataset_id, str):
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import (
            is_delta_lake_path,
            DeltaLakeDataset,
        )

        if is_delta_lake_path(path_or_dataset_id):
            if not streaming:
                raise ValueError(
                    "Delta Lake / Databricks sources are only supported in streaming mode. "
                    "Use ColumnMappedTextInstructionIterableDataset to avoid accidental dataset materialization."
                )
            return DeltaLakeDataset(
                table_path=path_or_dataset_id,
                storage_options=delta_storage_options,
                version=delta_version,
                sql_query=delta_sql_query,
            )
    return _load_dataset(
        path_or_dataset_id,
        split=split,
        streaming=streaming,
        name=name,
    )


class ColumnMappedTextInstructionIterableDataset(IterableDataset, ColumnMappedTextInstructionDataset):
    """Streaming iterable variant that reuses the column-mapping/tokenization logic.

    This wraps a Hugging Face streaming dataset (IterableDataset from `datasets`)
    or Delta Lake table and yields tokenized samples compatible with the non-streaming
    variant, while supporting sharding and epoch-setting for deterministic shuffles upstream.

    Supports the following data sources:
    - HuggingFace Hub datasets
    - Local JSON/JSONL files
    - Delta Lake tables (via delta://, dbfs:/, or local directories with _delta_log)
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
        delta_storage_options: Optional[Dict[str, str]] = None,
        delta_version: Optional[int] = None,
        delta_sql_query: Optional[str] = None,
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
        self.repeat_on_exhaustion = repeat_on_exhaustion
        if self.repeat_on_exhaustion is not True:
            raise ValueError("repeat_on_exhaustion must be True; False will be supported in the future.")

        # Always load in streaming mode
        ds = _load_streaming_dataset(
            path_or_dataset_id,
            split=split,
            streaming=True,
            name=name,
            delta_storage_options=delta_storage_options,
            delta_version=delta_version,
            delta_sql_query=delta_sql_query,
        )
        if limit_dataset_samples is not None and callable(getattr(ds, "take", None)):
            ds = ds.take(limit_dataset_samples)
        else:
            logger.warning("limit_dataset_samples ignored; 'take' not supported on this dataset")

        self.dataset = ds
        # Expose the underlying dataset's shard count (HF streaming uses `n_shards`).
        # This enables sharding strategies that depend on shard metadata.
        try:
            self.num_shards = int(getattr(ds, "num_shards", getattr(ds, "n_shards", self.num_shards)))
        except Exception:
            # Keep the default if the underlying dataset doesn't expose shard count.
            pass

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

    def __len__(self) -> int:
        raise RuntimeError("__len__ is not supported in streaming mode.")

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        raise RuntimeError("__getitem__ is not supported in streaming mode.")

    def set_epoch(self, epoch: int) -> None:
        if self.dataset is not None and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def shard(self, num_shards: int, index: int):
        if self.dataset is not None and callable(getattr(self.dataset, "shard", None)):
            self.dataset = self.dataset.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None):
        if self.dataset is not None and callable(getattr(self.dataset, "shuffle", None)):
            self.dataset = self.dataset.shuffle(buffer_size=buffer_size, seed=seed)
        return self
