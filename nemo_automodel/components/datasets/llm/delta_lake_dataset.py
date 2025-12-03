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

"""Delta Lake dataset support for streaming instruction-tuning datasets.

This module provides support for reading Delta Lake tables from Databricks or
local storage as streaming datasets. It integrates with the existing
ColumnMappedTextInstructionDataset infrastructure.

Usage:
    To use Delta Lake tables, prefix your path with "delta://" or point to
    a directory containing a "_delta_log" subdirectory:

    ```python
    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id="delta:///path/to/delta_table",
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
    )
    ```

    For Databricks tables accessed via Unity Catalog:

    ```python
    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id="delta://catalog.schema.table",
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
        delta_storage_options={"DATABRICKS_TOKEN": "dapi..."},
    )
    ```
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

logger = logging.getLogger(__name__)

# Lazy imports to avoid requiring deltalake when not used
_DELTALAKE_AVAILABLE: Optional[bool] = None


def _check_deltalake_available() -> bool:
    """Check if the deltalake package is available."""
    global _DELTALAKE_AVAILABLE
    if _DELTALAKE_AVAILABLE is None:
        try:
            import deltalake  # noqa: F401

            _DELTALAKE_AVAILABLE = True
        except ImportError:
            _DELTALAKE_AVAILABLE = False
    return _DELTALAKE_AVAILABLE


def is_delta_lake_path(path: str) -> bool:
    """Check if a path refers to a Delta Lake table.

    A path is considered a Delta Lake path if:
    1. It starts with "delta://" protocol prefix
    2. It's a local directory containing a "_delta_log" subdirectory
    3. It starts with "dbfs:/" (Databricks file system)

    Args:
        path: The path to check.

    Returns:
        True if the path is a Delta Lake table, False otherwise.
    """
    if not isinstance(path, str):
        return False

    # Check for explicit delta:// protocol
    if path.startswith("delta://"):
        return True

    # Check for Databricks file system paths
    if path.startswith("dbfs:/"):
        return True

    # Check for abfss:// (Azure Blob Storage with hierarchical namespace)
    if path.startswith("abfss://"):
        return True

    # Check for s3:// or s3a:// paths with _delta_log hint
    if path.startswith(("s3://", "s3a://", "gs://")) and "_delta" in path.lower():
        return True

    # Check for local directory with _delta_log
    local_path = Path(path)
    if local_path.exists() and local_path.is_dir():
        delta_log = local_path / "_delta_log"
        if delta_log.exists() and delta_log.is_dir():
            return True

    return False


def _normalize_delta_path(path: str) -> str:
    """Normalize a Delta Lake path by removing the delta:// prefix if present.

    Args:
        path: The Delta Lake path.

    Returns:
        The normalized path suitable for the deltalake library.
    """
    if path.startswith("delta://"):
        return path[8:]  # Remove "delta://" prefix
    return path


class DeltaLakeIterator:
    """Iterator that yields rows from a Delta Lake table.

    This class provides a streaming interface for Delta Lake tables,
    yielding rows as dictionaries one at a time to support memory-efficient
    iteration over large tables.

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read. If None, reads all columns.
        storage_options: Optional storage options for cloud storage access.
        batch_size: Number of rows to read at a time (default: 1024).
        version: Optional version of the table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        batch_size: int = 1024,
        version: Optional[int] = None,
    ):
        if not _check_deltalake_available():
            raise ImportError(
                "The 'deltalake' package is required for Delta Lake support. "
                "Install it with: pip install deltalake"
            )

        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.batch_size = batch_size
        self.version = version

        # Add environment-based storage options
        self._add_env_storage_options()

    def _add_env_storage_options(self):
        """Add storage options from environment variables if not already set."""
        env_mappings = {
            "DATABRICKS_TOKEN": ["DATABRICKS_TOKEN", "DATABRICKS_ACCESS_TOKEN"],
            "DATABRICKS_HOST": ["DATABRICKS_HOST", "DATABRICKS_WORKSPACE_URL"],
            "AWS_ACCESS_KEY_ID": ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": ["AWS_SECRET_ACCESS_KEY"],
            "AWS_SESSION_TOKEN": ["AWS_SESSION_TOKEN"],
            "AWS_REGION": ["AWS_REGION", "AWS_DEFAULT_REGION"],
            "AZURE_STORAGE_ACCOUNT_NAME": ["AZURE_STORAGE_ACCOUNT_NAME"],
            "AZURE_STORAGE_ACCOUNT_KEY": ["AZURE_STORAGE_ACCOUNT_KEY"],
            "AZURE_STORAGE_SAS_TOKEN": ["AZURE_STORAGE_SAS_TOKEN"],
            "GOOGLE_APPLICATION_CREDENTIALS": ["GOOGLE_APPLICATION_CREDENTIALS"],
        }

        for key, env_vars in env_mappings.items():
            if key not in self.storage_options:
                for env_var in env_vars:
                    value = os.environ.get(env_var)
                    if value:
                        self.storage_options[key] = value
                        break

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the Delta Lake table.

        Yields:
            Dict containing column name to value mappings for each row.
        """
        from deltalake import DeltaTable

        # Open the Delta table
        dt = DeltaTable(self.table_path, storage_options=self.storage_options, version=self.version)

        # Get PyArrow dataset for efficient streaming
        pa_dataset = dt.to_pyarrow_dataset()

        # Iterate over batches
        for batch in pa_dataset.to_batches(columns=self.columns, batch_size=self.batch_size):
            # Convert batch to Python dicts
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[next(iter(batch_dict))])

            for i in range(num_rows):
                yield {col: batch_dict[col][i] for col in batch_dict}


class DeltaLakeDataset:
    """A dataset wrapper for Delta Lake tables that integrates with HuggingFace datasets.

    This class provides a HuggingFace-compatible interface for Delta Lake tables,
    supporting both iteration and indexing (when the table is small enough).

    Args:
        table_path: Path to the Delta Lake table. Can be:
            - Local path: "/path/to/delta_table"
            - Delta protocol: "delta:///path/to/delta_table"
            - DBFS: "dbfs:/path/to/delta_table"
            - S3: "s3://bucket/path/to/delta_table"
            - Azure: "abfss://container@account.dfs.core.windows.net/path"
            - GCS: "gs://bucket/path/to/delta_table"
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns an iterable dataset. If False, loads into memory.
        version: Optional specific version of the Delta table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        streaming: bool = False,
        version: Optional[int] = None,
    ):
        if not _check_deltalake_available():
            raise ImportError(
                "The 'deltalake' package is required for Delta Lake support. "
                "Install it with: pip install deltalake"
            )

        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.streaming = streaming
        self.version = version
        self._data: Optional[list] = None
        self._length: Optional[int] = None

        # Add environment-based storage options
        self._iterator = DeltaLakeIterator(
            table_path=table_path,
            columns=columns,
            storage_options=storage_options,
            version=version,
        )

        if not streaming:
            self._load_data()

    def _load_data(self):
        """Load the entire Delta table into memory."""
        from deltalake import DeltaTable

        dt = DeltaTable(
            self.table_path,
            storage_options=self._iterator.storage_options,
            version=self.version,
        )
        pa_table = dt.to_pyarrow_table(columns=self.columns)
        self._data = pa_table.to_pylist()
        self._length = len(self._data)

    def __len__(self) -> int:
        """Return the number of rows in the table.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("__len__ is not supported in streaming mode. Use iteration instead.")
        if self._length is None:
            self._load_data()
        return self._length  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific row by index.

        Args:
            idx: The row index.

        Returns:
            Dict containing column name to value mappings.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("__getitem__ is not supported in streaming mode. Use iteration instead.")
        if self._data is None:
            self._load_data()
        return self._data[idx]  # type: ignore[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the dataset.

        Yields:
            Dict containing column name to value mappings for each row.
        """
        if self.streaming:
            yield from self._iterator
        else:
            if self._data is None:
                self._load_data()
            yield from self._data  # type: ignore[misc]


def load_delta_lake_dataset(
    path: str,
    columns: Optional[list] = None,
    storage_options: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    version: Optional[int] = None,
) -> Union[DeltaLakeDataset, "HFDeltaLakeDataset"]:
    """Load a Delta Lake table as a HuggingFace-compatible dataset.

    This function creates a dataset that can be used with the ColumnMappedTextInstructionDataset
    and other HuggingFace-compatible dataset consumers.

    Args:
        path: Path to the Delta Lake table.
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns a streaming iterable dataset.
        version: Optional specific version of the Delta table to read.

    Returns:
        A HuggingFace-compatible dataset object.

    Example:
        ```python
        # Local Delta table
        ds = load_delta_lake_dataset("/path/to/delta_table")

        # Databricks table with authentication
        ds = load_delta_lake_dataset(
            "delta://catalog.schema.table",
            storage_options={"DATABRICKS_TOKEN": "dapi..."},
            streaming=True,
        )
        ```
    """
    if not _check_deltalake_available():
        raise ImportError(
            "The 'deltalake' package is required for Delta Lake support. "
            "Install it with: pip install deltalake"
        )

    # Return HF-compatible wrapper if datasets library is available
    try:
        from datasets import Dataset, IterableDataset

        return HFDeltaLakeDataset(
            table_path=path,
            columns=columns,
            storage_options=storage_options,
            streaming=streaming,
            version=version,
        )
    except ImportError:
        return DeltaLakeDataset(
            table_path=path,
            columns=columns,
            storage_options=storage_options,
            streaming=streaming,
            version=version,
        )


class HFDeltaLakeDataset:
    """HuggingFace datasets-compatible wrapper for Delta Lake tables.

    This class provides better integration with the HuggingFace datasets library,
    supporting features like sharding, shuffling, and epoch setting for distributed
    training scenarios.

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns a streaming iterable dataset.
        version: Optional specific version of the Delta table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        streaming: bool = False,
        version: Optional[int] = None,
    ):
        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.streaming = streaming
        self.version = version
        self._dataset: Optional[Any] = None
        self._epoch: int = 0
        self._shard_info: Optional[tuple] = None  # (num_shards, shard_index)
        self._shuffle_info: Optional[tuple] = None  # (buffer_size, seed)

        # Eagerly create the internal iterator to validate the table
        self._base_iterator = DeltaLakeIterator(
            table_path=table_path,
            columns=columns,
            storage_options=storage_options,
            version=version,
        )

        if not streaming:
            self._load_as_hf_dataset()

    def _load_as_hf_dataset(self):
        """Load the Delta table as a HuggingFace Dataset."""
        from datasets import Dataset

        from deltalake import DeltaTable

        dt = DeltaTable(
            self.table_path,
            storage_options=self._base_iterator.storage_options,
            version=self.version,
        )
        pa_table = dt.to_pyarrow_table(columns=self.columns)
        self._dataset = Dataset.from_pandas(pa_table.to_pandas())

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        if self.streaming:
            raise RuntimeError("__len__ is not supported in streaming mode.")
        if self._dataset is None:
            self._load_as_hf_dataset()
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific row by index."""
        if self.streaming:
            raise RuntimeError("__getitem__ is not supported in streaming mode.")
        if self._dataset is None:
            self._load_as_hf_dataset()
        return self._dataset[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the dataset."""
        if self.streaming:
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._base_iterator.storage_options,
                version=self.version,
            )

            row_idx = 0
            for row in iterator:
                # Apply sharding if configured
                if self._shard_info is not None:
                    num_shards, shard_index = self._shard_info
                    if row_idx % num_shards != shard_index:
                        row_idx += 1
                        continue

                yield row
                row_idx += 1
        else:
            if self._dataset is None:
                self._load_as_hf_dataset()
            yield from self._dataset

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling.

        Args:
            epoch: The epoch number.
        """
        self._epoch = epoch

    def shard(self, num_shards: int, index: int) -> "HFDeltaLakeDataset":
        """Shard the dataset for distributed processing.

        Args:
            num_shards: Total number of shards.
            index: Index of this shard (0-based).

        Returns:
            Self for method chaining.
        """
        self._shard_info = (num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None) -> "HFDeltaLakeDataset":
        """Configure shuffling for the dataset.

        Note: For streaming Delta Lake datasets, shuffling is performed on-the-fly
        using a shuffle buffer. The actual shuffling happens during iteration.

        Args:
            buffer_size: Size of the shuffle buffer.
            seed: Random seed for reproducibility.

        Returns:
            Self for method chaining.
        """
        self._shuffle_info = (buffer_size, seed if seed is not None else self._epoch)
        return self

    def take(self, n: int) -> "HFDeltaLakeDataset":
        """Limit the dataset to the first n samples.

        Args:
            n: Number of samples to take.

        Returns:
            A new HFDeltaLakeDataset limited to n samples.
        """
        # Create a wrapper that limits iteration
        limited = _LimitedDeltaLakeDataset(self, n)
        return limited  # type: ignore[return-value]


class _LimitedDeltaLakeDataset:
    """Internal wrapper to limit a Delta Lake dataset to n samples."""

    def __init__(self, base: HFDeltaLakeDataset, limit: int):
        self._base = base
        self._limit = limit

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        count = 0
        for row in self._base:
            if count >= self._limit:
                break
            yield row
            count += 1

    def set_epoch(self, epoch: int) -> None:
        self._base.set_epoch(epoch)

    def shard(self, num_shards: int, index: int):
        self._base.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None):
        self._base.shuffle(buffer_size, seed)
        return self

    def take(self, n: int):
        return _LimitedDeltaLakeDataset(self._base, min(n, self._limit))

