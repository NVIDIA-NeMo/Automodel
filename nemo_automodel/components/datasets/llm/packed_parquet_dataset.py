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

"""Packed Parquet SFT dataset for reading pre-packed sequences from Parquet files.

Reads Parquet files with the RFC packed SFT format:
    - input_ids: list<int32>    variable-length token IDs (concatenated sequences)
    - loss_mask: list<uint8>    1 = compute loss, 0 = ignore
    - seq_start_id: list<int32> sequence boundary positions within each pack

Outputs the AutoModel packed format expected by ``packed_sequence_thd_collater``:
    - input_ids, labels, position_ids, seq_lens, seq_lens_padded (all tensors)

Usage::

    # YAML config
    dataset:
      _target_: nemo_automodel.components.datasets.llm.packed_parquet_dataset.PackedParquetDataset
      data_path: /data/packed_sft/shard_*.idx.parquet
      packed_sequence_size: 4096

    packed_sequence:
      packed_sequence_size: 4096

    dataloader:
      _target_: torchdata.stateful_dataloader.StatefulDataLoader
      collate_fn: nemo_automodel.components.datasets.utils.packed_sequence_thd_collater
      shuffle: true
"""

import bisect
import glob as glob_module
import logging
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.packed_sequence import (
    CROSS_ENTROPY_IGNORE_IDX,
    _tensorize_and_pad_pack,
)

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"input_ids", "loss_mask", "seq_start_id"}


class _ReaderState:
    """Per-worker reader state with row-group caching.

    Created lazily on first ``__getitem__`` so that the parent Dataset is
    pickle-safe for DataLoader worker spawning.  Each worker creates its
    own instance with independent file handles.

    Caches the last-read row group to avoid repeated I/O for consecutive
    samples that fall in the same row group.
    """

    def __init__(self, files: list[str]):
        self._files = files
        self._pf_cache: dict[int, object] = {}  # file_idx -> ParquetFile
        self._cached_rg_flat_idx: int = -1
        self._cached_table = None

    def _get_parquet_file(self, file_idx: int):
        import pyarrow.parquet as pq

        if file_idx not in self._pf_cache:
            self._pf_cache[file_idx] = pq.ParquetFile(self._files[file_idx], memory_map=True)
        return self._pf_cache[file_idx]

    def read_row(
        self,
        rg_flat_idx: int,
        row_within_rg: int,
        rg_index: list[tuple[int, int, int]],
    ) -> dict:
        """Read a single row, using the row-group cache."""
        if self._cached_rg_flat_idx != rg_flat_idx:
            file_idx, rg_idx, _num_rows = rg_index[rg_flat_idx]
            pf = self._get_parquet_file(file_idx)
            self._cached_table = pf.read_row_group(rg_idx, columns=list(_REQUIRED_COLUMNS))
            self._cached_rg_flat_idx = rg_flat_idx

        return {
            col: self._cached_table.column(col)[row_within_rg].as_py()
            for col in _REQUIRED_COLUMNS
        }


class PackedParquetDataset(Dataset):
    """Map-style dataset that reads pre-packed Parquet files in RFC format.

    Args:
        data_path: Path to a Parquet file, glob pattern, directory, or list of
            any of the above.  Directories are scanned for ``*.idx.parquet`` /
            ``*.idx.pq`` first, then ``*.parquet`` / ``*.pq``.
        packed_sequence_size: Target pack length.  Rows shorter than this are
            padded; rows longer raise ``ValueError``.
        padding_idx: Token ID used for padding ``input_ids``.
        cp_size: Context-parallel size for CP-aware padding.
        split: Accepted for config compatibility but unused (data is pre-split
            in files).
        tokenizer: Accepted for config compatibility but unused.
    """

    is_pre_packed: bool = True

    def __init__(
        self,
        data_path: Union[str, list[str]],
        packed_sequence_size: int,
        padding_idx: int = 0,
        cp_size: int = 1,
        split: str = "train",
        tokenizer=None,
    ):
        self._data_path = data_path
        self._packed_sequence_size = packed_sequence_size
        self._padding_idx = padding_idx
        self._cp_size = cp_size

        # Resolve file list eagerly (path resolution only, no file handles)
        self._files = self._resolve_files(data_path)
        if not self._files:
            raise FileNotFoundError(f"No Parquet files found at: {data_path}")

        # Build row-group index from metadata (no row data read)
        self._rg_index: list[tuple[int, int, int]] = []  # (file_idx, rg_idx, num_rows)
        self._rg_cumulative: list[int] = []  # cumulative row count at start of each rg
        self._total_rows = 0
        self._build_index()

        # Lazy reader state (created per-worker on first __getitem__)
        self._reader_state: Optional[_ReaderState] = None

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_files(data_path: Union[str, list[str]]) -> list[str]:
        """Resolve *data_path* to a sorted, deduplicated list of Parquet paths."""
        if isinstance(data_path, (list, tuple)):
            files: list[str] = []
            for p in data_path:
                files.extend(PackedParquetDataset._resolve_files(p))
            return sorted(set(files))

        path_str = str(data_path)

        # Glob pattern
        if "*" in path_str or "?" in path_str:
            return sorted(glob_module.glob(path_str))

        p = Path(path_str)
        if p.is_file():
            return [str(p)]

        if p.is_dir():
            # Prefer *.idx.parquet / *.idx.pq  (RFC naming convention)
            files = sorted(glob_module.glob(str(p / "*.idx.parquet")))
            files += sorted(glob_module.glob(str(p / "*.idx.pq")))
            if not files:
                files = sorted(glob_module.glob(str(p / "*.parquet")))
                files += sorted(glob_module.glob(str(p / "*.pq")))
            return sorted(set(files))

        return sorted(glob_module.glob(path_str))

    # ------------------------------------------------------------------
    # Index building (metadata only)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Read Parquet metadata to build a cumulative row-group index."""
        import pyarrow.parquet as pq

        cumulative = 0
        schema_validated = False

        for file_idx, filepath in enumerate(self._files):
            pf = pq.ParquetFile(filepath)

            # Validate schema on first file
            if not schema_validated:
                col_names = set(pf.schema_arrow.names)
                missing = _REQUIRED_COLUMNS - col_names
                if missing:
                    raise ValueError(
                        f"Parquet file {filepath} is missing required columns: {missing}. "
                        f"Expected columns: {_REQUIRED_COLUMNS}"
                    )
                schema_validated = True

            metadata = pf.metadata
            for rg_idx in range(metadata.num_row_groups):
                num_rows = metadata.row_group(rg_idx).num_rows
                self._rg_index.append((file_idx, rg_idx, num_rows))
                self._rg_cumulative.append(cumulative)
                cumulative += num_rows

        self._total_rows = cumulative
        # Sentinel for bisect
        self._rg_cumulative.append(cumulative)

        logger.info(
            "PackedParquetDataset: %d file(s), %d row group(s), %d total rows",
            len(self._files),
            len(self._rg_index),
            self._total_rows,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, idx: int) -> dict[str, list]:
        if idx < 0:
            idx += self._total_rows
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        # Lazy init (pickle-safe for DataLoader workers)
        if self._reader_state is None:
            self._reader_state = _ReaderState(self._files)

        rg_flat_idx, row_within_rg = self._locate_row(idx)
        raw = self._reader_state.read_row(rg_flat_idx, row_within_rg, self._rg_index)
        return self._convert_row(raw)

    # ------------------------------------------------------------------
    # Row location
    # ------------------------------------------------------------------

    def _locate_row(self, global_idx: int) -> tuple[int, int]:
        """Map *global_idx* → (flat_rg_index, row_offset_within_rg)."""
        rg_flat_idx = bisect.bisect_right(self._rg_cumulative, global_idx) - 1
        row_within_rg = global_idx - self._rg_cumulative[rg_flat_idx]
        return rg_flat_idx, row_within_rg

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    def _convert_row(self, raw: dict) -> dict[str, list]:
        """Convert an RFC Parquet row to AutoModel packed format.

        Returns lists (not tensors) to match the contract of HuggingFace
        ``Dataset.__getitem__``, which is what the ``packed_sequence_thd_collater``
        expects.

        Steps:
            1. labels  = input_ids where loss_mask == 1, else -100
            2. seq_lens = diffs between consecutive seq_start_id values
            3. position_ids = range resetting at each boundary
            4. Pad via ``_tensorize_and_pad_pack`` then convert to lists
        """
        input_ids: list[int] = raw["input_ids"]
        loss_mask: list[int] = raw["loss_mask"]
        seq_start_id: list[int] = raw["seq_start_id"]

        n = len(input_ids)

        if n > self._packed_sequence_size:
            raise ValueError(
                f"Parquet row has {n} tokens but packed_sequence_size is "
                f"{self._packed_sequence_size}. Increase packed_sequence_size or "
                f"ensure data is pre-packed to fit."
            )

        # 1. labels
        labels = [
            tok if mask == 1 else CROSS_ENTROPY_IGNORE_IDX
            for tok, mask in zip(input_ids, loss_mask)
        ]

        # 2. seq_lens
        seq_lens: list[int] = []
        for i in range(len(seq_start_id)):
            end = seq_start_id[i + 1] if i + 1 < len(seq_start_id) else n
            seq_lens.append(end - seq_start_id[i])

        # 3. position_ids (reset at each boundary)
        position_ids = [0] * n
        for i, start in enumerate(seq_start_id):
            end = seq_start_id[i + 1] if i + 1 < len(seq_start_id) else n
            for j in range(start, end):
                position_ids[j] = j - start

        # 4. Pad and tensorize via existing helper
        pack = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "seq_lens": seq_lens,
        }

        result = _tensorize_and_pad_pack(
            pack,
            padding_idx=self._padding_idx,
            packed_sequence_size=self._packed_sequence_size,
            cp_size=self._cp_size,
        )

        # Convert tensors to lists to match HF Dataset.__getitem__ contract.
        # The packed_sequence_thd_collater expects lists, not tensors.
        return {k: v.tolist() for k, v in result.items()}
