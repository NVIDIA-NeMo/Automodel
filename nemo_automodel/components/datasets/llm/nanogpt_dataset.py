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
"""PyTorch IterableDataset for .bin shards written by NanoGPT preprocessing scripts.

Supports both legacy fineweb.py format and the newer nanogpt_data_processor.py format.

Legacy format (fineweb.py)::

    int32[256] header
        header[0] = 20240520        # magic number
        header[1] = 1               # version
        header[2] = num_tokens      # number of uint16 tokens that follow
        header[3] = (unused)        # defaults to 0

    uint16[num_tokens] tokens

New format (nanogpt_data_processor.py)::

    int32[256] header
        header[0] = 2788_95051      # magic number
        header[1] = 1               # version
        header[2] = num_tokens      # number of tokens that follow
        header[3] = dtype.itemsize  # bytes per token (2 for uint16, 4 for uint32)

    uint16/uint32[num_tokens] tokens

Optionally, a corresponding .bos.idx file can exist alongside each .bin file::

    int32[n_bos_tokens] bos_positions
        # Array of absolute byte positions where BOS tokens occur in the .bin file

The dataset streams one contiguous *seq_len* token slice at a time and
returns the pair ``(inputs, labels)`` where ``labels`` is shifted by one
position.  Optionally, slices can be forced to start at the BOS token
(``align_to_bos=True``). When BOS alignment is enabled, the dataset will use
.bos.idx files for efficient BOS token lookup when available, falling back
to linear search otherwise.

This file is copied (with minimal adjustments) from
``modded-nanogpt/data/bin_dataset.py`` so that projects depending on
``nemo_automodel`` can directly import ``BinTokenDataset`` without taking a
runtime dependency on the NanoGPT codebase.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

__all__ = ["NanogptDataset", "load_bin_shard"]

# Support both legacy fineweb.py format and new nanogpt_data_processor.py format
MAGIC = 2788_95051  # New format magic number
LEGACY_MAGIC = 20240520  # Legacy fineweb.py magic number
VERSION = 1
HEADER_BYTES = 256 * 4  # 256 int32s

# Export both magic numbers for compatibility
HEADER_SIZE = 256


def _peek_num_tokens(path: str | os.PathLike) -> int:
    """
    Returns total number of tokens from the shard header, without traversing the data.
    Supports both legacy fineweb.py and new nanogpt_data_processor.py formats.
    """
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    # Validate magic number for both supported formats
    assert header[0] == MAGIC or header[0] == LEGACY_MAGIC, f"{path} magic number mismatch (got {header[0]})"
    return int(header[2])


def _load_bos_index(path: str | os.PathLike) -> np.ndarray | None:
    """
    Load BOS token positions from a .bos.idx file if it exists.

    Args:
        path: Path to the .bin file (will look for corresponding .bos.idx file)

    Returns:
        Array of BOS token positions if index file exists, None otherwise.
    """
    if isinstance(path, str):
        path = Path(path)

    # Look for .bos.idx file corresponding to the .bin file
    idx_path = path.with_suffix(".bos.idx")

    if not idx_path.exists():
        return None

    try:
        # Load BOS positions as int32 array
        bos_positions = np.fromfile(idx_path, dtype=np.int32)
        return bos_positions
    except Exception:
        # If there's any error loading the index file, return None to fall back to linear search
        return None


def _find_next_bos_with_index(bos_positions: np.ndarray, start_pos: int, max_pos: int) -> int:
    """
    Find the next BOS token position using the index.

    Args:
        bos_positions: Array of BOS token positions
        start_pos: Current position to search from
        max_pos: Maximum position to search up to

    Returns:
        Position of next BOS token, or max_pos if none found.
    """
    # Find BOS positions that are >= start_pos and < max_pos
    valid_positions = bos_positions[(bos_positions >= start_pos) & (bos_positions < max_pos)]

    if len(valid_positions) > 0:
        return int(valid_positions[0])

    return max_pos


def _get_dtype_from_val(n_bytes: int) -> torch.dtype:
    """
    Returns the torch.dtype for the given value.
    """
    if n_bytes == 2:
        return np.uint16
    elif n_bytes == 4:
        return np.uint32
    else:
        raise ValueError(f"Expected {n_bytes} to be equal to 2 (uint16) or 4 (uint32).")


def load_bin_shard(path: str | os.PathLike) -> torch.Tensor:
    """
    Memory-map a *.bin* shard and return it as a 1-D ``torch.uint16/uint32`` tensor.

    The returned tensor **shares** memory with the underlying file and is
    therefore extremely cheap.  Do *not* modify it in-place.
    """
    if isinstance(path, str):
        path = Path(path)

    # Read header to sanity-check
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    assert header[0] == MAGIC or header[0] == LEGACY_MAGIC, f"{path} magic number mismatch (got {header[0]})"
    assert header[1] == VERSION, f"{path} version mismatch (got {header[1]})"
    num_tokens = int(header[2])

    # Handle dtype detection for both legacy and new formats
    if header[0] == LEGACY_MAGIC:
        # Legacy fineweb.py format: always uint16, header[3] not used
        dtype = np.uint16
    else:
        # New nanogpt_data_processor.py format: header[3] contains bytes per token
        dtype = _get_dtype_from_val(int(header[3]))

    # Memory-map the tokens. Offset skips the 256x4-byte header.
    tokens_np = np.memmap(path, dtype=dtype, mode="r", offset=HEADER_BYTES, shape=(num_tokens,))
    # UserWarning: The given NumPy array is not writable, and PyTorch does not
    # support non-writable tensors. This means writing to this tensor will result
    # in undefined behavior. You may want to copy the array to protect its data or
    # make it writable before converting it to a tensor. This type of warning will
    # be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
    return torch.from_numpy(tokens_np)


def _get_next_bos_position(
    tokens: torch.Tensor, bos_token: int, bos_positions: np.ndarray, pos: int, max_pos: int
) -> int:
    """
    Get the next BOS token position.

    Args:
        tokens: Tensor of tokens
        bos_token: BOS token ID
        bos_positions: Array of BOS token positions
        pos: Current position
        max_pos: Maximum position

    Returns:
        Next BOS token position
    """
    if bos_positions is not None:
        # Use index file for efficient BOS search
        pos = _find_next_bos_with_index(bos_positions, pos, max_pos)
    else:
        # Fall back to linear search
        while pos < max_pos and tokens[pos].item() != bos_token:
            pos += 1
    return pos


def _get_start_end_pos_single_file(total_tokens: int, total_workers: int, global_worker_id: int) -> tuple[int, int]:
    """
    Get the start and end positions for a single file, accounting for the number of workers.

    Args:
        total_tokens: Total number of tokens in the file
        total_workers: Total number of workers
        global_worker_id: Global worker ID

    Returns:
        Tuple of (start position, end position)
    """
    # Calculate the portion for this worker
    tokens_per_worker = total_tokens // total_workers
    file_start_pos = global_worker_id * tokens_per_worker

    # Last worker gets any remaining tokens
    if global_worker_id == total_workers - 1:
        file_end_pos = total_tokens
    else:
        file_end_pos = file_start_pos + tokens_per_worker
    return file_start_pos, file_end_pos


def _get_worker_id_and_total_workers(worker: get_worker_info) -> tuple[int, int]:
    """
    Get the total number of workers.
    """
    # Determine the *global* worker id taking both DDP rank and DataLoader
    # worker id into account so that every worker processes a disjoint
    # subset of shards.
    try:
        import torch.distributed as dist

        dist_world_size = dist.get_world_size() if dist.is_initialized() else 1
        dist_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        dist_world_size = 1
        dist_rank = 0

    dl_num_workers = worker.num_workers if worker is not None else 1
    dl_worker_id = worker.id if worker is not None else 0

    total_workers = dist_world_size * dl_num_workers
    global_worker_id = dist_rank * dl_num_workers + dl_worker_id

    return global_worker_id, total_workers


class NanogptDataset(IterableDataset):
    """
    Dataset class for NanoGPT Dataset.

    A NanoGPT Dataset is a dataset that stores tokens in a binary file.
    The header contains:
    - 256x4-byte header (magic number, version, num_tokens, dtype.itemsize)
    - And the tokens themselves.

    Optionally, a corresponding .bos.idx file can be present alongside each .bin file
    containing precomputed BOS token positions for efficient alignment when
    ``align_to_bos=True``. If the index file is not present, the dataset falls back
    to linear search for BOS tokens.

    Args:
        file_pattern : str | Sequence[str]
            Glob pattern (e.g. ``"data/fineweb_*_train_*.bin"``) **or** an explicit
            list of file paths.
        seq_len : int
            Length of the training sample returned (not counting the next-token
            target).  labels are simply ``inputs[1:]``.
        shuffle_files : bool, default False
            Shuffle the order of shards each epoch/iteration.
        align_to_bos : bool, default False
            Ensure that every slice starts with ``bos_token``.  When enabled, the
            dataset searches forward from the current position until it finds the
            next BOS token and starts there. Uses .bos.idx files when available
            for efficient search, falls back to linear search otherwise.
            Requires ``bos_token`` to be provided.
        bos_token : int, optional, default None.
            Token ID marking beginning-of-document.
    """

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        *,
        bos_token: int | None = None,
        shuffle_files: bool = False,
        align_to_bos: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(file_pattern, (str, Path)):
            self.files: List[str] = sorted(glob.glob(str(file_pattern)))
        else:
            self.files = list(map(str, file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched pattern {file_pattern}")
        self.seq_len = int(seq_len)
        self.shuffle_files = shuffle_files
        self.align_to_bos = align_to_bos
        if self.align_to_bos and bos_token is None:
            raise ValueError("bos_token must be provided when align_to_bos is True")
        self.bos_token = bos_token

    def _setup_worker_context(self, files, shuffle) -> tuple[List[str], random.Random, bool, int, int]:
        """
        Set up worker-specific context including file assignment and splitting parameters.

        Returns:
            Tuple of (worker_files, rng, split_single_file, file_start_pos, file_end_pos)
        """
        # Worker-specific setup
        worker = get_worker_info()
        rng = random.Random()
        if worker is not None:
            # Ensure each worker gets a *different* but deterministic view by seeding based on worker_id.
            rng.seed(worker.id + 12345)
        else:
            rng.seed(os.getpid())

        global_worker_id, total_workers = _get_worker_id_and_total_workers(worker)
        # Slice the file list so that each global worker gets roughly equal number of shards.
        worker_files = files[global_worker_id::total_workers].copy()
        if not worker_files:
            worker_files = files.copy()  # fallback-duplication acceptable for small shard counts

        # Handle single file case: split the file among workers
        split_single_file = len(worker_files) == 1 and total_workers > 1
        file_start_pos = 0
        file_end_pos = None

        if split_single_file:
            # Get the total number of tokens in the single file
            total_tokens = _peek_num_tokens(worker_files[0])
            file_start_pos, file_end_pos = _get_start_end_pos_single_file(total_tokens, total_workers, global_worker_id)

        if shuffle:
            rng.shuffle(worker_files)

        return worker_files, rng, split_single_file, file_start_pos, file_end_pos

    def _process_file_tokens(
        self, file: str, split_single_file: bool, file_start_pos: int, file_end_pos: int
    ) -> Iterator[dict]:
        """
        Process tokens from a single file and yield training samples.

        Args:
            file: Path to the .bin file to process
            split_single_file: Whether we're splitting a single file among workers
            file_start_pos: Starting position in the file (for single file splitting)
            file_end_pos: Ending position in the file (for single file splitting)

        Yields:
            Dictionary containing 'input_ids' and 'labels' for training
        """
        tokens = load_bin_shard(file)

        # Load BOS index if available for efficient BOS alignment
        bos_positions = None
        if self.align_to_bos:
            bos_positions = _load_bos_index(file)

        # Set start and end positions based on whether we're splitting a single file
        if split_single_file:
            pos = file_start_pos
            if file_end_pos is not None:
                max_pos = min(file_end_pos, len(tokens))
            else:
                max_pos = len(tokens)
        else:
            pos = 0
            max_pos = len(tokens)

        # Optionally skip leading tokens until first BOS so slices start on BOS.
        if self.align_to_bos:
            pos = _get_next_bos_position(tokens, self.bos_token, bos_positions, pos, max_pos)

        while pos + self.seq_len < max_pos:
            end = pos + self.seq_len + 1  # +1 for target shift
            if end > max_pos:
                break
            buf = tokens[pos:end]
            assert len(buf) == self.seq_len + 1
            inputs = buf[:-1].to(torch.int32).tolist()
            labels = buf[1:].to(torch.int64).tolist()
            yield dict(input_ids=inputs, labels=labels)

            # Advance
            if self.align_to_bos:
                # Find next BOS token for the start of the next sample
                pos = _get_next_bos_position(tokens, self.bos_token, bos_positions, end, max_pos)
            else:
                pos = end

    def _get_file_iterator(
        self,
        worker_files: List[str],
        rng: random.Random,
        split_single_file: bool,
        file_start_pos: int,
        file_end_pos: int,
    ) -> Iterator[dict]:
        """
        Generate training samples from all assigned files, handling infinite iteration.

        Args:
            worker_files: List of files assigned to this worker
            rng: Random number generator for shuffling
            split_single_file: Whether we're splitting a single file among workers
            file_start_pos: Starting position in file (for single file splitting)
            file_end_pos: Ending position in file (for single file splitting)

        Yields:
            Training sample dictionaries from all files
        """
        while True:
            for file in worker_files:
                yield from self._process_file_tokens(file, split_single_file, file_start_pos, file_end_pos)

            # Start a new epoch, optionally reshuffle
            if self.shuffle_files:
                rng.shuffle(worker_files)

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over training samples from the dataset.

        Yields:
            Dictionary containing 'input_ids' and 'labels' for training
        """
        worker_files, rng, split_single_file, file_start_pos, file_end_pos = self._setup_worker_context(
            self.files, self.shuffle_files
        )

        yield from self._get_file_iterator(worker_files, rng, split_single_file, file_start_pos, file_end_pos)

    def __len__(self) -> int:  # type: ignore[override]
        raise NotImplementedError("__len__ is not implemented for NanogptDataset.")

    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented for NanogptDataset.")
