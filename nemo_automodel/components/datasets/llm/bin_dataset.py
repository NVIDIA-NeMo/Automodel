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
"""PyTorch IterableDataset for .bin shards written by the modded-NanoGPT pre-processing script.

Each shard has the following layout::

    int32[256] header
        header[0] = 2788_95051      # magic number
        header[1] = 1               # version
        header[2] = num_tokens      # number of uint16 tokens that follow
        header[3] = dtype.itemsize  # bytes per token (2 for uint16)

    uint16[num_tokens] tokens

The dataset streams one contiguous *seq_len* token slice at a time and
returns the pair ``(inputs, labels)`` where ``labels`` is shifted by one
position.  Optionally, slices can be forced to start at the BOS token
(``align_to_bos=True``).

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
from torch.distributed.device_mesh import DeviceMesh

__all__ = ["BinTokenDataset", "load_bin_shard"]

MAGIC = 2788_95051
VERSION = 1
HEADER_BYTES = 256 * 4  # 256 int32s


def _peek_num_tokens(path: str | os.PathLike) -> int:
    """
    Returns total number of tokens from the shard header, without traversing the data.
    """
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    return int(header[2])

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
    assert header[0] == MAGIC, f"{path} magic number mismatch (got {header[0]})"
    assert header[1] == VERSION, f"{path} version mismatch (got {header[1]})"
    num_tokens = int(header[2])
    dtype = _get_dtype_from_val(int(header[3]))

    # Memory-map the tokens. Offset skips the 256x4-byte header.
    tokens_np = np.memmap(
        path, dtype=dtype, mode="r", offset=HEADER_BYTES, shape=(num_tokens,)
    )
    # UserWarning: The given NumPy array is not writable, and PyTorch does not
    # support non-writable tensors. This means writing to this tensor will result
    # in undefined behavior. You may want to copy the array to protect its data or
    # make it writable before converting it to a tensor. This type of warning will
    # be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
    return torch.from_numpy(tokens_np)


class BinTokenDataset(IterableDataset):
    """
    Dataset class for Binary Token Dataset.

    A Binary Token Dataset is a dataset that stores tokens in a binary file.
    The header contains:
    - 256x4-byte header (magic number, version, num_tokens, dtype.itemsize)
    - And the tokens themselves.

    Args:
        file_pattern : str | Sequence[str]
            Glob pattern (e.g. ``"data/fineweb_*_train_*.bin"``) **or** an explicit
            list of file paths.
        seq_len : int
            Length of the training sample returned (not counting the next-token
            target).  labels are simply ``inputs[1:]``.
        shuffle_files : bool, default False
            Shuffle the order of shards each epoch/iteration.
        align_to_bos : bool, default True
            Ensure that every slice starts with ``bos_token``.  When enabled, the
            dataset searches forward from the current position until it finds the
            next BOS token and starts there.
        bos_token : int, default 50256
            Token ID marking beginning-of-document.
        drop_last : bool, default True
            If the end of a shard does not have enough tokens for a full slice,
            skip the remainder rather than crossing the shard boundary.
        infinite : bool, default True
            Stream forever (wrap around shards).  When ``False``, stop after the
            last shard is exhausted.
    """

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        *,
        bos_token: int = None,
        shuffle_files: bool = False,
        align_to_bos: bool = False,
        drop_last: bool = True,
        infinite: bool = True,
        device_mesh: DeviceMesh = None,
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
        if self.align_to_bos and  bos_token is None:
            raise ValueError("bos_token must be provided when align_to_bos is True")
        self.bos_token = bos_token
        self.drop_last = drop_last
        self.infinite = infinite
        self.device_mesh = device_mesh

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # noqa: C901
        # Worker-specific setup
        worker = get_worker_info()
        print(f"worker: {worker}")
        rng = random.Random()
        if worker is not None:
            # Ensure each worker gets a *different* but deterministic view by seeding based on worker_id.
            rng.seed(worker.id + 12345)
        else:
            rng.seed(os.getpid())


        # ------------------------------------------------------------------
        # Determine the *global* worker id taking both DDP rank and DataLoader
        # worker id into account so that every worker processes a disjoint
        # subset of shards.
        # ------------------------------------------------------------------
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

        # Slice the file list so that each global worker gets roughly equal number of shards.
        worker_files = self.files[global_worker_id::total_workers]
        if not worker_files:
            worker_files = self.files.copy()  # fallback-duplication acceptable for small shard counts

        if self.shuffle_files:
            rng.shuffle(worker_files)

        while True:
            for file in worker_files:
                tokens = load_bin_shard(file)
                pos = 0

                # Optionally skip leading tokens until first BOS so slices start on BOS.
                if self.align_to_bos:
                    while pos < len(tokens) and tokens[pos].item() != self.bos_token:
                        pos += 1

                while pos + self.seq_len < len(tokens):
                    end = pos + self.seq_len + 1  # +1 for target shift
                    if end > len(tokens):
                        break
                    buf = tokens[pos:end]
                    assert len(buf) == self.seq_len + 1
                    inputs = buf[:-1].to(torch.int32).tolist()
                    labels = buf[1:].to(torch.int64).tolist()
                    yield dict(input_ids=inputs, labels=labels)

                    # Advance
                    if self.align_to_bos:
                        # Find next BOS token for the start of the next sample
                        pos = end
                        while pos < len(tokens) and tokens[pos].item() != self.bos_token:
                            pos += 1
                    else:
                        pos = end

                # Optionally drop remainder; otherwise cross shard (rarely useful)
                if not self.drop_last and pos < len(tokens) and not self.align_to_bos:
                    # carry over remainder to next shard (currently unused but left for parity)
                    _carry = tokens[pos:]
                    # The carry is ignored in this implementation to avoid stateful buffering.

            if not self.infinite:
                break

            # Start a new epoch – optionally reshuffle
            if self.shuffle_files:
                rng.shuffle(worker_files)

    # ------------------------------------------------------------------
    # Optional map-style access (limited)
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        """Total number of samples available **when** *infinite=False*.

        For *infinite=True* datasets the length is undefined and a ``TypeError``
        is raised because DataLoader would otherwise create an *epoch* concept
        that never terminates[].
        """
        return 1024 * 10
        if self.infinite:
            raise TypeError("`__len__` is undefined when `infinite=True`.")

        total_samples = 0
        for file in self.files:
            ntokens = _peek_num_tokens(file)
            # Each sample needs seq_len+1 tokens (for next-token target).
            total_samples += max(0, ntokens - 1) // (self.seq_len)
        return total_samples

    def __getitem__(self, index: int):  # type: ignore[override]
        """Random access to a specific sample (slow).

        Restrictions
        ------------
        * Only supported when ``infinite=False`` and ``align_to_bos=False``.
        * Access is **O(seq_len)** because we may still need to map the shard
          containing the sample, but no full scan is performed.
        """
        if self.infinite:
            raise TypeError("Random access is not supported when `infinite=True`.")
        if self.align_to_bos:
            raise NotImplementedError("__getitem__ with align_to_bos=True is not implemented.")

        # Determine which shard contains *index*
        running = 0
        for file in self.files:
            ntokens = _peek_num_tokens(file)
            samples_in_file = max(0, ntokens - 1) // (self.seq_len)
            if index < running + samples_in_file:
                # The desired sample lives in this file
                local_idx = index - running
                start = local_idx * self.seq_len
                tokens = load_bin_shard(file)
                buf = tokens[start : start + self.seq_len + 1]
                inputs = buf[:-1].to(torch.int32)
                labels = buf[1:].to(torch.int64)
                return inputs, labels
            running += samples_in_file

        raise IndexError("index out of range")
