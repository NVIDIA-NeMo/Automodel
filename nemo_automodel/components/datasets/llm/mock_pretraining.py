# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class _MockPretrainingDataset(Dataset):
    """Deterministic document-based samples for pretraining smoke tests."""

    _SPLIT_BOUNDS = {
        "train": (0.0, 1.0 / 3.0),
        "validation": (1.0 / 3.0, 2.0 / 3.0),
        "test": (2.0 / 3.0, 1.0),
    }
    _CORPUS_SEED = 0
    _CORPUS_SIZE = 100_000
    _MAX_DOCUMENT_LENGTH = 4096
    _SAMPLE_SEED = 1234

    def __init__(
        self,
        *,
        vocab_size: int,
        eod_token_id: int,
        seq_length: int,
        num_samples: int,
        split: Literal["train", "validation", "test"],
    ) -> None:
        self.vocab_size = vocab_size
        self.eod_token_id = eod_token_id
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.position_ids = torch.arange(seq_length, dtype=torch.long)

        corpus_rng = np.random.default_rng(seed=self._CORPUS_SEED)
        sequence_lengths = corpus_rng.integers(
            low=1,
            high=self._MAX_DOCUMENT_LENGTH,
            size=self._CORPUS_SIZE,
            dtype=np.int32,
        )
        split_start, split_end = self._SPLIT_BOUNDS[split]
        document_ids = np.arange(
            int(split_start * self._CORPUS_SIZE),
            int(split_end * self._CORPUS_SIZE),
            dtype=np.int32,
        )

        sample_rng = np.random.RandomState(self._SAMPLE_SEED)
        sample_rng.shuffle(document_ids)
        self._document_lengths = sequence_lengths[document_ids]
        self._document_ends = np.cumsum(self._document_lengths, dtype=np.int64)

        available_samples = int((self._document_ends[-1] - 1) // seq_length)
        if num_samples > available_samples:
            raise ValueError(
                f"num_samples={num_samples} exceeds the {available_samples} complete samples "
                f"available in the {split} split"
            )
        shuffle_dtype = np.uint32 if available_samples < np.iinfo(np.uint32).max - 1 else np.int64
        self._shuffle_index = np.arange(available_samples, dtype=shuffle_dtype)
        sample_rng.shuffle(self._shuffle_index)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one deterministic autoregressive sample."""
        sample_index = int(self._shuffle_index[index])
        stream_offset = sample_index * self.seq_length
        tokens = np.empty(self.seq_length + 1, dtype=np.int64)
        write_offset = 0

        while write_offset < tokens.size:
            document_index = int(np.searchsorted(self._document_ends, stream_offset, side="right"))
            document_start = 0 if document_index == 0 else int(self._document_ends[document_index - 1])
            offset_in_document = stream_offset - document_start
            document_length = int(self._document_lengths[document_index])
            take = min(document_length - offset_in_document, tokens.size - write_offset)

            positions = np.arange(offset_in_document, offset_in_document + take, dtype=np.int64)
            part = (positions + 1) % self.vocab_size
            if positions[-1] == document_length - 1:
                part[-1] = self.eod_token_id
            tokens[write_offset : write_offset + take] = part
            stream_offset += take
            write_offset += take

        token_tensor = torch.from_numpy(tokens)
        return {
            "input_ids": token_tensor[:-1],
            "labels": token_tensor[1:],
            "position_ids": self.position_ids,
        }


@dataclass(frozen=True)
class MockPretrainingDatasetConfig:
    """Configuration for a deterministic document-based pretraining corpus."""

    vocab_size: int = 32000
    """Tokenizer vocabulary size, including the end-of-document token."""
    eod_token_id: int | None = None
    """End-of-document token ID. Defaults to the final vocabulary entry."""
    seq_length: int = 512
    """Number of input and label tokens in each shifted sample."""
    num_samples: int = 100_000
    """Number of samples exposed by the dataset."""
    split: Literal["train", "validation", "test"] = "train"
    """Disjoint document split to expose."""

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2, got {self.vocab_size}")
        if self.eod_token_id is not None and not 0 <= self.eod_token_id < self.vocab_size:
            raise ValueError(f"eod_token_id must be in [0, {self.vocab_size}), got {self.eod_token_id}")
        if self.seq_length < 1:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.split not in _MockPretrainingDataset._SPLIT_BOUNDS:
            raise ValueError(f"split must be train, validation, or test, got {self.split!r}")

    def build(self) -> Dataset:
        """Build the configured deterministic dataset."""
        return _MockPretrainingDataset(
            vocab_size=self.vocab_size,
            eod_token_id=self.vocab_size - 1 if self.eod_token_id is None else self.eod_token_id,
            seq_length=self.seq_length,
            num_samples=self.num_samples,
            split=self.split,
        )
