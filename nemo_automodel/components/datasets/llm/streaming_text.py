# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Real-text pretraining chunks without padding or variable-length attention packing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from datasets import IterableDataset as HFIterableDataset
    from transformers import PreTrainedTokenizerBase

_, datasets = safe_import("datasets")
_, datasets_distributed = safe_import("datasets.distributed")


@dataclass
class StreamingTextConfig:
    """Build fixed-length next-token samples from a Hugging Face text stream.

    Documents are separated by EOS and concatenated into a continuous causal
    stream. Attention may cross document boundaries, as in ordinary unpacked
    pretraining. The stream repeats on exhaustion; ``max_steps`` bounds training.
    """

    accepts_tokenizer: ClassVar[bool] = True

    path: str = "HuggingFaceFW/fineweb-edu"
    name: str | None = "sample-10BT"
    split: str = "train"
    revision: str | None = None
    text_column: str = "text"
    seq_length: int = 4096

    def build(self, *, tokenizer: PreTrainedTokenizerBase | None) -> StreamingTextDataset:
        """Load the text stream and bind the runtime tokenizer."""
        if tokenizer is None or tokenizer.eos_token_id is None:
            raise ValueError("StreamingTextConfig requires a tokenizer with an EOS token")
        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        source = datasets.load_dataset(
            self.path, name=self.name, split=self.split, revision=self.revision, streaming=True
        )
        return StreamingTextDataset(source, tokenizer, self.seq_length, self.text_column)


class StreamingTextDataset(IterableDataset):
    """Yield int64 input/label vectors of exactly ``seq_length`` tokens.

    Labels are the next tokens, including at chunk boundaries. No token is
    padded or loss-masked. HF owns worker sharding; the training loader calls
    ``shard`` for data-parallel ranks. StatefulDataLoader can resume by replaying
    this deterministic iterator (including its partially consumed document).
    """

    def __init__(
        self,
        source: HFIterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        text_column: str = "text",
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if tokenizer.eos_token_id is None:
            raise ValueError("A document-separating EOS token is required")
        self.source = source
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_column = text_column

    def shard(self, num_shards: int, index: int) -> StreamingTextDataset:
        """Return a data-parallel shard, leaving the unsharded source unchanged."""
        source = datasets_distributed.split_dataset_by_node(self.source, rank=index, world_size=num_shards)
        return StreamingTextDataset(source, self.tokenizer, self.seq_length, self.text_column)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Tokenize documents lazily, retaining the remainder across documents."""
        worker = get_worker_info()
        if worker is not None and worker.id >= self.source.num_shards:
            return
        buffer: list[int] = []
        while True:
            has_text = False
            for row in self.source:
                text = row[self.text_column]
                if not text:
                    continue
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if not tokens:
                    continue
                has_text = True
                buffer.extend(tokens)
                buffer.append(self.tokenizer.eos_token_id)
                offset = 0
                while len(buffer) - offset > self.seq_length:
                    chunk = torch.tensor(buffer[offset : offset + self.seq_length + 1], dtype=torch.long)
                    offset += self.seq_length
                    yield {"input_ids": chunk[:-1], "labels": chunk[1:]}
                buffer = buffer[offset:]
            if not has_text:
                raise ValueError("The text stream shard contains no nonempty tokenized documents")
