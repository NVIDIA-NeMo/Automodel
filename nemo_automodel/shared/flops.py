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

"""Typed results for model-owned CPU batch FLOPs accounting."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class BatchFlops:
    """Effective model work, excluding padding, recomputation and kernel overhead.

    Attributes:
        flops: Training FLOPs with the LM head evaluated on supervised tokens.
        all_token_head_flops: Comparison FLOPs with the LM head on every valid token.
        num_tokens: Valid input tokens before context-parallel sharding.
        num_supervised_tokens: Non-ignored labels on valid input tokens.
    """

    flops: float
    all_token_head_flops: float
    num_tokens: int
    num_supervised_tokens: int


class BatchFlopsCounter(Protocol):
    """A configured model's CPU-only, pre-context-parallel batch counter."""

    def __call__(self, batch: dict[str, torch.Tensor], *, ignore_index: int) -> BatchFlops:
        """Count useful training work without running model operations.

        Args:
            batch: CPU tensors with global shape [batch, sequence]: input_ids,
                labels, optional document-local position_ids and attention_mask.
                Positive attention-mask values indicate valid tokens; zero is padding.
            ignore_index: Ignored target value in addition to dataset sentinel -100.

        Returns:
            Effective model FLOPs and token counts, without mutating the batch.
        """
        ...
