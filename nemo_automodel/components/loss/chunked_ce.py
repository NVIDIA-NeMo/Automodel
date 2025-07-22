# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn.functional as F

_compiled_compute_cross_entropy = None


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index=-100,
):
    """Computes the cross-entropy loss between logits and targets.

    Args:
        logits (torch.Tensor): Model predictions of shape (sequence_length, num_classes).
        targets (torch.Tensor): Ground-truth labels of shape (sequence_length,).
        ignore_index (int, optional): Target value that is ignored when computing the loss.
            Defaults to -100.

    Returns:
        torch.Tensor: The sum of cross-entropy losses over the sequence.
    """
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="sum")


class ChunkedCrossEntropy:
    def __init__(self, chunk_len: int = 32, compile: bool = True, ignore_index: int = -100):
        """
        Chunked cross-entropy loss.

        Args:
            chunk_len (int, optional): The size of each chunk. The sequence will be split
                along the first dimension in chunks of this length. Defaults to 32.
            compile (bool, optional): If True, uses the compiled compute_cross_entropy function.
                Defaults to True.
            ignore_index (int, optional): Target value that is ignored when computing the loss.
                Defaults to -100.
        """
        self.chunk_len = chunk_len
        self.compile = compile
        self.ignore_index = ignore_index

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-entropy loss in chunks to handle long sequences more efficiently.

        Args:
            logits (torch.Tensor): Model output logits of shape [batch_size, seq_len, vocab_size].
            labels (torch.Tensor): Ground-truth labels of shape [batch_size, seq_len].
            mask (torch.Tensor, optional): Boolean mask indicating valid positions (1) and
                positions to ignore (0). Defaults to None.

        Returns:
            torch.Tensor: The sum of cross-entropy losses over the sequence.
        """
        # copied the following block from masked_ce
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask

        # maybe refactor if this is moved to a class?
        global _compiled_compute_cross_entropy
        if _compiled_compute_cross_entropy is None:
            _compiled_compute_cross_entropy = torch.compile(compute_cross_entropy, dynamic=True)

        seq_len = logits.shape[0]
        num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len
        loss = 0.0
        for logits_chunk, targets_chunk in zip(logits.chunk(num_chunks, dim=0), labels.chunk(num_chunks, dim=0)):
            loss += _compiled_compute_cross_entropy(logits_chunk, targets_chunk, self.ignore_index)
        return loss
