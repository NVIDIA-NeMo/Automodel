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
import torch.nn as nn
import torch.nn.functional as F

_compiled_compute_cross_entropy = None


def _validate_chunk_len(chunk_len: int) -> int:
    """Validate that ``chunk_len`` is positive."""
    chunk_len = int(chunk_len)
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be greater than zero; got {chunk_len}.")
    return chunk_len


class _ChunkedCrossEntropySum(torch.autograd.Function):
    """Sum-reduced cross-entropy with recomputed fp32 chunk activations."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        chunk_len: int,
    ) -> torch.Tensor:
        """Compute sum-reduced cross-entropy one fp32 chunk at a time.

        Args:
            ctx: Autograd context used to save tensors for backward.
            logits: Prediction scores of shape [tokens, vocab]. The tensor stays
                in its original floating-point dtype and is not mutated.
            labels: Target class indices of shape [tokens]. Positions equal to
                ``ignore_index`` contribute zero loss and gradient.
            ignore_index: Target value excluded from the loss.
            chunk_len: Maximum number of token rows upcast to fp32 at once.

        Returns:
            Scalar fp32 tensor containing the sum-reduced loss.
        """
        chunk_len = _validate_chunk_len(chunk_len)
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "_ChunkedCrossEntropySum requires logits shaped [tokens, vocab] and labels shaped [tokens]; "
                f"got logits.shape={tuple(logits.shape)} and labels.shape={tuple(labels.shape)}."
            )

        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, logits.shape[0], chunk_len):
            end = min(start + chunk_len, logits.shape[0])
            logits_chunk = logits[start:end].float()
            log_normalizer = torch.logsumexp(logits_chunk, dim=-1)
            target_logits = logits_chunk.gather(1, safe_labels[start:end].unsqueeze(1)).squeeze(1)
            total = total + ((log_normalizer - target_logits) * valid[start:end]).sum()

        ctx.save_for_backward(logits, safe_labels, valid)
        ctx.chunk_len = chunk_len
        return total

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Recompute the fp32 softmax chunks and return the logits gradient.

        Args:
            ctx: Autograd context containing the original-dtype tensors saved by
                :meth:`forward`.
            grad_out: Scalar tensor containing the upstream loss gradient.

        Returns:
            Tuple whose first element is the logits gradient of shape [tokens,
            vocab] in the logits' original dtype. The remaining entries are
            ``None`` for non-differentiable inputs.
        """
        logits, safe_labels, valid = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for start in range(0, logits.shape[0], ctx.chunk_len):
            end = min(start + ctx.chunk_len, logits.shape[0])
            logits_chunk = logits[start:end].float()
            grad_chunk = torch.softmax(logits_chunk, dim=-1)
            grad_chunk.scatter_add_(
                1,
                safe_labels[start:end].unsqueeze(1),
                torch.full((end - start, 1), -1.0, dtype=torch.float32, device=logits.device),
            )
            grad_chunk = grad_chunk * (valid[start:end].unsqueeze(1) * grad_out)
            grad[start:end] = grad_chunk.to(grad.dtype)
        return grad, None, None, None


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index=-100,
    reduction="sum",
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
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction)


class ChunkedCrossEntropy(nn.Module):
    """Cross-entropy loss computed over sequence chunks."""

    def __init__(
        self,
        chunk_len: int = 32,
        compile: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
    ):
        """
        Chunked cross-entropy loss.

        With the default ``reduction="sum"`` the loss is computed by the
        memory-efficient chunked kernel (:class:`_ChunkedCrossEntropySum`): each
        ``[chunk_len, V]`` slice is upcast to fp32 transiently, only the
        original-dtype logits are saved for backward, and the softmax is
        recomputed per chunk in backward. Other reductions fall back to the
        legacy per-chunk ``torch.compile``-d ``F.cross_entropy`` loop.

        Args:
            chunk_len (int, optional): The size of each chunk. The sequence will be split
                along the first dimension in chunks of this length. Defaults to 32.
            compile (bool, optional): If True, uses the compiled compute_cross_entropy function
                on the legacy (non-"sum") path. The "sum" path uses the chunked kernel and
                does not involve ``torch.compile``. Defaults to True.
            ignore_index (int, optional): Target value that is ignored when computing the loss.
                Defaults to -100.
            reduction (str, optional): Type of reduction. Defaults to "sum".
        """
        super().__init__()
        self.chunk_len = _validate_chunk_len(chunk_len)
        self.compile = compile
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes cross-entropy loss in chunks to handle long sequences more efficiently.

        Args:
            logits: Model output tensor of shape [..., vocab], with arbitrary
                leading token dimensions.
            labels: Target indices of shape [...] matching ``logits.shape[:-1]``.
                When ``mask`` is provided, ignored positions are replaced with
                ``ignore_index`` in this tensor.
            mask: Optional tensor of shape [...] matching ``labels``. Nonzero
                positions contribute to the loss and zero positions are ignored.
            num_label_tokens: Optional global count used to normalize the
                sum-reduced scalar loss.

        Returns:
            Scalar tensor containing the reduced cross-entropy loss.
        """
        # copied the following block from masked_ce
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)  # pragma: no cover
        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)  # pragma: no cover
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask

        if self.reduction == "sum":
            # Save only original-dtype logits and recompute each fp32 softmax
            # chunk in backward.
            loss = _ChunkedCrossEntropySum.apply(logits, labels, self.ignore_index, self.chunk_len)
        else:
            compute_loss = compute_cross_entropy
            if self.compile:
                global _compiled_compute_cross_entropy
                if _compiled_compute_cross_entropy is None:
                    _compiled_compute_cross_entropy = torch.compile(compute_cross_entropy, dynamic=True)
                compute_loss = _compiled_compute_cross_entropy

            seq_len = logits.shape[0]
            num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len
            loss = 0.0
            for logits_chunk, targets_chunk in zip(logits.chunk(num_chunks, dim=0), labels.chunk(num_chunks, dim=0)):
                loss += compute_loss(logits_chunk, targets_chunk, self.ignore_index, self.reduction)
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            loss = loss / num_label_tokens  # pragma: no cover
        return loss
