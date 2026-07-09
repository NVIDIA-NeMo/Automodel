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
from torch.distributed.tensor import DTensor


def _validate_chunk_size(chunk_size: int) -> int:
    """Validate that ``chunk_size`` is a positive integer and return it as ``int``."""
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero; got {chunk_size}.")
    return chunk_size


class _ChunkedMaskedCESum(torch.autograd.Function):
    """Sum-reduced masked cross-entropy computed one fp32 chunk at a time.

    Numerically equivalent (up to fp32 accumulation order) to
    ``F.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction="sum")``,
    but never materializes more than one fp32 ``[chunk_size, V]`` slice:

    - forward: per-chunk fp32 logsumexp + label-logit gather, accumulated into a
      scalar; only the original-dtype logits and the labels are saved for backward
      (``cross_entropy`` would additionally save its fp32 log-softmax).
    - backward: recomputes the softmax per chunk from the saved logits and writes
      ``(softmax - onehot) * grad_out`` directly into the gradient buffer.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        chunk_size: int,
        inplace_grad: bool,
    ) -> torch.Tensor:
        """Compute the sum-reduced masked cross-entropy.

        Args:
            ctx: Autograd context used to stash tensors for backward.
            logits: Prediction scores of shape ``[N, V]`` (``N`` = tokens, ``V`` =
                vocabulary size) in any floating dtype. When ``inplace_grad`` is
                True, backward reuses this tensor's storage as the gradient
                buffer, overwriting its values.
            labels: Target class indices of shape ``[N]``. Positions equal to
                ``ignore_index`` contribute zero loss and zero gradient.
            ignore_index: Label value to ignore.
            chunk_size: Number of rows upcast to fp32 per chunk.
            inplace_grad: If True, backward writes gradients into the logits storage.

        Returns:
            Scalar fp32 tensor holding the sum-reduced loss.
        """
        chunk_size = _validate_chunk_size(chunk_size)
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "_ChunkedMaskedCESum requires logits shaped [N, V] and labels shaped [N]; "
                f"got logits.shape={tuple(logits.shape)} and labels.shape={tuple(labels.shape)}."
            )
        valid = labels != ignore_index
        # cross_entropy(ignore_index) semantics: ignored rows contribute 0.
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, logits.shape[0], chunk_size):
            end = min(start + chunk_size, logits.shape[0])
            chunk = logits[start:end].float()
            lse = torch.logsumexp(chunk, dim=-1)
            picked = chunk.gather(1, safe_labels[start:end].unsqueeze(1)).squeeze(1)
            total = total + ((lse - picked) * valid[start:end]).sum()
        ctx.save_for_backward(logits, safe_labels, valid)
        ctx.chunk_size = chunk_size
        ctx.inplace_grad = inplace_grad
        return total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """Compute the gradient with respect to the logits, one chunk at a time.

        Args:
            ctx: Autograd context holding the tensors saved in forward.
            grad_out: Scalar gradient of the loss output.

        Returns:
            Tuple whose first element is the ``[N, V]`` gradient w.r.t. the logits
            (in the logits' original dtype); the remaining elements are ``None``
            for the non-tensor forward arguments. When ``inplace_grad`` is True the
            gradient aliases the logits tensor saved in forward.
        """
        logits, safe_labels, valid = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        if ctx.inplace_grad:
            # Reuse the saved logits storage as the grad buffer: each chunk is
            # copied to fp32 BEFORE its region is overwritten, and the producing
            # matmul's backward needs only grad_output, its saved input, and its
            # weight — not the logits values. This removes a full [N, V]
            # allocation at the loss-backward memory peak.
            grad = logits
        else:
            grad = torch.empty_like(logits)
        for start in range(0, logits.shape[0], chunk_size):
            end = min(start + chunk_size, logits.shape[0])
            chunk = logits[start:end].float()
            grad_chunk = torch.softmax(chunk, dim=-1)
            grad_chunk.scatter_add_(
                1,
                safe_labels[start:end].unsqueeze(1),
                torch.full((end - start, 1), -1.0, dtype=torch.float32, device=chunk.device),
            )
            grad_chunk = grad_chunk * (valid[start:end].unsqueeze(1) * grad_out)
            grad[start:end] = grad_chunk.to(grad.dtype)
        return grad, None, None, None, None


class MaskedCrossEntropy(nn.Module):
    """Cross-entropy loss that handles ignored or masked target positions."""

    def __init__(
        self,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
        chunk_size: int | None = None,
        inplace_grad: bool = True,
    ):
        """
        Masked cross-entropy loss.

        Args:
            fp32_upcast (bool): if True it will cast logits to float32 before computing
                cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): type of reduction. Defaults to "sum".
            chunk_size (int | None): if set, computes the loss one fp32
                ``[chunk_size, V]`` slice at a time instead of upcasting the full
                logits tensor, bounding the fp32 transient regardless of sequence
                length. Only the original-dtype logits are saved for backward.
                Requires ``reduction="sum"`` and ``fp32_upcast=True``. Defaults to
                None (full-tensor path).
            inplace_grad (bool): only used when ``chunk_size`` is set. If True,
                backward writes the logits gradient into the logits tensor's
                storage instead of allocating a new ``[N, V]`` buffer. Safe as
                long as no other autograd node consumes the logits *values* in
                backward (the producing linear layer does not). Defaults to True.
        """
        super().__init__()
        if chunk_size is not None:
            chunk_size = _validate_chunk_size(chunk_size)
            if reduction != "sum":
                raise ValueError("chunk_size requires reduction='sum'.")
            if not fp32_upcast:
                raise ValueError("chunk_size requires fp32_upcast=True; the chunked path upcasts each chunk to fp32.")
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.inplace_grad = inplace_grad

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the masked cross-entropy loss between logits and targets.

        If a mask is provided, the loss is computed per element, multiplied by the mask,
        and then averaged. If no mask is provided, the standard cross-entropy loss is used.

        Args:
            logits (torch.Tensor): The predicted logits with shape [batch_size, seq_len, vocab_size] where C is the number of classes.
            labels (torch.Tensor): The ground truth class indices with shape [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
                1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
                of the loss. Defaults to None.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        if self.chunk_size is not None:
            # The chunked kernel flattens logits/labels itself, so a silent
            # shape mismatch would mis-align tokens; reject it up front.
            chunk_size = _validate_chunk_size(self.chunk_size)
            if logits.ndim < 1 or tuple(logits.shape[:-1]) != tuple(labels.shape):
                raise ValueError(
                    "chunked MaskedCrossEntropy requires logits.shape[:-1] == labels.shape; "
                    f"got logits.shape={tuple(logits.shape)} and labels.shape={tuple(labels.shape)}."
                )
            if mask is not None and tuple(mask.shape) != tuple(labels.shape):
                raise ValueError(
                    "chunked MaskedCrossEntropy requires mask.shape == labels.shape; "
                    f"got mask.shape={tuple(mask.shape)} and labels.shape={tuple(labels.shape)}."
                )
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
        if self.fp32_upcast and self.chunk_size is None:
            logits = logits.float()

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        if self.chunk_size is not None:
            loss = _ChunkedMaskedCESum.apply(logits, labels, self.ignore_index, chunk_size, self.inplace_grad)
        else:
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens
        return loss
