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

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.loss.linear_ce_base import LinearCrossEntropy


def _validate_chunk_len(chunk_len: int) -> int:
    """Validate that ``chunk_len`` is positive."""
    chunk_len = int(chunk_len)
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be greater than zero; got {chunk_len}.")
    return chunk_len


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute fp32 CE.

    Args:
        logits: Scores shaped [tokens, vocab].
        targets: Class indices shaped [tokens].
        ignore_index: Ignored target value.
        reduction: PyTorch CE reduction.

    Returns:
        Scalar loss, or [tokens] losses for reduction="none".
    """
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction)


def _linear_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Project one token chunk and compute CE.

    Args:
        hidden: Local hidden states shaped [chunk_tokens, hidden].
        weight: Dense LM-head weight shaped [vocab, hidden].
        labels: Target indices shaped [chunk_tokens].
        ignore_index: Ignored target value.

    Returns:
        FP32 losses shaped [chunk_tokens].
    """
    return compute_cross_entropy(F.linear(hidden, weight), labels, ignore_index, "none")


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        chunk_len: int,
        ignore_index: int,
        compute_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
    ) -> torch.Tensor:
        """Compute chunk losses without retaining logits.

        Args:
            ctx: Autograd context.
            hidden: Local hidden states shaped [tokens, hidden].
            weight: Dense projection weight shaped [vocab, hidden].
            labels: Target indices shaped [tokens].
            chunk_len: Maximum token rows per projection.
            ignore_index: Ignored target value.
            compute_loss: Eager or compiled per-chunk loss.

        Returns:
            FP32 losses shaped [tokens]; inputs are not mutated.
        """
        losses = torch.empty(labels.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk_len):
            end = start + chunk_len
            losses[start:end] = compute_loss(hidden[start:end], weight, labels[start:end], ignore_index)
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_len = chunk_len
        ctx.ignore_index = ignore_index
        ctx.compute_loss = compute_loss
        return losses

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        """Recompute one projection/CE graph at a time.

        Args:
            ctx: Context holding [tokens, hidden] states, [vocab, hidden]
                weight and [tokens] labels.
            grad_out: Upstream loss gradients shaped [tokens].

        Returns:
            Hidden-state and weight gradients matching the original shapes and
            dtypes, followed by four None entries. No input is mutated.
        """
        hidden, weight, labels = ctx.saved_tensors
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        # Accumulate across chunks in fp32, avoiding repeated bf16 rounding.
        accum_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype) if need_weight else None
        with torch.enable_grad():
            local_weight = weight.detach().requires_grad_(need_weight)
            for start in range(0, hidden.shape[0], ctx.chunk_len):
                end = start + ctx.chunk_len
                local_hidden = hidden[start:end].detach().requires_grad_(need_hidden)
                inputs = [x for x in (local_hidden, local_weight) if x.requires_grad]
                loss = ctx.compute_loss(local_hidden, local_weight, labels[start:end], ctx.ignore_index)
                grads = torch.autograd.grad(loss, inputs, grad_out[start:end])
                if need_hidden:
                    grad_hidden[start:end] = grads[0]
                if need_weight:
                    grad_weight.add_(grads[-1])
                del loss, grads
        return grad_hidden, grad_weight.to(weight.dtype) if need_weight else None, None, None, None, None


class ChunkedCrossEntropy(LinearCrossEntropy):
    """Chunk the LM-head projection and CE, recomputing each chunk in backward.

    Unlike a logits-based loss, this consumes final hidden states and the
    projection weight. Neither full logits nor a full logits gradient is
    allocated. Only one chunk's vocabulary activations are live at a time.
    Higher-order gradients are not supported.
    """

    def __init__(
        self,
        chunk_len: int = 512,
        compile: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
    ) -> None:
        """Initialize chunked projection/CE.

        Args:
            chunk_len: Maximum token rows per chunk; must be positive.
            compile: Compile the per-chunk projection and CE, including backward.
            ignore_index: Ignored target value.
            reduction: "sum", "mean", or "none" across all valid tokens.
        """
        super().__init__()
        chunk_len = _validate_chunk_len(chunk_len)
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        self.chunk_len = chunk_len
        self.compile = compile
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._compute_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor] = (
            torch.compile(_linear_cross_entropy, dynamic=True) if compile else _linear_cross_entropy
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: torch.Tensor,
        num_label_tokens: int | None = None,
        grad_reduce_group: dist.ProcessGroup | None = None,
        loss_weights: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CE directly from hidden states, with no full-logit allocation.

        Args:
            hidden_states: Rank-local states shaped [..., hidden], with arbitrary
                leading token dimensions, including packed [tokens, hidden].
            labels: Target indices shaped [...] matching the token dimensions.
            lm_weight: Weight shaped [vocab, hidden]; FSDP DTensors use the
                LinearCrossEntropy.materialize_lm_weight contract. TP-sharded
                vocabulary/hidden states are not supported by this loss.
            num_label_tokens: Global count normalizing a sum-reduced loss.
            grad_reduce_group: DP/CP group contributing independent token losses.
            loss_weights: Optional constant multipliers shaped [...], sum only.
            mask: Optional mask shaped [...]; zero positions are ignored.

        Returns:
            FP32 scalar, or losses shaped [...] for reduction="none". Inputs
            are not mutated. A fully ignored sum is zero with zero gradients.
        """
        if hidden_states.shape[:-1] != labels.shape:
            raise ValueError("hidden_states token dimensions must match labels.shape")
        if lm_weight.ndim != 2 or hidden_states.shape[-1] != lm_weight.shape[-1]:
            raise ValueError("lm_weight must have shape [vocab, hidden] matching hidden_states")
        if (num_label_tokens is not None or loss_weights is not None) and self.reduction != "sum":
            raise ValueError("num_label_tokens and loss_weights are only supported when reduction is 'sum'")
        labels = labels.to(hidden_states.device)
        if mask is not None:
            if mask.shape != labels.shape:
                raise ValueError("mask.shape must match labels.shape")
            labels = labels.masked_fill(mask.to(labels.device) == 0, self.ignore_index)
        if loss_weights is not None and loss_weights.shape != labels.shape:
            raise ValueError("loss_weights.shape must match labels.shape")
        weight = self.materialize_lm_weight(lm_weight, grad_reduce_group=grad_reduce_group)
        # FSDP can expose fp32 master weights outside the head's forward.
        weight = weight.to(hidden_states.dtype)
        losses = _ChunkedLinearCE.apply(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            weight,
            labels.reshape(-1),
            self.chunk_len,
            self.ignore_index,
            self._compute_loss,
        ).reshape(labels.shape)
        if loss_weights is not None:
            losses = losses * loss_weights.to(device=losses.device, dtype=torch.float32)
        if self.reduction == "none":
            return losses
        loss = losses.sum()
        if self.reduction == "mean":
            loss = loss / (labels != self.ignore_index).sum()
        if num_label_tokens is not None:
            loss = loss * 0.0 if num_label_tokens == 0 else loss / num_label_tokens
        return loss
