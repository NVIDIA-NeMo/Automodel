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


"""Softcapped cross entropy over the model's usable vocabulary."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.linear_ce_base import LinearCrossEntropy


def _usable_head_rows(weight: torch.Tensor, unpadded_vocab_size: int) -> torch.Tensor:
    """Keep the usable rows of a gathered head as the BF16 compute weight.

    Args:
        weight: Dense head weight of shape [padded_vocab, hidden].
        unpadded_vocab_size: Number of leading rows that enter the softmax.

    Returns:
        BF16 tensor of shape [unpadded_vocab_size, hidden]. Gradients reach the
        original weight; excluded rows receive zero.
    """
    if weight.shape[0] < unpadded_vocab_size:
        raise ValueError(f"lm_head has {weight.shape[0]} rows, fewer than unpadded_vocab_size={unpadded_vocab_size}")
    return weight[:unpadded_vocab_size].to(torch.bfloat16)


class Inkling2FusedLinearCrossEntropy(FusedLinearCrossEntropy):
    """Fused projection and cross entropy over the first ``unpadded_vocab_size`` head rows.

    The kernel projects BF16 hidden states against the BF16 head rows with FP32
    accumulation and applies the logit softcap and the cross entropy in FP32,
    which is the model's output contract. Rows past the usable vocabulary never
    enter the softmax. An FP32 head is rounded to BF16, the cast FSDP applies to
    the model's own projection.
    """

    def __init__(
        self,
        unpadded_vocab_size: int,
        *,
        logit_softcapping: float = 30.0,
        ignore_index: int = -100,
        reduction: str = "sum",
    ) -> None:
        if unpadded_vocab_size <= 0:
            raise ValueError("unpadded_vocab_size must be positive")
        if logit_softcapping <= 0:
            raise ValueError("logit_softcapping must be positive")
        super().__init__(ignore_index=ignore_index, logit_softcapping=logit_softcapping, reduction=reduction)
        self.unpadded_vocab_size = unpadded_vocab_size

    def materialize_lm_weight(
        self,
        lm_weight: torch.Tensor,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Gather the head and keep its usable rows as the BF16 compute weight.

        Args:
            lm_weight: Head weight with global shape [padded_vocab, hidden], a regular
                tensor or an FSDP-sharded DTensor.
            grad_reduce_group: Ranks contributing independent token losses, as in the
                shared materialization contract.

        Returns:
            BF16 tensor of shape [unpadded_vocab_size, hidden]. Gradients reach the
            original weight; excluded rows receive zero.
        """
        weight = super().materialize_lm_weight(lm_weight, grad_reduce_group=grad_reduce_group)
        return _usable_head_rows(weight, self.unpadded_vocab_size)


# cuBLAS runs a product whose vocabulary dimension is not a multiple of eight several times slower.
_VOCAB_ALIGNMENT = 8


def _token_losses_core(logits: torch.Tensor, targets: torch.Tensor, softcap: float, usable: int) -> torch.Tensor:
    """Compute the cross entropy of softcapped logits over the usable vocabulary.

    Args:
        logits: FP32 tensor of shape [tokens, padded_vocab].
        targets: Class indices of shape [tokens], below ``usable``.
        softcap: Positive tanh cap applied to the logits.
        usable: Number of leading columns that enter the softmax.

    Returns:
        FP32 losses of shape [tokens].
    """
    capped = softcap * torch.tanh(logits / softcap)
    columns = torch.arange(logits.shape[-1], device=logits.device)
    capped = torch.where(columns < usable, capped, float("-inf"))
    return torch.logsumexp(capped, dim=-1) - capped.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _logit_gradients_core(
    logits: torch.Tensor, targets: torch.Tensor, softcap: float, usable: int, upstream: torch.Tensor
) -> torch.Tensor:
    """Compute the gradient of the weighted token losses with respect to the logits.

    Args:
        logits: FP32 tensor of shape [tokens, padded_vocab].
        targets: Class indices of shape [tokens], below ``usable``.
        softcap: Positive tanh cap applied to the logits.
        usable: Number of leading columns that enter the softmax.
        upstream: FP32 gradient of each token loss, shape [tokens].

    Returns:
        FP32 tensor of shape [tokens, padded_vocab]; columns past ``usable`` are zero.
    """
    squashed = torch.tanh(logits / softcap)
    columns = torch.arange(logits.shape[-1], device=logits.device)
    capped = torch.where(columns < usable, softcap * squashed, float("-inf"))
    probabilities = torch.softmax(capped, dim=-1)
    is_target = columns == targets.unsqueeze(-1)
    return (probabilities - is_target.float()) * (1 - squashed.square()) * upstream.unsqueeze(-1)


def _losses_and_logit_gradients_core(
    logits: torch.Tensor, targets: torch.Tensor, softcap: float, usable: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the token losses and their gradients with respect to the logits in one pass.

    Args:
        logits: FP32 tensor of shape [tokens, padded_vocab].
        targets: Class indices of shape [tokens], below ``usable``.
        softcap: Positive tanh cap applied to the logits.
        usable: Number of leading columns that enter the softmax.

    Returns:
        FP32 losses of shape [tokens] and FP32 gradients of shape [tokens, padded_vocab] for a
        unit gradient of every token loss; columns past ``usable`` are zero.
    """
    squashed = torch.tanh(logits / softcap)
    columns = torch.arange(logits.shape[-1], device=logits.device)
    capped = torch.where(columns < usable, softcap * squashed, float("-inf"))
    losses = torch.logsumexp(capped, dim=-1) - capped.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    is_target = columns == targets.unsqueeze(-1)
    return losses, (torch.softmax(capped, dim=-1) - is_target.float()) * (1 - squashed.square())


# Compiled for static shapes first: a symbolic vocabulary width makes the generated kernels divide per element.
_compiled_token_losses = torch.compile(_token_losses_core, fullgraph=True)
_compiled_logit_gradients = torch.compile(_logit_gradients_core, fullgraph=True)
_compiled_losses_and_logit_gradients = torch.compile(_losses_and_logit_gradients_core, fullgraph=True)


def _wide_head(weight: torch.Tensor) -> torch.Tensor:
    """Return the head rows in FP32, zero-padded to an aligned number of rows.

    Args:
        weight: Head rows of shape [vocab, hidden].

    Returns:
        FP32 tensor of shape [padded_vocab, hidden] that does not alias the input.
    """
    rows = weight.shape[0] + -weight.shape[0] % _VOCAB_ALIGNMENT
    wide = torch.zeros(rows, weight.shape[1], dtype=torch.float32, device=weight.device)
    wide[: weight.shape[0]] = weight
    return wide


@contextmanager
def _tf32_products() -> Iterator[None]:
    """Run FP32 CUDA matrix products on tensor cores inside the block."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class _ChunkedSoftcappedCrossEntropy(torch.autograd.Function):
    """Project token chunks with FP32 accumulation; no [tokens, vocab] tensor outlives a chunk.

    The products run as TF32 GEMMs: BF16 operands are exact in TF32 and the sums are FP32.
    When a gradient is needed, the forward also computes both gradients for a unit gradient of every
    token loss, so a step projects the logits once and the backward scales them. Token losses with
    unequal upstream gradients (token weights), and a second backward through the same graph, recompute
    the chunks in the backward instead.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        softcap: float,
        chunk_tokens: int,
    ) -> torch.Tensor:
        """Compute token losses chunk by chunk.

        Args:
            ctx: Autograd context.
            hidden: Supervised hidden states of shape [tokens, hidden].
            weight: Head rows of shape [vocab, hidden].
            targets: Class indices of shape [tokens].
            softcap: Positive tanh cap applied to the logits.
            chunk_tokens: Largest number of tokens projected at once.

        Returns:
            FP32 losses of shape [tokens]. The inputs are not mutated.
        """
        token_losses = _compiled_token_losses if hidden.is_cuda else _token_losses_core
        losses_and_gradients = (
            _compiled_losses_and_logit_gradients if hidden.is_cuda else _losses_and_logit_gradients_core
        )
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        wide_weight = _wide_head(weight)
        losses = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device) if need_hidden else None
        grad_weight = torch.zeros_like(wide_weight) if need_weight else None
        with _tf32_products():
            for start in range(0, hidden.shape[0], chunk_tokens):
                rows = slice(start, start + chunk_tokens)
                wide_hidden = hidden[rows].float()
                logits = wide_hidden @ wide_weight.T
                if grad_hidden is None and grad_weight is None:
                    losses[rows] = token_losses(logits, targets[rows], softcap, weight.shape[0])
                    continue
                losses[rows], grad_logits = losses_and_gradients(logits, targets[rows], softcap, weight.shape[0])
                if grad_hidden is not None:
                    grad_hidden[rows] = grad_logits @ wide_weight
                if grad_weight is not None:
                    grad_weight.addmm_(grad_logits.T, wide_hidden)
        ctx.save_for_backward(hidden, weight, targets)
        ctx.unit_gradients = (grad_hidden, grad_weight)
        ctx.softcap = softcap
        ctx.chunk_tokens = chunk_tokens
        return losses

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_losses: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        """Scale the forward's gradients, or recompute the chunks when the token gradients differ.

        Args:
            ctx: Context holding hidden states [tokens, hidden], head rows
                [vocab, hidden] and targets [tokens].
            grad_losses: Upstream gradient of the token losses, shape [tokens].

        Returns:
            Gradients of the hidden states and the head rows in their own dtypes,
            followed by three None entries.
        """
        hidden, weight, targets = ctx.saved_tensors
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        upstream = grad_losses.float()
        # The forward's gradients serve one backward: release them here, not when the graph is collected.
        unit_hidden, unit_weight = ctx.unit_gradients
        ctx.unit_gradients = (None, None)
        cached = (unit_hidden is not None or not need_hidden) and (unit_weight is not None or not need_weight)
        if cached and (upstream.numel() == 0 or bool((upstream == upstream[0]).all())):
            # Every token loss has the same upstream gradient, the case of a summed or averaged loss.
            scale = upstream[0] if upstream.numel() else upstream.new_zeros(())
            return (
                unit_hidden.mul_(scale).to(hidden.dtype) if need_hidden else None,
                unit_weight[: weight.shape[0]].mul_(scale).to(weight.dtype) if need_weight else None,
                None,
                None,
                None,
            )
        del unit_hidden, unit_weight
        logit_gradients = _compiled_logit_gradients if hidden.is_cuda else _logit_gradients_core
        wide_weight = _wide_head(weight)
        grad_hidden = torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device) if need_hidden else None
        grad_weight = torch.zeros_like(wide_weight) if need_weight else None
        with _tf32_products():
            for start in range(0, hidden.shape[0], ctx.chunk_tokens):
                rows = slice(start, start + ctx.chunk_tokens)
                wide_hidden = hidden[rows].float()
                logits = wide_hidden @ wide_weight.T
                grad_logits = logit_gradients(logits, targets[rows], ctx.softcap, weight.shape[0], upstream[rows])
                if grad_hidden is not None:
                    grad_hidden[rows] = grad_logits @ wide_weight
                if grad_weight is not None:
                    grad_weight.addmm_(grad_logits.T, wide_hidden)
        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight[: weight.shape[0]].to(weight.dtype) if grad_weight is not None else None,
            None,
            None,
            None,
        )


class Inkling2ChunkedLinearCrossEntropy(LinearCrossEntropy):
    """Chunked projection and cross entropy over the first ``unpadded_vocab_size`` head rows.

    Each chunk of supervised tokens is projected against the BF16 head rows with
    FP32 accumulation, and the logit softcap and the cross entropy run in FP32,
    which is the model's output contract. The gradients are computed with the losses,
    chunk by chunk, so only one chunk of vocabulary activations is alive at a time.
    """

    def __init__(
        self,
        unpadded_vocab_size: int,
        *,
        logit_softcapping: float = 30.0,
        ignore_index: int = -100,
        reduction: str = "sum",
        chunk_tokens: int = 2048,
    ) -> None:
        super().__init__()
        if unpadded_vocab_size <= 0:
            raise ValueError("unpadded_vocab_size must be positive")
        if logit_softcapping <= 0:
            raise ValueError("logit_softcapping must be positive")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        self.unpadded_vocab_size = unpadded_vocab_size
        self.logit_softcapping = logit_softcapping
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_tokens = chunk_tokens

    def materialize_lm_weight(
        self,
        lm_weight: torch.Tensor,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Gather the head and keep its usable rows as the BF16 compute weight.

        Args:
            lm_weight: Head weight with global shape [padded_vocab, hidden], a regular
                tensor or an FSDP-sharded DTensor.
            grad_reduce_group: Ranks contributing independent token losses, as in the
                shared materialization contract.

        Returns:
            BF16 tensor of shape [unpadded_vocab_size, hidden]. Gradients reach the
            original weight; excluded rows receive zero.
        """
        weight = super().materialize_lm_weight(lm_weight, grad_reduce_group=grad_reduce_group)
        return _usable_head_rows(weight, self.unpadded_vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: torch.Tensor,
        num_label_tokens: int | None = None,
        grad_reduce_group: dist.ProcessGroup | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the softcapped cross entropy from final hidden states.

        Args:
            hidden_states: Rank-local hidden states of shape [batch, sequence, hidden],
                or packed [tokens, hidden] / [1, tokens, hidden]; token order matches
                the flattened labels.
            labels: Target token IDs of shape [batch, sequence].
            lm_weight: Head weight with global shape [padded_vocab, hidden], a regular
                tensor or an FSDP-sharded DTensor; rows this loss already materialized pass through.
            num_label_tokens: Global number of supervised tokens that normalizes a
                sum-reduced loss.
            grad_reduce_group: Ranks contributing independent token losses when
                ``lm_weight`` is a sharded DTensor.
            loss_weights: Optional per-token multipliers of the labels' shape, for
                ``reduction="sum"`` only.

        Returns:
            FP32 scalar, or losses of the labels' shape for ``reduction="none"``. The
            inputs are not mutated.
        """
        if (num_label_tokens is not None or loss_weights is not None) and self.reduction != "sum":
            raise ValueError("num_label_tokens and loss_weights are only supported when reduction is 'sum'")
        if loss_weights is not None and loss_weights.shape != labels.shape:
            raise ValueError(
                f"loss_weights.shape must match labels.shape, got {tuple(loss_weights.shape)} and {tuple(labels.shape)}"
            )
        weight = self.materialize_lm_weight(lm_weight, grad_reduce_group=grad_reduce_group)
        targets = labels.to(hidden_states.device).reshape(-1)
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden.shape[0] != targets.shape[0]:
            raise ValueError("hidden_states token dimensions must match labels.shape")
        supervised = (targets != self.ignore_index).nonzero().squeeze(-1)
        if supervised.shape[0] != targets.shape[0]:
            hidden, targets = hidden[supervised], targets[supervised]
        losses = _ChunkedSoftcappedCrossEntropy.apply(
            hidden, weight, targets, self.logit_softcapping, self.chunk_tokens
        )
        if loss_weights is not None:
            token_weights = loss_weights.to(device=losses.device, dtype=torch.float32).reshape(-1)
            losses = losses * (
                token_weights if supervised.shape[0] == token_weights.shape[0] else token_weights[supervised]
            )
        if self.reduction == "none":
            return (
                torch.zeros(labels.numel(), dtype=torch.float32, device=losses.device)
                .index_copy(0, supervised, losses)
                .reshape(labels.shape)
            )
        loss = losses.sum()
        if self.reduction == "mean":
            loss = loss / max(supervised.shape[0], 1)
        if num_label_tokens is not None:
            loss = loss * 0.0 if num_label_tokens == 0 else loss / num_label_tokens
        return loss
