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

"""LM-head weight handling shared by the linear-projection cross-entropy losses.

A linear CE loss consumes the LM-head weight outside the owning FSDP module's
forward. Gathering an FSDP-sharded head whole would hold a full
``[vocab, hidden]`` weight, plus a full-size gradient during backward, on every
rank. Instead, :class:`ShardedLMHeadWeight` streams the head in vocabulary
chunks. Forward all-gathers one chunk at a time and merges its log-sum-exp.
Backward re-gathers each chunk and reduce-scatters its gradient straight into
the local shard.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

# Size budget for one gathered vocabulary chunk, in bytes. Backward holds about four
# chunk-sized buffers at once: the current chunk, the prefetched next chunk, the
# chunk gradient, and the previous gradient while it is reduce-scattered.
_STREAM_CHUNK_BYTES = 128 * 1024**2


class VocabChunkKernel(Protocol):
    """Projection and cross-entropy math for one vocabulary chunk of an LM head."""

    def forward(
        self, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the chunk's log-sum-exp and target logits.

        Args:
            hidden: Tensor of shape [tokens, hidden] holding supervised tokens only;
                contiguous.
            weight: Tensor of shape [chunk_vocab, hidden] holding a subset of the
                LM-head rows; contiguous.
            targets: Int64 tensor of shape [tokens] with each token's target row
                in ``weight``, or -1 when the target row is in another chunk.

        Returns:
            Two FP32 tensors of shape [tokens]: the log-sum-exp over the chunk's
            logits, and each token's target logit (zero where ``targets`` is -1).
        """
        ...

    def backward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        lse: torch.Tensor,
        grad: torch.Tensor,
        *,
        need_hidden: bool,
        need_weight: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the chunk's share of the hidden-state and weight gradients.

        Each token's loss is ``lse - target_logit`` over the whole vocabulary, so
        the logit gradient is ``grad * (exp(logit - lse) - onehot(target))``.

        Args:
            hidden: Tensor of shape [tokens, hidden], as in :meth:`forward`.
            weight: Tensor of shape [chunk_vocab, hidden], as in :meth:`forward`.
            targets: Int64 tensor of shape [tokens], as in :meth:`forward`.
            lse: FP32 tensor of shape [tokens], the log-sum-exp over the full
                vocabulary.
            grad: FP32 tensor of shape [tokens], the upstream gradient of each
                token's loss.
            need_hidden: Whether to return the hidden-state gradient.
            need_weight: Whether to return the weight gradient.

        Returns:
            The hidden-state gradient of shape [tokens, hidden] in any floating
            dtype, and the weight gradient of shape [chunk_vocab, hidden] in
            ``weight.dtype``. Each is None when not requested.
        """
        ...


@dataclass(frozen=True)
class ShardedLMHeadWeight:
    """FSDP-sharded LM-head weight that linear CE losses stream in vocabulary chunks.

    The weight is never gathered whole: see :func:`sharded_linear_cross_entropy`.
    Its gradient is averaged over the weight's whole mesh, whose ranks own
    independent token losses (``resolve_lm_weight`` checks that the loss's
    reduction group has the mesh's size).

    Attributes:
        weight: DTensor of global shape [vocab, hidden], sharded on the
            vocabulary axis (``Shard(0)``) over exactly one mesh dimension and
            replicated over any others (an FSDP or HSDP layout).
        dtype: Projection dtype. Each shard chunk is cast to it before the
            all-gather; the weight gradient keeps ``weight.dtype``.
    """

    weight: DTensor
    dtype: torch.dtype

    @property
    def shape(self) -> torch.Size:
        """Global ``[vocab, hidden]`` shape of the weight."""
        return self.weight.shape


class _PendingChunk:
    """An in-flight all-gather of one vocabulary chunk; owns its buffers until waited."""

    def __init__(self, work: dist.Work, gathered: torch.Tensor, send: torch.Tensor, num_rows: int):
        """Track one all-gather.

        Args:
            work: Handle of the asynchronous all-gather.
            gathered: Receive buffer of shape [ranks * chunk_rows, hidden].
            send: Send buffer of shape [chunk_rows, hidden]; kept alive until the
                all-gather completes.
            num_rows: Leading rows of ``gathered`` that hold real vocabulary rows.
        """
        self._work = work
        self._gathered = gathered
        self._send: torch.Tensor | None = send
        self._num_rows = num_rows

    def wait(self) -> torch.Tensor:
        """Return the gathered chunk of shape [chunk_vocab, hidden].

        The result is a contiguous view of the receive buffer without the padding
        rows of short trailing shards.
        """
        self._work.wait()
        self._send = None
        return self._gathered[: self._num_rows]


class _PendingGradient:
    """An in-flight reduce-scatter of one chunk gradient; owns its buffers until finished."""

    def __init__(self, work: dist.Work, send: torch.Tensor, received: torch.Tensor, start: int, num_local_rows: int):
        """Track one reduce-scatter.

        Args:
            work: Handle of the asynchronous reduce-scatter.
            send: Send buffer of shape [ranks * chunk_rows, hidden]; kept alive
                until the reduce-scatter completes.
            received: Receive buffer of shape [chunk_rows, hidden].
            start: First local shard row this chunk covers.
            num_local_rows: Number of received rows that exist in the local shard.
        """
        self._work = work
        self._send: torch.Tensor | None = send
        self._received = received
        self._start = start
        self._num_local_rows = num_local_rows

    def finish(self, grad_shard: torch.Tensor, replicate_groups: list[dist.ProcessGroup]) -> None:
        """Wait for the reduce-scatter and write the result into ``grad_shard``.

        Args:
            grad_shard: Tensor of shape [local_vocab, hidden], this rank's
                weight-gradient shard; written in place.
            replicate_groups: Groups of the mesh dimensions the weight is
                replicated over. Their partial sums are all-reduced here.
        """
        self._work.wait()
        self._send = None
        for group in replicate_groups:
            dist.all_reduce(self._received, group=group)
        grad_shard[self._start : self._start + self._num_local_rows] = self._received[: self._num_local_rows]


class _VocabShardLayout:
    """Chunking and collectives for an LM head sharded on its vocabulary axis.

    Shards follow ``torch.chunk`` semantics, as in FSDP2 and DTensor: rank ``r``
    holds rows ``[r * rows_per_rank, min((r + 1) * rows_per_rank, vocab))``, so
    trailing ranks may hold fewer rows, or none. Chunk ``k`` takes local rows
    ``[k * chunk_rows, (k + 1) * chunk_rows)`` from every rank. Its gathered rows
    are ordered by rank, and the rows that exist form a prefix of the gathered
    buffer because only trailing ranks are short.
    """

    def __init__(self, lm_weight: ShardedLMHeadWeight):
        """Validate the layout and plan the chunks.

        Args:
            lm_weight: Sharded LM head of global shape [vocab, hidden].

        Raises:
            ValueError: If the weight is not ``Shard(0)`` over one mesh dimension
                and ``Replicate()`` over the others, or its local shard does not
                match that layout.
        """
        weight = lm_weight.weight
        mesh = weight.device_mesh
        placements = tuple(weight.placements)
        shard_dims = [dim for dim, placement in enumerate(placements) if type(placement) is Shard]
        replicate_dims = [dim for dim, placement in enumerate(placements) if type(placement) is Replicate]
        if (
            weight.ndim != 2
            or len(shard_dims) != 1
            or placements[shard_dims[0]].dim != 0
            or len(shard_dims) + len(replicate_dims) != mesh.ndim
        ):
            raise ValueError(
                "Linear cross entropy streams an LM head sharded on its vocabulary axis: expected a 2-D weight "
                f"with one Shard(0) mesh dimension and Replicate() on the others, got shape {tuple(weight.shape)} "
                f"and placements {placements}."
            )
        shard_dim = shard_dims[0]
        self.group = mesh.get_group(shard_dim)
        self.replicate_groups = [mesh.get_group(dim) for dim in replicate_dims]
        self.world_size = mesh.size(shard_dim)
        self.reduce_world_size = mesh.size()
        self.dtype = lm_weight.dtype
        self.vocab, self.hidden = weight.shape
        self.rows_per_rank = -(-self.vocab // self.world_size)
        rank = mesh.get_local_rank(shard_dim)
        expected_local_rows = max(0, min(self.rows_per_rank, self.vocab - rank * self.rows_per_rank))
        with torch.no_grad():
            local_shape = tuple(weight.to_local().shape)
        if local_shape != (expected_local_rows, self.hidden):
            raise ValueError(
                f"LM-head shard on rank {rank} has shape {local_shape}; Shard(0) of {tuple(weight.shape)} over "
                f"{self.world_size} ranks expects {(expected_local_rows, self.hidden)}."
            )
        row_bytes = self.world_size * self.hidden * torch.empty((), dtype=self.dtype).element_size()
        self.chunk_rows = min(self.rows_per_rank, max(1, _STREAM_CHUNK_BYTES // row_bytes))
        self.num_chunks = -(-self.rows_per_rank // self.chunk_rows)

    def chunk_size(self, chunk: int) -> int:
        """Return the number of rows each rank contributes to ``chunk``."""
        return min(self.chunk_rows, self.rows_per_rank - chunk * self.chunk_rows)

    def chunk_targets(self, owner: torch.Tensor, offset: torch.Tensor, chunk: int) -> torch.Tensor:
        """Map global targets to rows of the gathered ``chunk``.

        Args:
            owner: Int64 tensor of shape [tokens], the rank whose shard holds
                each token's target row.
            offset: Int64 tensor of shape [tokens], the target row within that
                shard.
            chunk: Chunk index.

        Returns:
            Int64 tensor of shape [tokens] with the gathered-chunk row of each
            target, or -1 for targets in another chunk.
        """
        rows = self.chunk_size(chunk)
        local = offset - chunk * self.chunk_rows
        in_chunk = (local >= 0) & (local < rows)
        return torch.where(in_chunk, owner * rows + local, -1)

    def _start_gather(self, shard: torch.Tensor, chunk: int) -> _PendingChunk:
        """Start the all-gather of ``chunk``.

        Args:
            shard: Tensor of shape [local_vocab, hidden], this rank's shard.
            chunk: Chunk index.

        Returns:
            The pending all-gather.
        """
        rows = self.chunk_size(chunk)
        start = chunk * self.chunk_rows
        local = shard[start : start + rows].to(self.dtype)
        if local.shape[0] < rows:
            local = torch.cat([local, local.new_zeros(rows - local.shape[0], self.hidden)])
        local = local.contiguous()
        gathered = local.new_empty(self.world_size * rows, self.hidden)
        work = dist.all_gather_into_tensor(gathered, local, group=self.group, async_op=True)
        # Rows past the vocabulary only pad the trailing shards, so the real rows are a prefix.
        real_rows = sum(
            max(0, min(rows, self.vocab - rank * self.rows_per_rank - start)) for rank in range(self.world_size)
        )
        return _PendingChunk(work, gathered, local, real_rows)

    def stream(self, shard: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield each gathered chunk while the next one is in flight.

        Every rank must consume the whole iterator: each step enters a collective.

        Args:
            shard: Tensor of shape [local_vocab, hidden], this rank's shard.

        Yields:
            The chunk index and a contiguous tensor of shape [chunk_vocab, hidden]
            holding the chunk's rows in ``self.dtype``.
        """
        pending = self._start_gather(shard, 0)
        for chunk in range(self.num_chunks):
            gathered = pending.wait()
            if chunk + 1 < self.num_chunks:
                pending = self._start_gather(shard, chunk + 1)
            yield chunk, gathered

    def start_reduce(self, grad_chunk: torch.Tensor, chunk: int, grad_shard: torch.Tensor) -> _PendingGradient:
        """Start averaging one chunk gradient over the mesh into its owners' shards.

        Args:
            grad_chunk: Tensor of shape [chunk_vocab, hidden], this rank's
                gradient for the gathered chunk rows. It is consumed: when no
                padding or dtype change is needed it is scaled in place and sent.
            chunk: Chunk index.
            grad_shard: Tensor of shape [local_vocab, hidden], this rank's
                gradient shard; it fixes the reduction dtype.

        Returns:
            The pending reduce-scatter.
        """
        rows = self.chunk_size(chunk)
        start = chunk * self.chunk_rows
        if grad_chunk.shape[0] == self.world_size * rows and grad_chunk.dtype == grad_shard.dtype:
            send = grad_chunk  # the kernel's own buffer; nothing else reads it
        else:
            send = grad_chunk.new_zeros(self.world_size * rows, self.hidden, dtype=grad_shard.dtype)
            send[: grad_chunk.shape[0]] = grad_chunk
        # Recipes scale each rank's loss by the reduction world size to cancel
        # FSDP's averaged-gradient convention, so average rather than sum.
        send.div_(self.reduce_world_size)
        received = send.new_empty(rows, self.hidden)
        work = dist.reduce_scatter_tensor(received, send, group=self.group, async_op=True)
        num_local_rows = max(0, min(rows, grad_shard.shape[0] - start))
        return _PendingGradient(work, send, received, start, num_local_rows)


class _StreamedLinearCrossEntropy(torch.autograd.Function):
    """Per-token CE against an LM head that is gathered one vocabulary chunk at a time."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        shard: torch.Tensor,
        targets: torch.Tensor,
        layout: _VocabShardLayout,
        kernel: VocabChunkKernel,
    ) -> torch.Tensor:
        """Merge the log-sum-exp of every chunk, retaining no gathered rows.

        Args:
            ctx: Autograd context.
            hidden: Tensor of shape [tokens, hidden], the rank-local supervised
                tokens; contiguous, in a dtype ``kernel`` accepts.
            shard: Tensor of shape [local_vocab, hidden], this rank's LM-head
                shard.
            targets: Int64 tensor of shape [tokens] with global vocabulary ids.
            layout: Chunk plan shared by every rank.
            kernel: Per-chunk projection and CE math.

        Returns:
            FP32 tensor of shape [tokens] with each token's CE loss. Inputs are
            not mutated.
        """
        owner = torch.div(targets, layout.rows_per_rank, rounding_mode="floor")
        offset = targets - owner * layout.rows_per_rank
        lse = torch.full(targets.shape, float("-inf"), dtype=torch.float32, device=hidden.device)
        target_logit = torch.zeros(targets.shape, dtype=torch.float32, device=hidden.device)
        for chunk, weight in layout.stream(shard):
            # Empty ranks still enter every collective, but have nothing to compute.
            if hidden.shape[0]:
                chunk_lse, chunk_target = kernel.forward(hidden, weight, layout.chunk_targets(owner, offset, chunk))
                lse = torch.logaddexp(lse, chunk_lse)
                target_logit += chunk_target
        ctx.save_for_backward(hidden, shard, owner, offset, lse)
        ctx.layout = layout
        ctx.kernel = kernel
        return lse - target_logit

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, grad_loss: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        """Re-gather each chunk and reduce-scatter its weight gradient into the shard.

        Args:
            ctx: Context holding the forward inputs and the full-vocabulary
                log-sum-exp of shape [tokens].
            grad_loss: Tensor of shape [tokens], the upstream gradient of the
                per-token losses.

        Returns:
            The gradients of ``hidden`` ([tokens, hidden], hidden dtype) and of
            ``shard`` ([local_vocab, hidden], shard dtype, averaged over the
            mesh), followed by three None entries.
        """
        hidden, shard, owner, offset, lse = ctx.saved_tensors
        layout: _VocabShardLayout = ctx.layout
        kernel: VocabChunkKernel = ctx.kernel
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        grad_loss = grad_loss.float().contiguous()
        grad_hidden = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device) if need_hidden else None
        grad_shard = torch.empty_like(shard) if need_weight else None
        pending = None
        for chunk, weight in layout.stream(shard):
            chunk_grad_hidden, chunk_grad_weight = None, None
            if hidden.shape[0]:
                chunk_grad_hidden, chunk_grad_weight = kernel.backward(
                    hidden,
                    weight,
                    layout.chunk_targets(owner, offset, chunk),
                    lse,
                    grad_loss,
                    need_hidden=need_hidden,
                    need_weight=need_weight,
                )
            if chunk_grad_hidden is not None:
                grad_hidden += chunk_grad_hidden
            if need_weight:
                if chunk_grad_weight is None:
                    chunk_grad_weight = torch.zeros_like(weight)
                if pending is not None:
                    pending.finish(grad_shard, layout.replicate_groups)
                pending = layout.start_reduce(chunk_grad_weight, chunk, grad_shard)
            del weight, chunk_grad_hidden, chunk_grad_weight
        if pending is not None:
            pending.finish(grad_shard, layout.replicate_groups)
        return grad_hidden.to(hidden.dtype) if need_hidden else None, grad_shard, None, None, None


def sharded_linear_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_weight: ShardedLMHeadWeight,
    *,
    kernel: VocabChunkKernel,
    ignore_index: int,
) -> torch.Tensor:
    """Compute per-token CE while streaming a sharded LM head in vocabulary chunks.

    At most two gathered chunks and their gradients are live at once, so neither
    the full weight nor its full-size gradient is allocated. Every rank in the
    weight's mesh must call this with the same layout, including ranks with no
    supervised tokens. Backward averages the weight gradient over the mesh into
    the local shard, matching the recipes' FSDP gradient convention.

    Args:
        hidden_states: Tensor of shape [tokens, hidden] with rank-local hidden
            states, in a dtype ``kernel`` accepts (the cut_cross_entropy kernels
            need ``lm_weight.dtype``). Token order matches ``labels``.
        labels: Integer tensor of shape [tokens] with vocabulary ids in
            ``[0, vocab)``, or ``ignore_index`` for unsupervised tokens.
        lm_weight: Sharded LM head of global shape [vocab, hidden].
        kernel: Per-chunk projection and CE math.
        ignore_index: Label value that marks unsupervised tokens.

    Returns:
        FP32 tensor of shape [tokens] with each token's CE loss, zero where the
        label is ``ignore_index``. Differentiable with respect to
        ``hidden_states`` and ``lm_weight.weight``.
    """
    layout = _VocabShardLayout(lm_weight)
    valid = labels != ignore_index
    rows = valid.nonzero().squeeze(1)
    if rows.numel() == labels.numel():
        hidden, targets = hidden_states.contiguous(), labels.long()
    else:
        hidden, targets = hidden_states.index_select(0, rows).contiguous(), labels.index_select(0, rows).long()
    losses = _StreamedLinearCrossEntropy.apply(hidden, lm_weight.weight.to_local(), targets, layout, kernel)
    if rows.numel() == labels.numel():
        return losses
    return losses.new_zeros(labels.shape).index_copy(0, rows, losses)


def _chunk_lse_and_target(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None,
    logits_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one token block onto one vocabulary chunk and reduce the logits.

    Args:
        hidden: Tensor of shape [tokens, hidden]; cast to ``weight.dtype``.
        weight: Tensor of shape [chunk_vocab, hidden].
        targets: Int64 tensor of shape [tokens] with the target row in
            ``weight``, or -1 for targets in another chunk.
        softcap: Optional logit softcapping value.
        logits_dtype: Optional dtype that rounds the projection output, matching
            the owning model's logits.

    Returns:
        Two FP32 tensors of shape [tokens]: the chunk log-sum-exp and the target
        logit, zero where ``targets`` is -1.
    """
    logits = F.linear(hidden.to(weight.dtype), weight)
    if logits_dtype in (torch.float16, torch.bfloat16) and logits.dtype != logits_dtype:
        logits = logits.to(logits_dtype)
        # Materialize the model-owned rounding boundary: Inductor otherwise
        # elides the lowp -> FP32 round-trip, changing loss and grads.
        torch._dynamo.graph_break()
    elif logits_dtype is not None:
        logits = logits.to(logits_dtype)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float()
    target_logit = logits.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return torch.logsumexp(logits, dim=-1), torch.where(targets >= 0, target_logit, 0.0)


_compiled_chunk_lse_and_target = torch.compile(_chunk_lse_and_target, dynamic=True)


class TorchVocabChunkKernel:
    """PyTorch :class:`VocabChunkKernel` that recomputes token blocks of logits in backward."""

    def __init__(
        self,
        *,
        token_block: int | None,
        softcap: float | None,
        logits_dtype: torch.dtype | None,
        compile: bool,
    ) -> None:
        """Configure the per-chunk math.

        Args:
            token_block: Maximum tokens projected at once; None projects all of
                them, so memory grows as tokens x chunk_vocab.
            softcap: Optional logit softcapping value.
            logits_dtype: Optional dtype that rounds the projection output.
            compile: Run the projection and reduction through ``torch.compile``.
        """
        self.token_block = token_block
        self.softcap = softcap
        self.logits_dtype = logits_dtype
        self._compute: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
            _compiled_chunk_lse_and_target if compile else _chunk_lse_and_target
        )

    def _block_len(self, tokens: int) -> int:
        return self.token_block or max(tokens, 1)

    def forward(
        self, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """See :meth:`VocabChunkKernel.forward`."""
        lse = torch.empty(targets.shape, dtype=torch.float32, device=hidden.device)
        target_logit = torch.empty_like(lse)
        block = self._block_len(hidden.shape[0])
        for start in range(0, hidden.shape[0], block):
            end = start + block
            lse[start:end], target_logit[start:end] = self._compute(
                hidden[start:end], weight, targets[start:end], self.softcap, self.logits_dtype
            )
        return lse, target_logit

    def backward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        lse: torch.Tensor,
        grad: torch.Tensor,
        *,
        need_hidden: bool,
        need_weight: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """See :meth:`VocabChunkKernel.backward`.

        Differentiates ``grad * (exp(chunk_lse - lse) * chunk_lse - target_logit)``
        per token block. Its logit gradient equals the full-vocabulary CE gradient
        restricted to the chunk, and autograd carries it through the same casts
        and softcapping as the forward.
        """
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if need_weight else None
        with torch.enable_grad():
            local_weight = weight.detach().requires_grad_(need_weight)
            block = self._block_len(hidden.shape[0])
            for start in range(0, hidden.shape[0], block):
                end = start + block
                local_hidden = hidden[start:end].detach().requires_grad_(need_hidden)
                chunk_lse, chunk_target = self._compute(
                    local_hidden, local_weight, targets[start:end], self.softcap, self.logits_dtype
                )
                scale = torch.exp(chunk_lse.detach() - lse[start:end])
                surrogate = (grad[start:end] * (scale * chunk_lse - chunk_target)).sum()
                inputs = [x for x in (local_hidden, local_weight) if x.requires_grad]
                grads = torch.autograd.grad(surrogate, inputs)
                if need_hidden:
                    grad_hidden[start:end] = grads[0]
                if need_weight:
                    grad_weight += grads[-1]
        return grad_hidden, grad_weight.to(weight.dtype) if need_weight else None


class LinearCrossEntropy(nn.Module):
    """Losses consuming final hidden states and an LM-head weight instead of logits."""

    @staticmethod
    def resolve_lm_weight(
        lm_weight: torch.Tensor | ShardedLMHeadWeight,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor | ShardedLMHeadWeight:
        """Resolve the LM-head operand without communication or a full gather.

        Args:
            lm_weight: LM-head weight of global shape [vocab, hidden]: a regular
                tensor, a DTensor sharded by FSDP or HSDP on the vocabulary axis,
                or an already resolved :class:`ShardedLMHeadWeight`.
            grad_reduce_group: Group of ranks contributing independent token
                losses. Its size must match the DTensor mesh when a weight
                gradient is produced.

        Returns:
            A regular tensor or a :class:`ShardedLMHeadWeight` is returned
            unchanged. A DTensor on a one-rank mesh resolves to its local tensor
            of shape [vocab, hidden], which keeps the parameter's gradient path.
            Any other DTensor resolves to a :class:`ShardedLMHeadWeight`, which
            the losses stream in vocabulary chunks.

        Raises:
            ValueError: If a trainable sharded weight has no matching reduction
                group. This fails closed instead of producing rank-local shards.
        """
        if not isinstance(lm_weight, DTensor):
            return lm_weight

        mesh_world_size = lm_weight.device_mesh.size()
        if torch.is_grad_enabled() and lm_weight.requires_grad:
            reduce_world_size = dist.get_world_size(grad_reduce_group) if grad_reduce_group is not None else 1
            if mesh_world_size != reduce_world_size:
                raise ValueError(
                    "LinearCrossEntropy requires grad_reduce_group to match the LM-head "
                    f"DTensor mesh: mesh size={mesh_world_size}, reduction group size={reduce_world_size}. "
                    "Tensor-parallel or hierarchical layouts need an explicit compatible loss path."
                )
        if mesh_world_size == 1:
            return lm_weight.to_local()
        return ShardedLMHeadWeight(weight=lm_weight, dtype=lm_weight.dtype)
