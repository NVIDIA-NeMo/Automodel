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

"""DeepEP v2.5 transport with BF16 gradients and differentiable routing weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx

from nemo_automodel.shared.import_utils import safe_import_from

if TYPE_CHECKING:
    from deep_ep import EPBuffer
    from deep_ep.buffers.ep import EPHandle


@dataclass(frozen=True)
class _Fp8Dispatch:
    """DeepEP's block-128 E4M3 codec, constructed only for FP8 communication."""

    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    dequantize: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class _Runtime:
    buffer: EPBuffer | None = None
    group: dist.ProcessGroup | None = None
    hidden_dim: int = 0
    settings: tuple[int, ...] | None = None


_runtime = _Runtime()


def finalize_deepep_v25() -> None:
    """Release the shared transport after all backward passes, before group teardown."""
    if _runtime.buffer is not None:
        _runtime.buffer.destroy()
        _runtime.buffer = None
        _runtime.group = None
        _runtime.hidden_dim = 0
        _runtime.settings = None


@dataclass(kw_only=True)
class DeepEpV25Config:
    """Group-uniform capacity for eager, dropless DeepEP v2.5 dispatch.

    Args:
        max_tokens_per_rank: Maximum input rows per microbatch, including masked
            rows. The transport rounds this up to 64 and pads shorter inputs with
            invalid routes. Receive buffers use actual routing counts.
        use_fp8_dispatch: Send forward activations as block-scaled E4M3. Expert
            computation and all backward communication remain BF16. The input
            gradient uses a straight-through estimate for the quantization.
    """

    max_tokens_per_rank: int | None = None
    use_fp8_dispatch: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens_per_rank is not None and self.max_tokens_per_rank <= 0:
            raise ValueError("deepep_v25.max_tokens_per_rank must be positive")

    @property
    def capacity(self) -> int:
        """Per-rank row capacity rounded up to DeepEP's 64-row granularity."""
        if self.max_tokens_per_rank is None:
            raise ValueError("dispatcher='deepep_v25' requires deepep_v25.max_tokens_per_rank")
        return (self.max_tokens_per_rank + 63) // 64 * 64

    def init_buffer(self, *, group: dist.ProcessGroup, hidden_dim: int, top_k: int) -> EPBuffer:
        """Create the process-wide EPBuffer outside forward and checkpoint replay.

        Like the legacy DeepEP ``get_buffer``, the expert module calls this with its
        token width; the dispatcher only borrows the buffer at dispatch time.

        Args:
            group: NCCL expert-parallel group.
            hidden_dim: Dispatched token width.
            top_k: Distinct expert selections per token.

        Returns:
            The shared communication buffer.
        """
        if group.size() < 2 or hidden_dim <= 0 or top_k <= 0:
            raise ValueError("DeepEP v2.5 requires EP size >= 2, a positive hidden dimension, and top-k")
        available, buffer_type = safe_import_from("deep_ep", "EPBuffer")
        if not available:
            raise ImportError("dispatcher='deepep_v25' requires DeepEP v2.5 with the EPBuffer API")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("DeepEP v2.5 requires Hopper or newer CUDA GPUs (SM90+)")
        if dist.get_backend(group) != "nccl":
            raise ValueError("DeepEP v2.5 requires an NCCL process group")
        settings = (self.capacity, hidden_dim, top_k, torch.cuda.current_device(), self.use_fp8_dispatch)
        if _runtime.buffer is None:
            # Fresh EP subgroups may not own an NCCL communicator yet. DeepEP
            # borrows its pointer immediately when querying the NVLink topology.
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            # Reserve BF16 capacity even with FP8 forward: backward dispatch is BF16.
            _runtime.buffer = buffer_type(
                group,
                num_max_tokens_per_rank=self.capacity,
                hidden=hidden_dim,
                num_topk=top_k,
                explicitly_destroy=True,
            )
            _runtime.group, _runtime.hidden_dim, _runtime.settings = group, hidden_dim, settings
        elif _runtime.group is not group or _runtime.settings != settings:
            raise ValueError("DeepEP v2.5 supports one EP group and one capacity/shape configuration per process")
        return _runtime.buffer

    def build(self, *, group: dist.ProcessGroup, num_experts: int, top_k: int, num_sms: int) -> DeepEpV25Manager:
        """Create layer-local routing state; the transport comes from ``init_buffer``.

        Args:
            group: NCCL expert-parallel group with contiguous expert ownership.
            num_experts: Global expert count, divisible by the group size.
            top_k: Distinct expert selections per token.
            num_sms: Communication SM budget; zero selects DeepEP's automatic budget.

        Returns:
            Layer-local routing state backed by the shared transport.
        """
        if group.size() < 2 or num_experts % group.size():
            raise ValueError("DeepEP v2.5 requires EP size >= 2 dividing the expert count")
        if not 0 < top_k <= num_experts or num_sms < 0:
            raise ValueError("Invalid DeepEP v2.5 top-k or SM budget")
        fp8 = None
        if self.use_fp8_dispatch:
            quantize_available, quantize = safe_import_from("deep_ep.utils.math", "per_token_cast_to_fp8")
            dequantize_available, dequantize = safe_import_from("deep_ep.utils.math", "per_token_cast_back")
            if not quantize_available or not dequantize_available:
                raise ImportError("FP8 dispatch requires DeepEP's block-128 E4M3 conversion helpers")
            fp8 = _Fp8Dispatch(quantize, dequantize)
        return DeepEpV25Manager(
            group=group,
            max_tokens=self.max_tokens_per_rank,
            capacity=self.capacity,
            num_experts=num_experts,
            top_k=top_k,
            num_sms=num_sms,
            fp8=fp8,
        )


@dataclass
class _DispatchHandle:
    """Pair a forward EPHandle with its dispatch-owned saved-tensor context.

    The handle follows DeepEP's EPHandle layout contract. Its tensors are saved
    once on dispatch so checkpoint unpack returns the recomputed receive order.
    Combine shares that context instead of registering a second set of hooks.
    """

    value: EPHandle
    context: FunctionCtx


def _restore_handle(ctx: FunctionCtx) -> EPHandle:
    """Restore routing metadata using dispatch's checkpoint-aware saved tensors.

    Args:
        ctx: Dispatch context. Its saved tensors are original indices
            [capacity, top_k], rank prefix counts [scaleup_ranks], expert prefix
            counts and unaligned counts [local_experts], receive-source metadata
            [received_tokens, 2 + top_k], and destination-slot, forwarding, and
            linked-list metadata with the layouts documented by DeepEP EPHandle.
            The last two tensors are None for a single NVLink domain.

    Returns:
        A fresh EPHandle referencing the saved or recomputed metadata tensors.
        It does not mutate the original handle or any saved tensor.
    """
    if ctx.restored_handle is not None:
        return ctx.restored_handle
    handle = ctx.handle
    indices, rank_counts, expert_counts, unaligned_counts, sources, slots, forwarding, links = ctx.saved_tensors
    ctx.restored_handle = type(handle)(
        do_expand=handle.do_expand,
        num_experts=handle.num_experts,
        expert_alignment=handle.expert_alignment,
        num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
        num_sms=handle.num_sms,
        topk_idx=indices,
        num_recv_tokens=handle.num_recv_tokens,
        num_expanded_tokens=handle.num_expanded_tokens,
        num_recv_tokens_per_expert_list=handle.num_recv_tokens_per_expert_list,
        psum_num_recv_tokens_per_scaleup_rank=rank_counts,
        psum_num_recv_tokens_per_expert=expert_counts,
        num_unaligned_recv_tokens_per_expert=unaligned_counts,
        recv_src_metadata=sources,
        dst_buffer_slot_idx=slots,
        token_metadata_at_forward=forwarding,
        channel_linked_list=links,
    )
    return ctx.restored_handle


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        buffer: EPBuffer,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        num_experts: int,
        num_sms: int,
        fp8: _Fp8Dispatch | None,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _DispatchHandle]:
        """Dispatch tokens and probabilities into expert-major rows.

        Args:
            ctx: Autograd context retaining this call's immutable routing handle.
            buffer: Shared communication workspace; operations are stream ordered.
            hidden: Contiguous BF16 tensor of shape [capacity, hidden].
            indices: Contiguous int64 tensor of shape [capacity, top_k], with -1
                for invalid routes. The handle retains this tensor without mutation.
            weights: Contiguous FP32 tensor of shape [capacity, top_k].
            num_experts: Global expert count.
            num_sms: Communication SM budget.
            fp8: Optional forward-only block-128 E4M3 codec.
            async_finish: Use a deferred epilogue and wait before expert compute.
            allocate_on_comm_stream: Allocate output storage on the communication stream.

        Returns:
            Expert-packed BF16 activations [received_tokens, hidden], FP32 weights
            [received_tokens], integer counts [local_experts], and the routing
            handle described by DeepEP's EPHandle contract. Tensor storage is
            unique to this dispatch and remains owned by its autograd graph.
        """
        payload = hidden if fp8 is None else fp8.quantize(hidden)
        result = buffer.dispatch(
            payload,
            topk_idx=indices,
            topk_weights=weights,
            num_experts=num_experts,
            expert_alignment=1,
            num_sms=num_sms,
            do_expand=True,
            do_cpu_sync=True,
            previous_event=buffer.capture() if allocate_on_comm_stream else None,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            defer_epilogue=async_finish,
        )
        received, _, probs, handle = result.wait() if async_finish else result[:4]
        if fp8 is not None:
            received = fp8.dequantize(*received)
        ctx.async_finish, ctx.allocate_on_comm_stream = async_finish, allocate_on_comm_stream
        ctx.buffer, ctx.handle, ctx.num_sms = buffer, handle, num_sms
        ctx.restored_handle = None
        # Receive order can change during checkpoint replay even with identical
        # routes. Unpack this metadata together with recomputed expert activations.
        # The paired combine shares these hooks, avoiding replay through combine.
        ctx.save_for_backward(
            handle.topk_idx,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.psum_num_recv_tokens_per_expert,
            handle.num_unaligned_recv_tokens_per_expert,
            handle.recv_src_metadata,
            handle.dst_buffer_slot_idx,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
        )
        counts = handle.num_unaligned_recv_tokens_per_expert
        ctx.mark_non_differentiable(counts)
        return received, probs, counts, _DispatchHandle(handle, ctx)

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_hidden: torch.Tensor,
        grad_probs: torch.Tensor,
        grad_counts: torch.Tensor,
        grad_handle: None,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor, None, None, None, None, None]:
        """Reverse dispatch for activation and routing-probability gradients.

        Args:
            ctx: Context retaining the forward routing metadata and buffer.
            grad_hidden: BF16 tensor of shape [received_tokens, hidden].
            grad_probs: FP32 tensor of shape [received_tokens].
            grad_counts: Unused integer count gradient of shape [local_experts].
            grad_handle: None; routing indices are nondifferentiable.

        Returns:
            Gradients for forward arguments: None for buffer, BF16 activation
            gradients [capacity, hidden], None for indices, FP32 routing-weight
            gradients [capacity, top_k], and None for the five runtime settings.
        """
        result = ctx.buffer.combine(
            grad_hidden.contiguous(),
            _restore_handle(ctx),
            topk_weights=grad_probs.contiguous(),
            num_sms=ctx.num_sms,
            previous_event=ctx.buffer.capture() if ctx.allocate_on_comm_stream else None,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            defer_epilogue=ctx.async_finish,
        )
        hidden, weights = result.wait() if ctx.async_finish else result[:2]
        return None, hidden, None, weights, None, None, None, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        buffer: EPBuffer,
        hidden: torch.Tensor,
        handle: _DispatchHandle,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ) -> torch.Tensor:
        """Sum weighted expert outputs into original token order.

        Args:
            ctx: Autograd context owning the handle through backward.
            buffer: Shared communication workspace.
            hidden: Contiguous BF16 tensor of shape [received_tokens, hidden],
                already multiplied by routing weights in the expert computation.
            handle: Immutable routing metadata from the paired dispatch, following
                DeepEP's EPHandle tensor-layout contract.
            async_finish: Use a deferred epilogue and wait before consuming outputs.
            allocate_on_comm_stream: Allocate output storage on the communication stream.

        Returns:
            BF16 tensor of shape [capacity, hidden] in source token order.
        """
        ctx.buffer, ctx.dispatch_context = buffer, handle.context
        ctx.async_finish, ctx.allocate_on_comm_stream = async_finish, allocate_on_comm_stream
        result = buffer.combine(
            hidden,
            handle.value,
            num_sms=handle.value.num_sms,
            previous_event=buffer.capture() if allocate_on_comm_stream else None,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            defer_epilogue=async_finish,
        )
        combined, _ = result.wait() if async_finish else result[:2]
        return combined

    @staticmethod
    def backward(ctx: FunctionCtx, grad_hidden: torch.Tensor) -> tuple[None, torch.Tensor, None, None, None]:
        """Route output gradients using the forward's cached expanded layout.

        Args:
            ctx: Context retaining the paired forward routing handle.
            grad_hidden: BF16 tensor of shape [capacity, hidden].

        Returns:
            None for buffer, BF16 gradients [received_tokens, hidden], and None
            for the routing handle and stream options. Probabilities are differentiated in the experts.
        """
        handle = _restore_handle(ctx.dispatch_context)
        result = ctx.buffer.dispatch(
            grad_hidden.contiguous(),
            handle=handle,
            num_experts=handle.num_experts,
            num_sms=handle.num_sms,
            do_expand=True,
            previous_event=ctx.buffer.capture() if ctx.allocate_on_comm_stream else None,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            defer_epilogue=ctx.async_finish,
        )
        received = (result.wait() if ctx.async_finish else result)[0]
        return None, received, None, None, None


class DeepEpV25Manager:
    """Layer-local flex-dispatcher state; each autograd call owns its routing handle."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        max_tokens: int,
        capacity: int,
        num_experts: int,
        top_k: int,
        num_sms: int,
        fp8: _Fp8Dispatch | None,
    ) -> None:
        """Store fixed routing dimensions and initially empty routing state."""
        self.group = group
        self.fp8 = fp8
        self.max_tokens, self.capacity = max_tokens, capacity
        self.num_experts = num_experts
        self.top_k, self.num_sms = top_k, num_sms
        self.token_indices: torch.Tensor | None = None
        self.token_probs: torch.Tensor | None = None
        self.dispatched_probs: torch.Tensor | None = None
        self.tokens_per_expert: torch.Tensor | None = None
        self._handle: _DispatchHandle | None = None
        self._num_tokens = 0

    @property
    def buffer(self) -> EPBuffer:
        """The process-wide EPBuffer created by ``DeepEpV25Config.init_buffer``."""
        if _runtime.buffer is None:
            raise RuntimeError("DeepEP v2.5 buffer is not initialized; the expert module creates it")
        if _runtime.group is not self.group:
            raise ValueError("DeepEP v2.5 buffer belongs to a different EP group")
        return _runtime.buffer

    def setup_metadata(self, num_local_tokens: int, probs: torch.Tensor) -> None:
        """Select routes from dense weights.

        Args:
            num_local_tokens: Local input row count.
            probs: Tensor of shape [tokens, ep_size, local_experts], enumerating
                contiguous global expert IDs, with zero for inactive routes.
        """
        self.token_probs, indices = probs.reshape(num_local_tokens, self.num_experts).topk(self.top_k, dim=-1)
        self.token_indices = indices.masked_fill(self.token_probs == 0, -1)

    def dispatch(
        self, hidden_states: torch.Tensor, async_finish: bool = False, allocate_on_comm_stream: bool = False
    ) -> torch.Tensor:
        """Dispatch into a fresh expert-packed allocation.

        Args:
            hidden_states: BF16 CUDA tensor of shape [local_tokens, hidden].
            async_finish: Use deferred communication epilogues with stream waits.
            allocate_on_comm_stream: Allocate communication outputs on the comm stream;
                requires async_finish.

        Returns:
            BF16 tensor of shape [received_tokens, hidden], packed by local expert.
        """
        if allocate_on_comm_stream and not async_finish:
            raise ValueError("allocate_on_comm_stream requires async_finish")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The eager DeepEP v2.5 dispatcher does not support CUDA Graph capture")
        if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
            raise ValueError("DeepEP v2.5 dispatch requires BF16 [local_tokens, hidden] inputs")
        buffer = self.buffer
        num_tokens, hidden = hidden_states.shape
        if hidden != _runtime.hidden_dim or num_tokens > self.max_tokens:
            raise ValueError(
                "DeepEP v2.5 input exceeds deepep_v25.max_tokens_per_rank or has incompatible hidden width"
            )
        if self.token_indices is None or self.token_probs is None:
            raise RuntimeError("DeepEP v2.5 routing metadata must be set before dispatch")
        if self.token_indices.shape != (num_tokens, self.top_k) or self.token_probs.shape != (num_tokens, self.top_k):
            raise ValueError("DeepEP v2.5 indices and probabilities must have shape [local_tokens, top_k]")
        padding = self.capacity - num_tokens
        tokens = F.pad(hidden_states, (0, 0, 0, padding)).contiguous()
        indices = F.pad(self.token_indices.to(torch.int64), (0, 0, 0, padding), value=-1).contiguous()
        weights = F.pad(self.token_probs.float(), (0, 0, 0, padding)).contiguous()
        received, self.dispatched_probs, self.tokens_per_expert, self._handle = _Dispatch.apply(
            buffer,
            tokens,
            indices,
            weights,
            self.num_experts,
            self.num_sms,
            self.fp8,
            async_finish,
            allocate_on_comm_stream,
        )
        self._num_tokens = num_tokens
        return received

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the native expanded layout.

        Args:
            hidden_states: Expert-packed tensor of shape [received_tokens, hidden].

        Returns:
            The input tensor by reference and FP32 routing weights [received_tokens].
        """
        if self.dispatched_probs is None:
            raise RuntimeError("DeepEP v2.5 dispatch has not completed")
        return hidden_states, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Return integer device counts of shape [local_experts]."""
        if self.tokens_per_expert is None:
            raise RuntimeError("DeepEP v2.5 dispatch has not completed")
        return self.tokens_per_expert

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep the native expert-packed layout for combine.

        Args:
            hidden_states: Tensor of shape [received_tokens, hidden], already weighted.

        Returns:
            The input tensor [received_tokens, hidden] by reference.
        """
        return hidden_states

    def combine(
        self, hidden_states: torch.Tensor, async_finish: bool = False, allocate_on_comm_stream: bool = False
    ) -> torch.Tensor:
        """Combine expert outputs and discard invalid source padding.

        Args:
            hidden_states: BF16 tensor of shape [received_tokens, hidden], already weighted.
            async_finish: Use deferred communication epilogues with stream waits.
            allocate_on_comm_stream: Allocate communication outputs on the comm stream;
                requires async_finish.

        Returns:
            BF16 tensor of shape [local_tokens, hidden] in source token order.
        """
        if allocate_on_comm_stream and not async_finish:
            raise ValueError("allocate_on_comm_stream requires async_finish")
        if self._handle is None:
            raise RuntimeError("DeepEP v2.5 combine requires a matching dispatch")
        result = _Combine.apply(
            self.buffer, hidden_states.contiguous(), self._handle, async_finish, allocate_on_comm_stream
        )
        self._handle = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.token_indices = None
        self.token_probs = None
        return result[: self._num_tokens]
