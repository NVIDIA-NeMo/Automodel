# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
import threading
from contextlib import contextmanager

from nemo_automodel.shared.import_utils import safe_import

_DEEP_EP_INSTALLED, deep_ep = safe_import("deep_ep")
HAVE_DEEP_EP = _DEEP_EP_INSTALLED and hasattr(deep_ep, "ElasticBuffer")
if HAVE_DEEP_EP:
    ElasticBuffer = deep_ep.ElasticBuffer


# ── DeepEP dispatch replay across activation-checkpoint recompute ────────────
#
# Activation checkpointing replays a block's forward during backward, which
# re-runs its MoE dispatch from scratch: DeepEP recomputes the routing layout
# (`get_dispatch_layout` -> `notify_dispatch`) and then moves the tokens. The
# layout is pure a function of the routing, and the routing is identical on the
# replay (checkpointing restores the same inputs and the AC policy pins the
# router's top-k), so that recomputation is redundant.
#
# DeepEP already exposes the cheap path: passing the `handle` returned by a
# previous dispatch skips the layout exchange entirely -- that is what
# `FusedDispatch.backward` does, and it is why the backward dispatch costs
# ~4 ms against ~4.4 s for the layout-computing forward dispatches.
#
# Recording is scoped to one checkpoint frame: the AC wrapper builds a recorder
# per checkpointed call, the forward appends each dispatch's handle and routing
# metadata in call order, and the recompute consumes them in the same order.
# Only the handle and the (small) per-token routing metadata are retained --
# the dispatched activations themselves are still re-communicated, so the
# memory that activation checkpointing saves is preserved.
class DispatchReplayRecorder:
    """Per-checkpoint-frame log of DeepEP dispatch results, replayed on recompute."""

    def __init__(self) -> None:
        self._records: list = []
        self._cursor = 0
        self.replay_misses = 0

    def record(self, entry) -> None:
        self._records.append(entry)

    def take(self):
        """Next recorded dispatch, or None when the replay outruns the log."""
        if self._cursor >= len(self._records):
            # Recompute issued more dispatches than the forward did. Fall back to
            # a full dispatch rather than replaying a mismatched layout.
            self.replay_misses += 1
            return None
        entry = self._records[self._cursor]
        self._cursor += 1
        return entry

    def rewind(self) -> None:
        self._cursor = 0


class _DispatchReplayState(threading.local):
    """Thread-local replay binding. ``threading.local`` runs ``__init__`` once per
    thread, so both attributes always exist on every thread that touches it."""

    def __init__(self) -> None:
        self.recorder: DispatchReplayRecorder | None = None
        self.mode: str | None = None


_dispatch_replay_state = _DispatchReplayState()


def _replay_mode() -> str | None:
    return _dispatch_replay_state.mode


def _active_recorder() -> "DispatchReplayRecorder | None":
    return _dispatch_replay_state.recorder


@contextmanager
def dispatch_replay_scope(recorder: "DispatchReplayRecorder | None", mode: str):
    """Bind ``recorder`` for the enclosed region. ``mode`` is 'record' or 'replay'."""
    prev_recorder = _dispatch_replay_state.recorder
    prev_mode = _dispatch_replay_state.mode
    _dispatch_replay_state.recorder = recorder
    _dispatch_replay_state.mode = mode if recorder is not None else None
    try:
        yield
    finally:
        _dispatch_replay_state.recorder = prev_recorder
        _dispatch_replay_state.mode = prev_mode


try:
    import importlib.util

    if importlib.util.find_spec("uccl") is None and importlib.util.find_spec("ep") is None:
        raise ImportError("Neither uccl nor ep package is installed")
    from nemo_automodel.components.moe.uccl_ep import UCCLBuffer
    from nemo_automodel.components.moe.uccl_ep.buffer import EventHandle as UCCLEventHandle
    from nemo_automodel.components.moe.uccl_ep.buffer import EventOverlap as UCCLEventOverlap

    HAVE_UCCL_EP = True
    # Default from env; overridden by MoEFlexTokenDispatcher.set_uccl_num_sms() at init time
    UCCLBuffer.set_num_sms(int(os.environ.get("UCCL_EP_SM_NUMS", os.environ.get("DEEP_EP_SM_NUMS", 20))))
except ImportError:
    HAVE_UCCL_EP = False

import torch

_buffer = None
_buffer_spec = None
_deepep_num_sms = 0
_uccl_buffer = None


class HybridEPDispatchReplayRecorder:
    """Record HybridEP layouts from checkpoint forward for deterministic replay."""

    def __init__(self) -> None:
        self._records: list = []
        self._cursor = 0
        self.replay_misses = 0

    def record(self, handle, tokens_per_expert) -> None:
        self._records.append([handle, tokens_per_expert, None])

    def finalize(self) -> None:
        """Cache each layout extent after the checkpoint-forward op context exits."""
        for entry in self._records:
            if entry[2] is None:
                # HybridEP's sync-free replay API expects a host integer.  Do
                # both the reduction and device-to-host scalar conversion only
                # after the selective-checkpoint context exits; otherwise the
                # replay-only conversion adds aten._local_scalar_dense to the
                # recompute trace.
                entry[2] = int(entry[1].sum().item())

    def take(self):
        """Return the next forward dispatch record, or ``None`` on divergence."""
        if self._cursor >= len(self._records):
            self.replay_misses += 1
            return None
        entry = self._records[self._cursor]
        self._cursor += 1
        return entry

    def rewind(self) -> None:
        self._cursor = 0


class _HybridEPDispatchReplayState(threading.local):
    def __init__(self) -> None:
        self.recorder: HybridEPDispatchReplayRecorder | None = None
        self.mode: str | None = None


_hybridep_dispatch_replay_state = _HybridEPDispatchReplayState()


@contextmanager
def hybridep_dispatch_replay_scope(recorder: HybridEPDispatchReplayRecorder | None, mode: str):
    """Bind a HybridEP dispatch recorder for checkpoint forward or recompute."""
    if mode not in ("record", "replay"):
        raise ValueError(f"Unsupported HybridEP dispatch replay mode: {mode}")
    previous_recorder = _hybridep_dispatch_replay_state.recorder
    previous_mode = _hybridep_dispatch_replay_state.mode
    _hybridep_dispatch_replay_state.recorder = recorder
    _hybridep_dispatch_replay_state.mode = mode if recorder is not None else None
    try:
        yield
    finally:
        _hybridep_dispatch_replay_state.recorder = previous_recorder
        _hybridep_dispatch_replay_state.mode = previous_mode


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(
    group: torch.distributed.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
):
    """Get or grow the process-global DeepEP V2 elastic buffer.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        num_max_tokens_per_rank (int): Maximum input tokens on each EP rank.
        hidden (int): Hidden dimension of each token.
        num_topk (int): Number of selected experts per token.

    Returns:
        ElasticBuffer: Communication buffer.
    """
    global _buffer, _buffer_spec
    configured_capacity = int(os.environ.get("DEEPEP_MAX_TOKENS_PER_RANK", "0"))
    num_max_tokens_per_rank = max(num_max_tokens_per_rank, configured_capacity)
    allow_hybrid_mode = os.environ.get("DEEPEP_ALLOW_HYBRID_MODE", "0") == "1"
    requested_spec = (group, hidden, num_topk, allow_hybrid_mode)

    if _buffer is None or _buffer_spec != requested_spec or _buffer.num_max_tokens_per_rank < num_max_tokens_per_rank:
        free_buffer()
        # ProcessGroupNCCL creates its communicator lazily unless distributed
        # initialization received ``device_id``. ElasticBuffer consumes the raw
        # communicator pointer immediately, so initialize it once before handing
        # that pointer to DeepEP.
        torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
        _buffer = ElasticBuffer(
            group=group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            allow_hybrid_mode=allow_hybrid_mode,
            explicitly_destroy=True,
        )
        _buffer_spec = requested_spec
    return _buffer


def free_buffer() -> None:
    """Destroy the global DeepEP ``ElasticBuffer`` communication runtime.

    ElasticBuffer owns GPU communication memory and internal NCCL communicators. Destroy it
    before replacing a too-small buffer or tearing down the caller's process group.
    """
    global _buffer, _buffer_spec
    if _buffer is not None:
        try:
            _buffer.destroy()
        except Exception:  # pragma: no cover - best effort
            pass
        _buffer = None
        _buffer_spec = None


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        sync_free=False,
    ):
        """Forward pass of fused dispatch through DeepEP V2's ElasticBuffer."""
        buffer = get_buffer(group, x.shape[0], x.shape[1], token_indices.shape[1])
        allocate_on_comm_stream = allocate_on_comm_stream or async_finish
        previous_event = buffer.capture() if async_finish else None

        # Activation-checkpoint replay: reuse the layout this dispatch computed
        # on the original forward instead of recomputing it. Cached-mode dispatch
        # returns only recv_x, so the routing metadata comes from the record.
        recorder = _active_recorder()
        if recorder is not None and _replay_mode() == "replay":
            replayed = recorder.take()
            if replayed is not None:
                cached_handle, recv_token_indices, recv_token_probs, tokens_per_expert = replayed
                recv_x, _, recv_token_probs, _, after_event_overlap = buffer.dispatch(
                    x,
                    topk_weights=token_probs,
                    handle=cached_handle,
                    previous_event=previous_event,
                    async_with_compute_stream=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
                    do_cpu_sync=False if sync_free else None,
                    do_expand=sync_free,
                    do_zero_padding=sync_free,
                    num_sms=_deepep_num_sms,
                )
                if async_finish:
                    after_event_overlap.current_stream_wait()
                ctx.group = group
                ctx.handle = cached_handle
                ctx.async_finish = async_finish
                ctx.allocate_on_comm_stream = allocate_on_comm_stream
                ctx.num_max_tokens_per_rank = x.shape[0]
                ctx.hidden = x.shape[1]
                ctx.num_topk = token_indices.shape[1]
                return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, cached_handle)

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_experts=num_experts,
            num_max_tokens_per_rank=x.shape[0],
            expert_alignment=1,
            num_sms=_deepep_num_sms,
            previous_event=previous_event,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_cpu_sync=not sync_free,
            do_expand=sync_free,
            do_zero_padding=sync_free,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.num_max_tokens_per_rank = x.shape[0]
        ctx.hidden = x.shape[1]
        ctx.num_topk = token_indices.shape[1]
        if sync_free:
            tokens_per_expert = handle.num_unaligned_recv_tokens_per_expert
        else:
            tokens_per_expert = torch.tensor(handle.num_recv_tokens_per_expert_list)

        if recorder is not None and _replay_mode() == "record":
            # Keep the handle and routing metadata (small) so the recompute can
            # skip the layout exchange; recv_x is deliberately not retained.
            recorder.record((handle, recv_token_indices, recv_token_probs, tokens_per_expert))

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, ctx.num_max_tokens_per_rank, ctx.hidden, ctx.num_topk)
        handle = ctx.handle
        previous_event = buffer.capture() if ctx.async_finish else None
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        buffer = get_buffer(group, handle.num_max_tokens_per_rank, x.shape[1], handle.topk_idx.shape[1])
        allocate_on_comm_stream = allocate_on_comm_stream or async_finish
        previous_event = buffer.capture() if async_finish else None
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_with_compute_stream=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.num_max_tokens_per_rank = handle.num_max_tokens_per_rank
        ctx.hidden = x.shape[1]
        ctx.num_topk = handle.topk_idx.shape[1]
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, ctx.num_max_tokens_per_rank, ctx.hidden, ctx.num_topk)
        previous_event = buffer.capture() if ctx.async_finish else None
        grad_x, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
            num_sms=_deepep_num_sms,
            do_cpu_sync=False,
            do_expand=ctx.handle.do_expand,
            do_zero_padding=ctx.handle.do_expand,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        sync_free=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
            sync_free,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP."""
        global _deepep_num_sms
        _deepep_num_sms = num_sms

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


# HybridEP support
HAVE_HYBRIDEP = _DEEP_EP_INSTALLED and hasattr(deep_ep, "HybridEPBuffer")
if HAVE_HYBRIDEP:
    HybridEPBuffer = deep_ep.HybridEPBuffer

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    """Initialize the HybridEP buffer, including buffer allocation and metadata initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group: Process group for HybridEP all-to-all communication.
        hidden_dim: Hidden dimension of the input tensor.
        seq_len: Maximum sequence length of the input tensor.
        num_local_experts: Number of local experts.
        num_sms_dispatch_api: Number of SMs used by the dispatch API.
        num_sms_combine_api: Number of SMs used by the combine API.
        fp8_dispatch: Whether to use FP8 communication during the dispatch phase.
    """
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
    )


def reset_hybrid_ep_buffer():
    """Reset the HybridEP buffer."""
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    """Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend."""

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """Forward pass of fused dispatch of the HybridEP backend."""
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )

        recorder = _hybridep_dispatch_replay_state.recorder
        if recorder is not None and _hybridep_dispatch_replay_state.mode == "replay":
            replayed = recorder.take()
            if replayed is not None:
                handle, tokens_per_expert, num_permuted_tokens = replayed
                replayed_outputs = _hybrid_ep_buffer.dispatch_with_permute(
                    hidden=x,
                    probs=probs,
                    scaling_factor=None,
                    handle=handle,
                    pad_multiple=pad_multiple,
                    num_permuted_tokens=num_permuted_tokens,
                )
                dispatched_hidden, dispatched_probs, dispatched_scaling_factor, _, _ = replayed_outputs
                ctx.handle = handle
                ctx.pad_multiple = pad_multiple
                return (
                    dispatched_hidden,
                    dispatched_probs,
                    dispatched_scaling_factor,
                    tokens_per_expert,
                    handle,
                )

        non_blocking = num_permuted_tokens is not None
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        if recorder is not None and _hybridep_dispatch_replay_state.mode == "record":
            # Keep only the reusable layout and its output extent. Recomputed
            # activations and probabilities are still redispatched through it.
            recorder.record(handle, tokens_per_expert)
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch of the HybridEP backend."""
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x, probs=grad_probs, handle=handle, pad_multiple=ctx.pad_multiple
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None


class HybridEPCombine(torch.autograd.Function):
    """Fused combine operation for permute + combine a2a + permute using the HybridEP backend."""

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        """Forward pass of fused combine of the HybridEP backend."""
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        """Backward pass of fused combine of the HybridEP backend."""
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None


if HAVE_HYBRIDEP:

    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        """Perform fused dispatch for permute + dispatch a2a + permute using the HybridEP backend."""
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    def hybrid_ep_combine(x, handle, num_permuted_tokens=None, pad_multiple=None):
        """Perform fused combine for unpermute + combine a2a + unpermute using the HybridEP backend."""
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None


def get_uccl_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a UCCL-EP buffer for all-to-all communication."""
    global _uccl_buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        UCCLBuffer.get_dispatch_config(group.size()),
        UCCLBuffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (
        _uccl_buffer is None
        or _uccl_buffer.group != group
        or _uccl_buffer.num_nvl_bytes < num_nvl_bytes
        or _uccl_buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _uccl_buffer = UCCLBuffer(group, num_nvl_bytes, num_rdma_bytes)
    return _uccl_buffer


class UCCLFusedDispatch(torch.autograd.Function):
    """Fused dispatch using UCCL-EP instead of DeepEP."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = None
        if async_finish:
            previous_event = UCCLEventOverlap(UCCLEventHandle())
        buffer = get_uccl_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        recv_x, recv_token_indices, recv_token_probs, num_recv_tokens_per_expert_list, handle, after_event = (
            buffer.dispatch(
                x,
                topk_idx=token_indices,
                topk_weights=token_probs,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=layout_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)
        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        buffer = get_uccl_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = UCCLEventOverlap(UCCLEventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class UCCLFusedCombine(torch.autograd.Function):
    """Fused combine using UCCL-EP instead of DeepEP."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        previous_event = None
        if async_finish:
            previous_event = UCCLEventOverlap(UCCLEventHandle())
        buffer = get_uccl_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, _grad_event=None):
        previous_event = None
        if ctx.async_finish:
            previous_event = UCCLEventOverlap(UCCLEventHandle())
        buffer = get_uccl_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_UCCL_EP:

    def uccl_fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch using UCCL-EP."""
        return UCCLFusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def uccl_fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine using UCCL-EP."""
        return UCCLFusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_uccl_num_sms(num_sms):
        """Sets the number of SMs to use for UCCL-EP."""
        UCCLBuffer.set_num_sms(num_sms)

else:
    uccl_fused_dispatch = None
    uccl_fused_combine = None
    set_uccl_num_sms = None
