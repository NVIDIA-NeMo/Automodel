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

import atexit
import importlib
import os
import shutil
from pathlib import Path

import torch

from nemo_automodel.shared.import_utils import safe_import_from


def _safe_import_first_symbol(module_names: tuple[str, ...], symbol: str):
    """Import the first available symbol from several optional module layouts."""
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return True, getattr(module, symbol)
        except (AttributeError, ImportError):
            continue
    return safe_import_from(module_names[-1], symbol)


HAVE_DEEP_EP, Buffer = _safe_import_first_symbol(
    (
        "deep_ep.legacy",
        "deep_ep.buffers.legacy",
        "deep_ep",
    ),
    "Buffer",
)
HAVE_DEEP_EP_V2, ElasticBuffer = _safe_import_first_symbol(
    (
        "deep_ep",
        "deep_ep.buffers.elastic",
    ),
    "ElasticBuffer",
)
_, EventHandle = _safe_import_first_symbol(
    (
        "deep_ep",
        "deep_ep.utils",
        "deep_ep.utils.event",
    ),
    "EventHandle",
)
_, EventOverlap = _safe_import_first_symbol(
    (
        "deep_ep",
        "deep_ep.utils",
        "deep_ep.utils.event",
    ),
    "EventOverlap",
)

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

_buffer = None
_deepep_v2_buffer = None
_deepep_v2_num_sms = 0
_deepep_v2_num_qps = 0
_nvshmem_available = None
_uccl_buffer = None

_DEEPEP_V2_HANDLE_TENSOR_FIELDS = (
    "topk_idx",
    "psum_num_recv_tokens_per_scaleup_rank",
    "psum_num_recv_tokens_per_expert",
    "recv_src_metadata",
    "dst_buffer_slot_idx",
    "token_metadata_at_forward",
    "channel_linked_list",
)


def _is_nvshmem_available() -> bool:
    """Check if DeepEP was compiled with NVSHMEM support.

    Uses is_sm90_compiled() as proxy — DeepEP's build enforces that
    NVSHMEM is disabled when SM90 features are disabled.
    """
    global _nvshmem_available
    if _nvshmem_available is None:
        _nvshmem_available = Buffer.is_sm90_compiled()
    return _nvshmem_available


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    nvshmem = _is_nvshmem_available()
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        if nvshmem:
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if not nvshmem and group.size() > 8:
        raise RuntimeError(
            f"DeepEP was compiled without NVSHMEM support (SM90 features disabled), "
            f"but expert parallelism group size {group.size()} > 8 requires internode "
            f"RDMA communication. Recompile DeepEP with NVSHMEM or reduce ep_size to "
            f"fit within a single node (max 8 GPUs)."
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        # explicitly_destroy=True lets callers free the NVSHMEM/cpp runtime via
        # ``_buffer.destroy()`` (see free_buffer()). Without an explicit teardown the DeepEP
        # state lingers on the GPUs for the lifetime of the process / Slurm allocation and
        # corrupts later forwards (e.g. a checkpoint-robustness HF reload after training).
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, explicitly_destroy=True)
    return _buffer


def free_buffer() -> None:
    """Destroy the global DeepEP ``Buffer`` and release its NVSHMEM/cpp runtime.

    DeepEP keeps a process-global communication buffer backed by NVSHMEM symmetric memory.
    It is normally never torn down (``destroy_process_group`` hangs on DeepEP's NCCL
    sub-groups, so cleanup is skipped), but that leftover GPU state survives process exit for
    the whole Slurm allocation and corrupts subsequent forwards. Destroying the buffer first
    frees the runtime and lets a clean ``destroy_process_group`` follow without hanging.
    """
    global _buffer
    if _buffer is not None:
        try:
            _buffer.destroy()
        except Exception:  # pragma: no cover - best effort
            pass
        _buffer = None


def destroy_deepep_v2_buffer() -> None:
    """Explicitly destroy the DeepEP V2 ElasticBuffer if one is allocated."""
    global _deepep_v2_buffer

    if _deepep_v2_buffer is not None:
        try:
            _deepep_v2_buffer.destroy()
        except Exception:  # pragma: no cover - best effort
            pass
    _deepep_v2_buffer = None


def _warmup_deepep_v2_group(group: torch.distributed.ProcessGroup) -> None:
    """Materialize the EP group's NCCL communicator before ElasticBuffer reads it."""
    warmup = torch.empty(1, device=torch.cuda.current_device())
    work = torch.distributed.all_reduce(warmup, group=group, async_op=True)
    work.wait()


def init_deepep_v2_buffer(
    group: torch.distributed.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
) -> None:
    """Initialize the process-global DeepEP V2 ElasticBuffer."""
    global _deepep_v2_buffer

    if hidden % 256 != 0:
        raise ValueError(f"DeepEP V2 requires a hidden dimension divisible by 256, got {hidden}.")
    if _deepep_v2_buffer is not None:
        return

    _warmup_deepep_v2_group(group)
    _deepep_v2_buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        num_gpu_timeout_secs=300,
        explicitly_destroy=True,
    )


def _save_deepep_v2_handle(ctx, handle) -> None:
    """Save ElasticBuffer handle tensors for non-reentrant checkpoint replay."""
    ctx.handle = handle
    ctx.handle_tensor_fields = tuple(
        field for field in _DEEPEP_V2_HANDLE_TENSOR_FIELDS if getattr(handle, field) is not None
    )
    ctx.save_for_backward(*(getattr(handle, field) for field in ctx.handle_tensor_fields))


def _restore_deepep_v2_handle(ctx):
    """Restore checkpoint-recomputed tensors onto the original ElasticBuffer handle."""
    for field, tensor in zip(ctx.handle_tensor_fields, ctx.saved_tensors, strict=True):
        setattr(ctx.handle, field, tensor)
    return ctx.handle


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
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

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
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
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
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


class DeepEPV2FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation using DeepEP V2 ElasticBuffer."""

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
        """Forward pass of DeepEP V2 fused dispatch."""
        num_topk = token_indices.shape[1]
        num_max_tokens_per_rank = x.size(0)
        if _deepep_v2_buffer is None:
            init_deepep_v2_buffer(
                group,
                num_max_tokens_per_rank,
                x.size(1),
                num_topk,
            )
        recv_x, recv_token_indices, recv_token_probs, handle, after_event = _deepep_v2_buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_sms=_deepep_v2_num_sms,
            num_qps=_deepep_v2_num_qps,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        _save_deepep_v2_handle(ctx, handle)
        tokens_per_expert = torch.as_tensor(handle.num_recv_tokens_per_expert_list, dtype=torch.int64)

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
        """Backward pass of DeepEP V2 fused dispatch."""
        handle = _restore_deepep_v2_handle(ctx)
        grad_x, grad_token_probs, after_event = _deepep_v2_buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float() if grad_token_probs is not None else None,
            num_sms=_deepep_v2_num_sms,
            num_qps=_deepep_v2_num_qps,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return (
            grad_x,
            None,
            grad_token_probs,
            None,
            None,
            None,
            None,
        )


class DeepEPV2FusedCombine(torch.autograd.Function):
    """Fused combine operation using DeepEP V2 ElasticBuffer."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of DeepEP V2 fused combine."""
        del group
        combined_x, _, after_event = _deepep_v2_buffer.combine(
            x.contiguous(),
            handle=handle,
            num_sms=_deepep_v2_num_sms,
            num_qps=_deepep_v2_num_qps,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        _save_deepep_v2_handle(ctx, handle)
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, _grad_event=None):
        """Backward pass of DeepEP V2 fused combine."""
        handle = _restore_deepep_v2_handle(ctx)
        grad_x, _, _, _, after_event = _deepep_v2_buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            num_sms=handle.num_sms,
            num_qps=_deepep_v2_num_qps,
            async_with_compute_stream=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
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

else:
    fused_dispatch = None
    fused_combine = None


def set_deepep_num_sms(num_sms):
    """Set the number of SMs used by the legacy DeepEP Buffer."""
    if HAVE_DEEP_EP:
        Buffer.set_num_sms(num_sms)


def set_deepep_v2_num_sms(num_sms):
    """Set the number of SMs passed to DeepEP V2 ElasticBuffer operations."""
    global _deepep_v2_num_sms
    _deepep_v2_num_sms = num_sms


def set_deepep_v2_num_qps(num_qps):
    """Set the number of QPs passed to DeepEP V2 ElasticBuffer operations."""
    global _deepep_v2_num_qps
    _deepep_v2_num_qps = num_qps


atexit.register(destroy_deepep_v2_buffer)


if HAVE_DEEP_EP_V2:

    def deepep_v2_fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch with DeepEP V2 ElasticBuffer."""
        return DeepEPV2FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def deepep_v2_fused_combine(
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused combine with DeepEP V2 ElasticBuffer."""
        return DeepEPV2FusedCombine.apply(
            x,
            group,
            handle,
            async_finish,
            allocate_on_comm_stream,
        )

else:
    deepep_v2_fused_dispatch = None
    deepep_v2_fused_combine = None


# HybridEP support
try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def _sync_hybridep_jit_cache(*, persist: bool) -> None:
    """Bridge HybridEP's process-local JIT directory to a persistent cache.

    ``load_cached_kernels`` only scans ``proc-<pid>``, so a new process cannot
    reuse kernels from an earlier launch without staging them into that directory.
    """
    cache_root = os.environ.get("HYBRID_EP_CACHE_DIR")
    if not cache_root:
        return

    jit_dir = Path(cache_root).expanduser() / ".deepep" / "hybrid_ep" / "jit"
    stable_dir, process_dir = jit_dir / "kernel-cache", jit_dir / f"proc-{os.getpid()}"
    source_dir, destination_dir = (process_dir, stable_dir) if persist else (stable_dir, process_dir)
    if source_dir.is_dir():
        destination_dir.mkdir(parents=True, exist_ok=True)
        for kernel in source_dir.glob("*.so"):
            try:
                os.link(kernel, destination_dir / kernel.name)
            except OSError:
                # Cache reuse is an optimization and must not prevent training.
                pass
    if persist:
        shutil.rmtree(process_dir, ignore_errors=True)


atexit.register(_sync_hybridep_jit_cache, persist=True)


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
    _sync_hybridep_jit_cache(persist=False)
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
        load_cached_kernels=True,
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
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None


class HybridEPCombine(torch.autograd.Function):
    """Fused combine operation for permute + combine a2a + permute using the HybridEP backend."""

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        """Forward pass of fused combine of the HybridEP backend."""
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
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
