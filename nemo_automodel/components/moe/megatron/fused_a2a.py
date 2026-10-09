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
import logging
import os
import shutil
import tempfile
import time

from nemo_automodel.shared.recompute_replay import RecomputeReplay, RecomputeReplayRecorder

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False


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
# Checkpoint-recompute replay of DeepEP dispatch layouts. Entries are
# (handle, recv_token_indices, recv_token_probs, tokens_per_expert): only the
# handle and the (small) per-token routing metadata are retained -- the
# dispatched activations themselves are still re-communicated, so the memory
# that activation checkpointing saves is preserved.
deepep_dispatch_replay: RecomputeReplay[tuple] = RecomputeReplay("DeepEP dispatch")


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
_nvshmem_available = None
_uccl_buffer = None

logger = logging.getLogger(__name__)


# Checkpoint-recompute replay of HybridEP dispatch layouts. Entries are
# [handle, tokens_per_expert, num_permuted_tokens]; the last slot is filled by
# finalize_hybridep_dispatch_records after the checkpoint-forward op context exits.
hybridep_dispatch_replay: RecomputeReplay[list] = RecomputeReplay("HybridEP dispatch")


def finalize_hybridep_dispatch_records(recorder: RecomputeReplayRecorder[list]) -> None:
    """Cache each recorded layout's receive extent as a host integer.

    HybridEP's sync-free replay API expects a host integer. Both the reduction and
    the device-to-host scalar conversion run only after the selective-checkpoint
    context exits; otherwise the replay-only conversion would add
    ``aten._local_scalar_dense`` to the recompute trace.
    """
    for entry in recorder.records:
        if entry[2] is None:
            entry[2] = int(entry[1].sum().item())


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
        buffer = get_buffer(group, get_hidden_bytes(x))

        # Activation-checkpoint replay: reuse the layout this dispatch computed
        # on the original forward instead of recomputing it. Cached-mode dispatch
        # returns only recv_x, so the routing metadata comes from the record.
        replay = deepep_dispatch_replay.current()
        if replay is not None and replay[1] == "replay":
            replayed = replay[0].take()
            if replayed is not None:
                cached_handle, recv_token_indices, recv_token_probs, tokens_per_expert = replayed
                recv_x, _, _, _, _, after_event_overlap = buffer.dispatch(
                    x,
                    handle=cached_handle,
                    previous_event=previous_event,
                    async_finish=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
                )
                if async_finish:
                    after_event_overlap.current_stream_wait()
                ctx.group = group
                ctx.handle = cached_handle
                ctx.async_finish = async_finish
                ctx.allocate_on_comm_stream = allocate_on_comm_stream
                return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, cached_handle)

        # Calculate layout before actual dispatch
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

        if replay is not None and replay[1] == "record":
            # Keep the handle and routing metadata (small) so the recompute can
            # skip the layout exchange; recv_x is deliberately not retained.
            replay[0].record((handle, recv_token_indices, recv_token_probs, tokens_per_expert))

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

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP."""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


# HybridEP support
try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None
_hybrid_ep_runtime_signature: tuple[object, ...] | None = None
_hybrid_ep_initialized_capacity = 0


def get_hybrid_ep_initialized_capacity(signature: tuple[object, ...]) -> int:
    """Return process-global initialized capacity after validating the resource signature.

    Args:
        signature: Full configuration and device signature for the HybridEP buffer.

    Returns:
        The greatest successfully initialized token capacity, or zero before initialization.

    Raises:
        RuntimeError: If the active process-global buffer has an incompatible signature.
    """
    if _hybrid_ep_runtime_signature is not None and _hybrid_ep_runtime_signature != signature:
        raise RuntimeError("HybridEP's process-global buffer is already active with an incompatible signature")
    return _hybrid_ep_initialized_capacity


def publish_hybrid_ep_initialized_capacity(signature: tuple[object, ...], capacity: int) -> None:
    """Publish a successfully initialized process-global HybridEP capacity.

    Args:
        signature: Full configuration and device signature for the HybridEP buffer.
        capacity: Successfully initialized token capacity.

    Raises:
        RuntimeError: If the active process-global buffer has an incompatible signature.
    """
    global _hybrid_ep_runtime_signature, _hybrid_ep_initialized_capacity
    if _hybrid_ep_runtime_signature is not None and _hybrid_ep_runtime_signature != signature:
        raise RuntimeError("HybridEP's process-global buffer is already active with an incompatible signature")
    _hybrid_ep_runtime_signature = signature
    _hybrid_ep_initialized_capacity = max(_hybrid_ep_initialized_capacity, capacity)


# HybridEP compiles its preprocessing / dispatch / combine kernels with nvcc on first use into a
# per-process directory (``$HYBRID_EP_CACHE_DIR`` or ``$HOME``, then ``.deepep/hybrid_ep/jit/proc-<pid>``),
# so every process of every job compiles them again (7 nvcc runs, ~50 s per process for the Kimi-K3
# shapes). ``NEMO_HYBRIDEP_JIT_CACHE=<dir>`` warm-starts that per-process directory from a shared
# cache and stores newly compiled kernels back into it. The compiled .so files depend only on the
# kernel config and the toolchain (rank and job id only appear in the transient file name), and
# HybridEP's ``load_cached_kernels`` loads every .so present in the per-process directory when the
# buffer is constructed, keyed by file stem.
_JIT_CACHE_ROOT = os.environ.get("NEMO_HYBRIDEP_JIT_CACHE")
_jit_proc_dir: str | None = None
_jit_shared_dir: str | None = None
_jit_stored = False


def _deep_ep_version() -> str:
    """Installed DeepEP version (dist-info first: the package itself carries no ``__version__``)."""
    try:
        from importlib.metadata import version

        return version("deep_ep")
    except Exception:
        import deep_ep

        return str(getattr(deep_ep, "__version__", "unknown"))


def _hybrid_ep_jit_dirs() -> tuple[str, str]:
    """Return (shared cache dir for this toolchain and GPU, this process's HybridEP JIT dir).

    The per-process path mirrors ``get_jit_dir()`` in DeepEP's ``csrc/hybrid_ep/jit/compiler.cu``.
    """
    base = os.environ.get("HYBRID_EP_CACHE_DIR")
    if not base:
        base = tempfile.mkdtemp(prefix="hybrid_ep_jit_")
        os.environ["HYBRID_EP_CACHE_DIR"] = base
    proc_dir = os.path.join(base, ".deepep", "hybrid_ep", "jit", f"proc-{os.getpid()}")
    major, minor = torch.cuda.get_device_capability()
    fingerprint = f"deep_ep-{_deep_ep_version()}_cuda-{torch.version.cuda}_sm{major}{minor}"
    return os.path.join(_JIT_CACHE_ROOT, fingerprint), proc_dir


def _warm_start_hybrid_ep_jit() -> bool:
    """Copy the shared cache's kernels into this process's JIT dir. Returns whether the cache is active."""
    global _jit_proc_dir, _jit_shared_dir
    if not _JIT_CACHE_ROOT:
        return False
    _jit_shared_dir, _jit_proc_dir = _hybrid_ep_jit_dirs()
    os.makedirs(_jit_proc_dir, exist_ok=True)
    loaded = 0
    if os.path.isdir(_jit_shared_dir):
        for name in sorted(os.listdir(_jit_shared_dir)):
            if name.endswith(".so"):
                shutil.copy2(os.path.join(_jit_shared_dir, name), os.path.join(_jit_proc_dir, name))
                loaded += 1
    logger.info("HybridEP JIT cache: warm-started %d kernel(s) from %s", loaded, _jit_shared_dir)
    atexit.register(store_hybrid_ep_jit_cache)
    return True


def store_hybrid_ep_jit_cache() -> int:
    """Store the kernels HybridEP compiled in this process into the shared cache (local rank 0 only).

    Files are written to a temporary name and atomically renamed, so concurrent writers from
    several nodes leave a complete file behind; existing entries are kept.
    """
    if not _jit_proc_dir or not _jit_shared_dir or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return 0
    if not os.path.isdir(_jit_proc_dir):
        return 0
    os.makedirs(_jit_shared_dir, exist_ok=True)
    stored = 0
    for name in sorted(os.listdir(_jit_proc_dir)):
        if not name.endswith(".so"):
            continue
        dst = os.path.join(_jit_shared_dir, name)
        if os.path.exists(dst):
            continue
        tmp = f"{dst}.{os.getpid()}.tmp"
        shutil.copy2(os.path.join(_jit_proc_dir, name), tmp)
        os.replace(tmp, dst)
        stored += 1
    if stored:
        logger.info("HybridEP JIT cache: stored %d new kernel(s) into %s", stored, _jit_shared_dir)
    return stored


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
    num_sms_preprocessing_api: int | None = None,
    num_blocks_permute: int | None = None,
    num_blocks_unpermute: int | None = None,
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
        num_sms_preprocessing_api: Optional SM count for routing-metadata preprocessing.
        num_blocks_permute: Optional dispatch permutation block count.
        num_blocks_unpermute: Optional combine unpermutation block count.
    """
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    load_cached_kernels = _warm_start_hybrid_ep_jit()
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
        num_sms_preprocessing_api=num_sms_preprocessing_api,
        num_blocks_permute=num_blocks_permute,
        num_blocks_unpermute=num_blocks_unpermute,
        **({"load_cached_kernels": True} if load_cached_kernels else {}),
    )


def reset_hybrid_ep_buffer():
    """Reset the HybridEP buffer and its published pipeline runtime state."""
    global _hybrid_ep_buffer, _hybrid_ep_runtime_signature, _hybrid_ep_initialized_capacity
    _hybrid_ep_buffer = None
    _hybrid_ep_runtime_signature = None
    _hybrid_ep_initialized_capacity = 0


class HybridEPDispatch(torch.autograd.Function):
    """Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        routing_map: torch.Tensor | None,
        probs: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_sms_dispatch_api: int = 24,
        num_sms_combine_api: int = 24,
        num_permuted_tokens: int | None = None,
        pad_multiple: int | None = None,
        topk_idx: torch.Tensor | None = None,
        num_experts: int | None = None,
        fuse_permute: bool = False,
        num_sms_preprocessing_api: int | None = None,
        num_blocks_permute: int | None = None,
        num_blocks_unpermute: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, object]:
        """Dispatch hidden states using compact indices or a dense routing map.

        The probability tensor remains dense so its gradient layout is unchanged. When both
        routing representations are provided, HybridEP gives ``routing_map`` precedence.

        Args:
            ctx: Autograd context that retains the dispatch handle for backward.
            x: Hidden states with shape [tokens, hidden].
            routing_map: Optional Boolean routing map with shape [tokens, experts].
            probs: Dense routing probabilities with shape [tokens, experts].
            group: Expert-parallel process group.
            num_local_experts: Number of experts owned by each rank.
            num_sms_dispatch_api: Number of SMs used by the dispatch API.
            num_sms_combine_api: Number of SMs used by the combine API.
            num_permuted_tokens: Optional static dispatched-token capacity.
            pad_multiple: Optional token padding multiple.
            topk_idx: Optional global expert indices with shape [tokens, top_k], with -1 for unused slots.
            num_experts: Global expert count required with compact indices.
            fuse_permute: Whether to fuse token permutation into dispatch.
            num_sms_preprocessing_api: Optional SM count for routing-metadata preprocessing.
            num_blocks_permute: Optional dispatch permutation block count.
            num_blocks_unpermute: Optional combine unpermutation block count.

        Returns:
            Dispatched hidden states with shape [dispatched_tokens, hidden], grouped by local
            expert; float32 probabilities with shape [dispatched_tokens], aligned with the
            hidden-state rows; no scaling metadata (FP8 dispatch is unsupported); token counts
            with shape [local_experts]; and an opaque combine handle.
        """
        first_call = _hybrid_ep_buffer is None
        if first_call:
            t_first = time.perf_counter()
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
                num_sms_preprocessing_api,
                num_blocks_permute,
                num_blocks_unpermute,
            )

        replay = hybridep_dispatch_replay.current()
        if replay is not None and replay[1] == "replay":
            replayed = replay[0].take()
            if replayed is not None:
                handle, tokens_per_expert, num_permuted_tokens = replayed
                replayed_outputs = _hybrid_ep_buffer.dispatch_with_permute(
                    hidden=x,
                    probs=probs,
                    scaling_factor=None,
                    handle=handle,
                    pad_multiple=pad_multiple,
                    num_permuted_tokens=num_permuted_tokens,
                    fuse_permute_dispatch=fuse_permute,
                )
                dispatched_hidden, dispatched_probs, dispatched_scaling_factor, _, _ = replayed_outputs
                ctx.handle = handle
                ctx.pad_multiple = pad_multiple
                ctx.fuse_permute = fuse_permute
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
            topk_idx=topk_idx,
            routing_map=routing_map,
            probs=probs,
            num_of_experts=num_experts,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            fuse_permute_dispatch=fuse_permute,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.fuse_permute = fuse_permute
        if first_call:
            # One-time cost: buffer allocation + handle exchange + nvcc JIT of the preprocessing /
            # dispatch / combine kernels (HybridEP compiles per process, so every job pays it).
            torch.cuda.synchronize(x.device)
            logger.info(
                "HybridEP first dispatch (buffer init + kernel JIT + call): %.1f s", time.perf_counter() - t_first
            )
        if replay is not None and replay[1] == "record":
            # Keep only the reusable layout; its output extent is cached by
            # finalize_hybridep_dispatch_records once the op context exits.
            # Recomputed activations and probabilities are still redispatched through it.
            replay[0].record([handle, tokens_per_expert, None])
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_x: torch.Tensor,
        grad_probs: torch.Tensor,
        grad_scaling_factor: torch.Tensor | None,
        grad_tokens_per_expert: torch.Tensor | None,
        grad_handle: None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Combine gradients while retaining the dense probability-gradient layout.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_x: Hidden-state gradients with shape [dispatched_tokens, hidden].
            grad_probs: Probability gradients with shape [dispatched_tokens], aligned with grad_x.
            grad_scaling_factor: Optional gradient for dispatch scaling metadata.
            grad_tokens_per_expert: Ignored gradient for token counts with shape [local_experts].
            grad_handle: Ignored gradient for the opaque dispatch handle.

        Returns:
            Gradients for every forward argument. The hidden-state gradient has shape
            [tokens, hidden], and the dense probability gradient has shape [tokens, experts].
        """
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            fuse_unpermute_combine=ctx.fuse_permute,
        )
        global _jit_stored
        if _JIT_CACHE_ROOT and not _jit_stored:
            # The backward combine is the last HybridEP kernel variant compiled in a training step.
            _jit_stored = True
            store_hybrid_ep_jit_cache()
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class HybridEPCombine(torch.autograd.Function):
    """Fused combine operation for permute + combine a2a + permute using the HybridEP backend."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        handle: object,
        num_permuted_tokens: int | None = None,
        pad_multiple: int | None = None,
        fuse_permute: bool = False,
    ) -> torch.Tensor:
        """Combine dispatched expert outputs through HybridEP.

        Args:
            ctx: Autograd context that retains the dispatch handle for backward.
            x: Expert outputs with shape [dispatched_tokens, hidden].
            handle: Opaque handle returned by dispatch.
            num_permuted_tokens: Optional static dispatched-token capacity.
            pad_multiple: Optional token padding multiple.
            fuse_permute: Whether to fuse output unpermutation into combine.

        Returns:
            Combined hidden states with shape [tokens, hidden].
        """
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
            fuse_unpermute_combine=fuse_permute,
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fuse_permute = fuse_permute
        return combined_hidden

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_x: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Redispatch combined-output gradients through the saved layout.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_x: Combined-output gradients with shape [tokens, hidden].

        Returns:
            Gradient for ``x`` with shape [dispatched_tokens, hidden], followed by ``None``
            for each non-tensor forward argument.
        """
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
            fuse_permute_dispatch=ctx.fuse_permute,
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    def hybrid_ep_dispatch(
        x: torch.Tensor,
        routing_map: torch.Tensor | None,
        probs: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_sms_dispatch_api: int = 24,
        num_sms_combine_api: int = 24,
        num_permuted_tokens: int | None = None,
        pad_multiple: int | None = None,
        *,
        topk_idx: torch.Tensor | None = None,
        num_experts: int | None = None,
        fuse_permute: bool = False,
        num_sms_preprocessing_api: int | None = None,
        num_blocks_permute: int | None = None,
        num_blocks_unpermute: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, object]:
        """Dispatch tokens through HybridEP while retaining dense probability layout.

        Args:
            x: Hidden states with shape [tokens, hidden].
            routing_map: Optional dense routing map with shape [tokens, experts].
            probs: Dense routing probabilities with shape [tokens, experts].
            group: Expert-parallel process group.
            num_local_experts: Number of experts owned by each rank.
            num_sms_dispatch_api: SMs reserved for HybridEP dispatch.
            num_sms_combine_api: SMs reserved for HybridEP combine.
            num_permuted_tokens: Static output capacity for non-blocking dispatch.
            pad_multiple: Optional token padding multiple.
            topk_idx: Optional global expert indices with shape [tokens, top_k], with -1 for unused slots.
            num_experts: Global expert count required with compact indices.
            fuse_permute: Fuse token permutation into the dispatch kernel.
            num_sms_preprocessing_api: Optional SM count for routing-metadata preprocessing.
            num_blocks_permute: Optional dispatch permutation block count.
            num_blocks_unpermute: Optional combine unpermutation block count.

        Returns:
            Dispatched hidden states with shape [dispatched_tokens, hidden], grouped by local
            expert; float32 probabilities with shape [dispatched_tokens], aligned with the
            hidden-state rows; no scaling metadata (FP8 dispatch is unsupported); token counts
            with shape [local_experts]; and the opaque HybridEP combine handle.
        """
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
            topk_idx,
            num_experts,
            fuse_permute,
            num_sms_preprocessing_api,
            num_blocks_permute,
            num_blocks_unpermute,
        )

    def hybrid_ep_combine(
        x: torch.Tensor,
        handle: object,
        num_permuted_tokens: int | None = None,
        pad_multiple: int | None = None,
        *,
        fuse_permute: bool = False,
    ) -> torch.Tensor:
        """Combine expert outputs through HybridEP.

        Args:
            x: Permuted expert outputs with shape [dispatched_tokens, hidden].
            handle: Opaque handle returned by :func:`hybrid_ep_dispatch`.
            num_permuted_tokens: Static token capacity used by non-blocking dispatch.
            pad_multiple: Optional token padding multiple.
            fuse_permute: Fuse output unpermutation into the combine kernel.

        Returns:
            Combined hidden states in the original token order with shape [tokens, hidden].
        """
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple, fuse_permute)

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
