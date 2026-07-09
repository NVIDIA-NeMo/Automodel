# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Setup-time prewarm utilities.

Several CUDA runtime components initialize lazily on first use:

- cuBLAS/cuBLASLt allocate their workspaces on the first backward matmul of
  each dtype;
- Triton autotuners benchmark every candidate config on a kernel's first
  launch (flash-linear-attention's gated-delta-rule backward kernels are the
  heavy case);
- NCCL creates a process group's communicator (and cudaMallocs its scratch
  buffers outside the torch caching-allocator pool) on the group's first
  collective.

When that first use happens during step 1 -- at peak activation/gradient
memory -- the out-of-pool allocation can fail with ``NCCL ... Cuda failure 2
'out of memory'`` or ``Triton Error [CUDA]: out of memory`` even though the
run would otherwise fit. Running these warmups at setup time, while the
caching-allocator pool is still small, moves the one-time initialization
costs and their memory spikes out of the first optimization step.

Prewarms are opt-in from the recipe config::

    prewarm:
      cublas_backward: true
      fla_gdn_autotune: true
      comm_groups: true
      dry_run: false  # RFC, see _dry_run_warmup

All prewarms are best-effort: failures are logged and never abort setup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

logger = logging.getLogger(__name__)


@dataclass
class PrewarmConfig:
    """Typed view over the ``prewarm`` recipe config section.

    Attributes:
        cublas_backward: Initialize cuBLAS/cuBLASLt backward-pass workspaces
            with a tiny fwd+bwd matmul per dtype.
        fla_gdn_autotune: Pre-populate the flash-linear-attention
            gated-delta-net Triton autotune caches for every GDN shape in the
            model.
        comm_groups: Eagerly create the NCCL communicators that grad-norm
            clipping will use on its first collective.
        dry_run: [RFC] Run one tiny synthetic forward+backward through the
            model to warm every lazily-initialized component at once.
    """

    cublas_backward: bool = False
    fla_gdn_autotune: bool = False
    comm_groups: bool = False
    dry_run: bool = False

    def apply(
        self,
        *,
        model_parts: list[torch.nn.Module],
        device: torch.device | int | str | None,
        pp_mesh: DeviceMesh | None = None,
        pp_enabled: bool = False,
    ) -> None:
        """Run the enabled prewarms.

        Args:
            model_parts: The (already parallelized) model parts.
            device: The device assigned to this rank, or None when no
                accelerator is available.
            pp_mesh: The pipeline-parallel submesh, if pipeline parallelism is
                enabled (its process group is warmed for the grad-norm
                all-reduce).
            pp_enabled: Whether pipeline parallelism is enabled.
        """
        if self.cublas_backward:
            _prewarm_cublas_backward(device)
        if self.fla_gdn_autotune:
            _prewarm_fla_gdn_autotune(model_parts, device)
        if self.comm_groups:
            _prewarm_comm_groups(model_parts, device, pp_mesh=pp_mesh)
        if self.dry_run:
            _dry_run_warmup(model_parts, device, pp_enabled=pp_enabled)


def _resolve_cuda_device(device: torch.device | int | str | None, label: str) -> torch.device | None:
    """Normalize ``device`` and return it if it is a usable CUDA device, else None."""
    if device is None:
        logger.info("Skipping %s prewarm: no device assigned.", label)
        return None
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if not torch.cuda.is_available() or device.type != "cuda":
        logger.info(
            "Skipping %s prewarm: device=%s cuda_available=%s",
            label,
            device,
            torch.cuda.is_available(),
        )
        return None
    torch.cuda.set_device(device)
    return device


def _prewarm_cublas_backward(device: torch.device | int | str | None, size: int = 16) -> bool:
    """Initialize cuBLAS/cuBLASLt backward-pass state before real activations exist.

    Runs a tiny fwd+bwd matmul per dtype so the library handles and workspaces
    are allocated while the allocator pool is small, instead of at step-1 peak.

    Args:
        device: Target CUDA device (skipped when None or not CUDA).
        size: Side length of the square ``[size, size]`` warmup matmul operands.

    Returns:
        True if the prewarm ran, False if it was skipped.
    """
    device = _resolve_cuda_device(device, "cuBLAS backward")
    if device is None:
        return False

    try:
        with torch.enable_grad():
            for dtype in (torch.float32, torch.bfloat16):
                lhs = torch.randn((size, size), device=device, dtype=dtype, requires_grad=True)
                rhs = torch.randn((size, size), device=device, dtype=dtype, requires_grad=True)
                loss = (lhs @ rhs).float().sum()
                loss.backward()
                torch.cuda.synchronize(device)
                del lhs, rhs, loss
        torch.cuda.empty_cache()
    except Exception:
        logger.exception("Skipping cuBLAS backward prewarm after failure.")
        return False

    logger.info("Finished cuBLAS backward prewarm (size=%d).", size)
    return True


@runtime_checkable
class _GDNAttention(Protocol):
    """Structural type of a gated-delta-net attention module.

    Matches modules (e.g. the qwen3_next / qwen3_5_moe GDN attention layers)
    that expose the head geometry needed to reconstruct the fla kernel shapes
    plus the ``chunk_gated_delta_rule`` op backed by those kernels.
    """

    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    chunk_gated_delta_rule: Callable[..., Any]


def _collect_gdn_autotune_shapes(
    model_parts: list[torch.nn.Module],
) -> dict[tuple[int, int, int, torch.dtype], str]:
    """Discover the gated-delta-net kernel shapes present in ``model_parts``.

    A module counts as a GDN attention module when it structurally matches
    :class:`_GDNAttention`. The fla autotune caches are keyed on (H, K, V[,
    BT]) -- never on sequence length or batch -- so one tiny warmup per unique
    shape covers the real workload.

    Args:
        model_parts: Model parts to scan for GDN modules.

    Returns:
        Mapping of ``(num_v_heads, head_k_dim, head_v_dim, dtype)`` to the
        qualified name of one module with that shape (for logging).
    """
    shapes: dict[tuple[int, int, int, torch.dtype], str] = {}
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, _GDNAttention):
                continue

            dtype = torch.bfloat16
            for param in module.parameters(recurse=True):
                if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    dtype = param.dtype
                    break
            shape = (int(module.num_v_heads), int(module.head_k_dim), int(module.head_v_dim), dtype)
            shapes.setdefault(shape, name)
    return shapes


def _prewarm_fla_gdn_autotune(
    model_parts: list[torch.nn.Module],
    device: torch.device | int | str | None,
    seq_len: int = 64,
) -> bool:
    """Pre-populate fla gated-delta-net autotune caches before large activations exist.

    On its first launch each Triton-autotuned kernel benchmarks all candidate
    configs; when that first launch is the step-1 backward at peak memory, the
    benchmark allocations can fail with ``Triton Error [CUDA]: out of
    memory``. The autotune cache keys are shape-based -- (H, K, V) and the
    chunk size, never the sequence length -- so launching each kernel once
    here with tiny tensors caches the winning configs for the real workload.

    Two warmups run per unique GDN shape found in ``model_parts``:

    1. the context-parallel backward kernels (pre-process and state merge),
       which the end-to-end warmup below cannot reach, and
    2. a tiny end-to-end fwd+bwd through the public ``chunk_gated_delta_rule``
       op, which covers every autotuned kernel in its call graph (per-kernel
       prewarms alone are whack-a-mole: warming one kernel just moves the
       step-1 autotune OOM to the next cold kernel).

    Args:
        model_parts: Model parts to scan for GDN modules.
        device: Target CUDA device (skipped when None or not CUDA).
        seq_len: Warmup sequence length (not part of the autotune key).

    Returns:
        True if at least the end-to-end prewarm ran, False if skipped.
    """
    device = _resolve_cuda_device(device, "fla GDN autotune")
    if device is None:
        return False

    shapes = _collect_gdn_autotune_shapes(model_parts)
    if not shapes:
        logger.warning("fla GDN autotune prewarm enabled but no gated-delta-net modules were found.")
        return False

    logger.info(
        "Prewarming fla GDN autotune caches for %d unique shape(s): %s",
        len(shapes),
        sorted((h, k, v, str(dtype), name) for (h, k, v, dtype), name in shapes.items()),
    )

    _prewarm_fla_gdn_cp_kernels(shapes, device, seq_len)
    return _prewarm_fla_gdn_end_to_end(shapes, device, seq_len)


# Keyword arguments the launches in _prewarm_fla_gdn_cp_kernels pass to the
# private fla CP kernels. pyproject only lower-bounds the fla version, so the
# installed kernels' parameter lists are validated against this hand-derived
# launch contract before launching; on drift the prewarm is skipped.
_FLA_PRE_PROCESS_BWD_KERNEL_ARGS = frozenset(
    (
        "q",
        "k",
        "w",
        "g",
        "gk",
        "do",
        "dhm",
        "dv",
        "cu_seqlens",
        "scale",
        "T",
        "H",
        "K",
        "V",
        "BT",
        "BK1",
        "USE_EXP2",
        "BLOCK_SIZE",
    )
)
_FLA_MERGE_FWD_BWD_KERNEL_ARGS = frozenset(
    (
        "h",
        "ag_hm",
        "pre_or_post_num_ranks",
        "rank",
        "seq_offsets",
        "init_offsets",
        "h0_seq_ids",
        "h0",
        "H",
        "K",
        "V",
        "BK",
        "FORWARD",
        "INTRACARD_MODE",
        "NUM_SEQ_ENTRIES",
    )
)


def _triton_kernel_accepts(kernel: object, expected_args: frozenset[str], kernel_name: str) -> bool:
    """Check that a (possibly wrapped) Triton kernel accepts the expected launch arguments.

    Unwraps autotuner/heuristics layers via their ``fn`` attribute until an
    object exposing the JITFunction's ``arg_names`` is found.

    Args:
        kernel: The Triton kernel object (a JITFunction, or an Autotuner /
            Heuristics wrapper around one).
        expected_args: Keyword-argument names the prewarm launch will pass.
        kernel_name: Kernel name used in log messages.

    Returns:
        True when every expected argument is in the kernel's parameter list.
    """
    arg_names = None
    unwrapped = kernel
    for _ in range(8):
        if unwrapped is None:
            break
        arg_names = getattr(unwrapped, "arg_names", None)
        if arg_names is not None:
            break
        unwrapped = getattr(unwrapped, "fn", None)
    if arg_names is None:
        logger.warning(
            "Cannot determine the parameter list of fla kernel %s; skipping its prewarm.",
            kernel_name,
        )
        return False
    missing = expected_args - set(arg_names)
    if missing:
        logger.warning(
            "fla kernel %s does not accept expected parameter(s) %s; the installed fla version has drifted "
            "from the prewarm launch contract, skipping its prewarm.",
            kernel_name,
            sorted(missing),
        )
        return False
    return True


def _prewarm_fla_gdn_cp_kernels(
    shapes: dict[tuple[int, int, int, torch.dtype], str],
    device: torch.device,
    seq_len: int,
) -> None:
    """Warm the fla context-parallel GDN backward kernels (best-effort).

    These kernels only fire on the context-parallel code path, which the
    end-to-end warmup in :func:`_prewarm_fla_gdn_end_to_end` does not reach,
    so they are launched once directly with zero-filled tiny tensors (the
    autotuner only measures timing; values are irrelevant). fla builds without
    CP kernels are skipped, as are kernels whose parameter list no longer
    matches the launch contract here (the kernels are private fla API, so the
    signature is validated before every launch attempt).

    Args:
        shapes: Mapping of ``(num_v_heads, head_k_dim, head_v_dim, dtype)`` to
            a module name, as returned by :func:`_collect_gdn_autotune_shapes`.
        device: Target CUDA device.
        seq_len: Warmup sequence length (not part of the autotune key).
    """
    has_triton, triton = safe_import("triton")
    has_pre_process, pre_process_bwd_kernel_merged = safe_import_from(
        "fla.ops.cp.chunk_delta_h", "pre_process_bwd_kernel_merged"
    )
    if not (has_triton and has_pre_process):
        logger.info("Skipping fla CP GDN kernel prewarm: fla CP kernels not importable.")
        return
    has_merge, merge_fwd_bwd_kernel = safe_import_from("fla.ops.cp.chunk_delta_h", "merge_fwd_bwd_kernel")
    if not has_merge:
        logger.info("fla merge_fwd_bwd_kernel not present in this fla version; skipping its prewarm.")

    warm_pre_process = _triton_kernel_accepts(
        pre_process_bwd_kernel_merged, _FLA_PRE_PROCESS_BWD_KERNEL_ARGS, "pre_process_bwd_kernel_merged"
    )
    warm_merge = has_merge and _triton_kernel_accepts(
        merge_fwd_bwd_kernel, _FLA_MERGE_FWD_BWD_KERNEL_ARGS, "merge_fwd_bwd_kernel"
    )
    if not (warm_pre_process or warm_merge):
        return

    with torch.no_grad():
        for (num_heads, head_k_dim, head_v_dim, dtype), module_name in shapes.items():
            block_size = 32 if head_k_dim <= 64 else 64
            bk1 = triton.next_power_of_2(head_k_dim)

            if warm_pre_process:
                grid = (
                    triton.cdiv(head_v_dim, block_size) + triton.cdiv(head_k_dim, block_size),
                    num_heads,
                )

                q = torch.zeros((1, seq_len, num_heads, head_k_dim), device=device, dtype=dtype)
                k = torch.zeros_like(q)
                w = torch.zeros_like(q)
                g = torch.zeros((1, seq_len, num_heads), device=device, dtype=torch.float32)
                do = torch.zeros((1, seq_len, num_heads, head_v_dim), device=device, dtype=dtype)
                dv = torch.zeros_like(do)
                dhm = torch.zeros((num_heads, head_k_dim, head_v_dim + head_k_dim), device=device, dtype=torch.float32)
                cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.long)

                logger.info(
                    "Prewarming fla CP GDN bwd kernel | module=%s H=%d K=%d V=%d dtype=%s",
                    module_name,
                    num_heads,
                    head_k_dim,
                    head_v_dim,
                    dtype,
                )
                try:
                    pre_process_bwd_kernel_merged[grid](
                        q=q,
                        k=k,
                        w=w,
                        g=g,
                        gk=None,
                        do=do,
                        dhm=dhm,
                        dv=dv,
                        cu_seqlens=cu_seqlens,
                        scale=1.0,
                        T=seq_len,
                        H=num_heads,
                        K=head_k_dim,
                        V=head_v_dim,
                        BT=64,
                        BK1=bk1,
                        USE_EXP2=False,
                        BLOCK_SIZE=block_size,
                    )
                    torch.cuda.synchronize(device)
                except Exception:
                    logger.exception("fla CP GDN pre-process prewarm failed for %s; continuing.", module_name)
                finally:
                    del q, k, w, g, do, dv, dhm, cu_seqlens
                    torch.cuda.empty_cache()

            if not warm_merge:
                continue

            # Also warm the CP-mode state merge kernel. In fla's CP GDN
            # backward the first CP rank skips pre_process_bwd_kernel_merged
            # and hits merge_fwd_bwd_kernel cold; its autotuner then
            # benchmarks its configs at peak backward memory. The autotune
            # key is (H, K, V) -- num_ranks and rank are not part of it -- so
            # a tiny launch here caches the winning config for real world
            # sizes.
            #
            # CP-mode indexing inside the kernel: the BWD variant reads
            # all-gathered rank-slices rank+1 .. rank+num_ranks and the FWD
            # variant reads rank-num_ranks .. rank-1. Allocate 2 rank-slices
            # and launch with num_ranks=1 so BWD (rank=0) touches slice 1 and
            # FWD (rank=1) touches slice 0 -- both in bounds.
            h_state = torch.zeros((num_heads, head_k_dim, head_v_dim), device=device, dtype=torch.float32)
            ag_hm = torch.zeros(
                (2 * num_heads, head_k_dim, head_v_dim + head_k_dim),
                device=device,
                dtype=torch.float32,
            )

            def _merge_grid(meta, _V=head_v_dim, _H=num_heads):
                return (triton.cdiv(_V, meta["BV"]), _H)

            try:
                for forward_mode, warm_rank in ((False, 0), (True, 1)):
                    merge_fwd_bwd_kernel[_merge_grid](
                        h=h_state,
                        ag_hm=ag_hm,
                        pre_or_post_num_ranks=1,
                        rank=warm_rank,
                        seq_offsets=None,
                        init_offsets=None,
                        h0_seq_ids=None,
                        h0=None,
                        H=num_heads,
                        K=head_k_dim,
                        V=head_v_dim,
                        BK=bk1,
                        FORWARD=forward_mode,
                        INTRACARD_MODE=False,
                        NUM_SEQ_ENTRIES=0,
                    )
                torch.cuda.synchronize(device)
            except Exception:
                logger.exception("fla CP GDN merge prewarm failed for %s; continuing.", module_name)
            finally:
                del h_state, ag_hm
                torch.cuda.empty_cache()


def _prewarm_fla_gdn_end_to_end(
    shapes: dict[tuple[int, int, int, torch.dtype], str],
    device: torch.device,
    seq_len: int,
) -> bool:
    """Run a tiny fwd+bwd through ``chunk_gated_delta_rule`` per unique shape.

    This caches the autotune config of every kernel in the op's call graph
    (chunk fwd, dqkwg backward, dhu pre-process, ...) while the allocator pool
    is empty. Must run with grad enabled so the backward kernels fire.

    Args:
        shapes: Mapping of ``(num_v_heads, head_k_dim, head_v_dim, dtype)`` to
            a module name, as returned by :func:`_collect_gdn_autotune_shapes`.
        device: Target CUDA device.
        seq_len: Warmup sequence length (not part of the autotune key).

    Returns:
        True if the warmup ran for at least one shape.
    """
    has_gdr, chunk_gated_delta_rule = safe_import_from("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
    if not has_gdr:
        logger.info("Skipping fla GDN end-to-end prewarm: fla is not importable.")
        return False

    ran = False
    for (num_heads, head_k_dim, head_v_dim, dtype), module_name in shapes.items():
        q = k = v = g = beta = out = None
        try:
            q = torch.randn((1, seq_len, num_heads, head_k_dim), device=device, dtype=dtype, requires_grad=True)
            k = torch.randn((1, seq_len, num_heads, head_k_dim), device=device, dtype=dtype, requires_grad=True)
            v = torch.randn((1, seq_len, num_heads, head_v_dim), device=device, dtype=dtype, requires_grad=True)
            g = torch.zeros((1, seq_len, num_heads), device=device, dtype=torch.float32, requires_grad=True)
            beta = torch.full((1, seq_len, num_heads), 0.5, device=device, dtype=dtype, requires_grad=True)
            out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            out.sum().backward()
            torch.cuda.synchronize(device)
            ran = True
        except Exception:
            logger.exception("fla GDN end-to-end prewarm failed for %s; continuing.", module_name)
        finally:
            del q, k, v, g, beta, out
            torch.cuda.empty_cache()

    logger.info("Finished fla GDN autotune prewarm.")
    return ran


def _prewarm_comm_groups(
    model_parts: list[torch.nn.Module],
    device: torch.device | int | str | None,
    pp_mesh: DeviceMesh | None = None,
) -> int:
    """Eagerly create the NCCL communicators gradient-norm clipping will use.

    torch creates a process group's NCCL communicator lazily on its first
    collective. ``clip_grad_norm`` all-reduces once per Shard mesh dim of each
    gradient DTensor group, and for some dims that first collective runs at
    step-1 peak memory; NCCL then cudaMallocs its scratch buffers outside the
    torch pool and can die with ``Cuda failure 2 'out of memory'`` after a
    clean forward+backward. Warm exactly those groups here, while the torch
    pool is still small, by replaying ``clip_grad_norm``'s own (mesh,
    shard-dim) enumeration and issuing a one-element all-reduce per group.

    Args:
        model_parts: Model parts whose DTensor parameters define the groups
            (every non-``Replicate`` placement dim of each parameter's mesh).
        device: Device for the scalar warmup all-reduce; falls back to the
            current CUDA device (or CPU) when None.
        pp_mesh: Pipeline-parallel submesh, if enabled. ``clip_grad_norm``
            also all-reduces the total norm across the PP group, but
            parameters are never sharded along pp, so the placement
            enumeration alone cannot discover that group.

    Returns:
        The number of process groups warmed.
    """
    if not torch.distributed.is_initialized():
        logger.info("Skipping comm-group prewarm: torch.distributed is not initialized.")
        return 0

    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)

    seen: set[int] = set()
    groups = []
    for part in model_parts:
        for p in part.parameters():
            if not isinstance(p, DTensor):
                continue
            mesh = p.device_mesh
            for dim_idx, pl in enumerate(p.placements):
                if isinstance(pl, Replicate):
                    continue
                group = mesh.get_group(mesh_dim=dim_idx)
                if id(group) in seen:
                    continue
                seen.add(id(group))
                groups.append(group)

    if pp_mesh is not None:
        try:
            group = pp_mesh.get_group()
        except Exception:
            logger.exception("Failed to resolve the PP mesh process group; skipping its prewarm.")
            group = None
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            groups.append(group)

    if not groups:
        return 0
    t = torch.zeros((), dtype=torch.float32, device=device)
    for group in groups:
        torch.distributed.all_reduce(t, group=group)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logger.info("Prewarmed %d process group(s) for grad-norm clipping.", len(groups))
    return len(groups)


def _dry_run_warmup(
    model_parts: list[torch.nn.Module],
    device: torch.device | int | str | None,
    *,
    seq_len: int = 64,
    pp_enabled: bool = False,
) -> bool:
    """[RFC] Run one tiny synthetic forward+backward through the model at setup time.

    The targeted prewarms above each address one lazy-initialization source; a
    dry-run step is the general answer: it exercises the model's real fwd+bwd
    call graph, so every lazily-initialized kernel library, Triton autotune
    cache, and gradient-reduction communicator (e.g. FSDP reduce-scatter) used
    by training is warmed with a tiny input while the allocator pool is empty.

    The dry run is side-effect free with respect to training state:

    - gradients are discarded with ``zero_grad(set_to_none=True)`` afterwards.
      (The loss is also scaled by zero so most produced gradients are
      numerically zero, but that alone is not a guarantee: the MoE aux-loss
      backward, ``MoEAuxLossAutoScaler``, injects a fixed gradient scale
      regardless of the incoming grad -- the ``zero_grad`` reset is what
      restores the pre-dry-run state);
    - no optimizer step runs;
    - module buffers (e.g. BatchNorm running statistics) are snapshotted
      before the forward and restored afterwards;
    - MoE router gates that accumulate expert load for the correction-bias
      update during training-mode forwards are reset via their public
      ``reset_cumulative_expert_load`` hook (matched structurally, since
      components must not import each other);
    - RNG state (CPU and the target CUDA device) is forked and restored, so
      data order and dropout draws in real training are unchanged.

    Known limitations (why this is an RFC):

    - Pipeline parallelism is skipped: a synthetic microbatch must flow
      through the PP schedule for stage-boundary communicators to warm, which
      needs recipe-level integration rather than a plain ``model(...)`` call.
    - Models whose forward requires more than ``input_ids`` (e.g.
      vision-language models needing pixel inputs) are only warmed as far as
      the forward gets; failures are logged and ignored.
    - A tiny dense synthetic batch may not reach code paths gated on real
      data (packed-sequence branches, MoE expert-routing spread).

    Args:
        model_parts: Model parts; only ``model_parts[0]`` is used (single-part
            models when PP is disabled).
        device: Device for the synthetic ``[1, seq_len]`` ``input_ids`` batch;
            falls back to the current CUDA device (or CPU) when None.
        seq_len: Sequence length of the synthetic batch.
        pp_enabled: Whether pipeline parallelism is enabled (skips the dry run).

    Returns:
        True if the dry run completed, False if it was skipped or failed.
    """
    if pp_enabled:
        logger.warning("Skipping dry-run warmup: pipeline parallelism is not supported yet.")
        return False
    if not model_parts:
        return False

    model = model_parts[0]
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None and hasattr(model, "get_input_embeddings"):
        vocab_size = getattr(model.get_input_embeddings(), "num_embeddings", None)
    if vocab_size is None:
        logger.warning("Skipping dry-run warmup: could not determine vocab size.")
        return False

    # Snapshot module buffers so the restore below erases any in-place buffer
    # mutation done by the dry-run forward (BatchNorm running stats, ...).
    buffer_snapshot = [(buf, buf.detach().clone()) for part in model_parts for buf in part.buffers()]

    fork_devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            input_ids = torch.randint(0, int(vocab_size), (1, seq_len), device=device)
            output = model(input_ids=input_ids)
            logits = getattr(output, "logits", None)
            if logits is None:
                logits = output[0] if isinstance(output, (tuple, list)) else output
            # Zero-scaled loss: the backward kernels and gradient-reduction
            # collectives all run, but the produced gradients are unused and
            # discarded by the zero_grad reset below.
            (logits.float().sum() * 0.0).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        logger.exception("Dry-run warmup failed; continuing without it.")
        return False
    finally:
        with torch.no_grad():
            for buf, saved in buffer_snapshot:
                buf.copy_(saved)
        for part in model_parts:
            part.zero_grad(set_to_none=True)
            for module in part.modules():
                # MoE router gates accumulate expert load on every
                # training-mode forward; drop what the dry run added so the
                # first real correction-bias update only sees real routing
                # statistics. Matched structurally: components must not
                # import each other, so Gate cannot be imported here.
                reset_expert_load = getattr(module, "reset_cumulative_expert_load", None)
                if callable(reset_expert_load):
                    reset_expert_load()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Finished dry-run warmup (seq_len=%d).", seq_len)
    return True
