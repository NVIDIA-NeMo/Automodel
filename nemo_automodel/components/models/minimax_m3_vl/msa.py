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

"""Flat MSA prefill training adapter for MiniMax M3 sparse attention.

Forward consumes compact Q/K/V directly and never constructs a paged or
aligned K/V view. The vendored SM100 backward alone materializes a temporary
128-row-aligned K/V workspace. Canonical support crosses the Adapter seam once
as document-local q2k[4,T,16]; CSR and scheduling tensors derived during
forward are transient execution metadata reused by backward.
"""

import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_patch import _patch_msa_fmax
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import _MSABackwardSchedule
from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSALaunchMetadata, _MSAPackedLayout
from nemo_automodel.shared.import_utils import UnavailableError, safe_import, safe_import_from

_MSA_BLOCK_SIZE = 128
_MSA_TOPK_BLOCKS = 16
_MSA_QUERY_HEADS = 64
_MSA_KV_HEADS = 4
_MSA_INDEX_HEADS = 4
_MSA_HEAD_DIM = 128
_MSA_ATTENTION_DROPOUT = 0.0

_MSA_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' requires the fixed fmha-sm100 optional dependency. "
    "Install the project with uv sync --extra msa on a CUDA SM100 system; the MSA revision "
    "must be compatible with nvidia-cutlass-dsl==4.6.2."
)
_MSA_BACKWARD_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' backward requires nvidia-cutlass-dsl==4.6.2 and cuda-python from the "
    "msa optional dependency. Install the project with uv sync --extra msa on a CUDA SM100 system."
)

_MSA_BACKWARD_MODULE = "nemo_automodel.components.models.minimax_m3_vl.kernels.msa_backward_sm100"


@dataclass(frozen=True, slots=True)
class _MSAForwardKernels:
    """The optional ``fmha_sm100`` entry points the flat forward launches.

    Attributes:
        build_k2q_csr: Builds the key-block-to-query CSR and forward schedule
            from canonical document-local ``q2k``.
        sparse_atten_func: Runs the flat, non-paged SM100 sparse prefill.
    """

    build_k2q_csr: Callable[..., Any]
    sparse_atten_func: Callable[..., Any]


# Both resolvers are cached and never run at import time: importing this
# model-private module stays free for generic MiniMax M3 runs, and the heavy
# CuTe DSL dependency is probed only after the MSA gates and validation pass.
@lru_cache(maxsize=1)
def _resolve_msa_forward() -> _MSAForwardKernels | None:
    """Resolve the optional forward entry points once, or ``None`` if absent."""
    available, module = safe_import("fmha_sm100.sparse", msg=_MSA_IMPORT_ERROR)
    if not available:
        return None
    build_k2q_csr = getattr(module, "build_k2q_csr", None)
    sparse_atten_func = getattr(module, "sparse_atten_func", None)
    if not callable(build_k2q_csr) or not callable(sparse_atten_func):
        return None
    _patch_msa_fmax(module)
    return _MSAForwardKernels(build_k2q_csr=build_k2q_csr, sparse_atten_func=sparse_atten_func)


@lru_cache(maxsize=1)
def _resolve_msa_backward() -> Callable[..., Any] | None:
    """Resolve the model-private SM100 backward launcher once, or ``None``."""
    available, launcher = safe_import_from(
        _MSA_BACKWARD_MODULE,
        "_run_msa_backward",
        msg=_MSA_BACKWARD_IMPORT_ERROR,
    )
    return launcher if available and callable(launcher) else None


def _require_msa() -> _MSAForwardKernels:
    """Resolve and return the optional ``fmha_sm100`` forward entry points.

    Raises:
        UnavailableError: If the fixed dependency or its sparse entry points are
            unavailable.
    """
    kernels = _resolve_msa_forward()
    if kernels is None:
        raise UnavailableError(_MSA_IMPORT_ERROR)
    return kernels


def _require_msa_backward() -> Callable[..., Any]:
    """Resolve and return the model-private SM100 backward launcher.

    Raises:
        UnavailableError: If the CuTe DSL runtime dependencies are unavailable.
    """
    launcher = _resolve_msa_backward()
    if launcher is None:
        raise UnavailableError(_MSA_BACKWARD_IMPORT_ERROR)
    return launcher


def _validate_msa_topology(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_index_heads: int,
    block_size: int,
    topk_blocks: int,
    attention_dropout: float,
) -> None:
    """Reject a layer whose topology cannot satisfy the fixed MSA kernel ABI.

    Canonical ``q2k`` carries document-local block ids and no geometry, so the
    launch topology has to be checked separately from the tensor contract.

    Args:
        num_heads: Main-attention query heads.
        num_kv_heads: Main-attention key/value heads.
        head_dim: Per-head channel count of Q, K, and V.
        num_index_heads: Indexer heads producing canonical support.
        block_size: Indexer key-block size in tokens.
        topk_blocks: Selected key blocks per query.
        attention_dropout: Main-attention dropout probability.

    Raises:
        ValueError: If any value differs from the first supported MSA contract.
    """
    actual = (num_heads, num_kv_heads, head_dim, num_index_heads, block_size, topk_blocks)
    expected = (
        _MSA_QUERY_HEADS,
        _MSA_KV_HEADS,
        _MSA_HEAD_DIM,
        _MSA_INDEX_HEADS,
        _MSA_BLOCK_SIZE,
        _MSA_TOPK_BLOCKS,
    )
    if actual != expected:
        raise ValueError(
            f"MiniMax M3 MSA requires num_heads={expected[0]}, num_kv_heads={expected[1]}, "
            f"head_dim={expected[2]}, indexer.num_index_heads={expected[3]}, "
            f"indexer.block_size={expected[4]}, and indexer.topk_blocks={expected[5]}; got "
            f"num_heads={actual[0]}, num_kv_heads={actual[1]}, head_dim={actual[2]}, "
            f"indexer.num_index_heads={actual[3]}, indexer.block_size={actual[4]}, "
            f"indexer.topk_blocks={actual[5]}."
        )
    if attention_dropout != _MSA_ATTENTION_DROPOUT:
        raise ValueError(
            f"MiniMax M3 MSA first supports attention_dropout={_MSA_ATTENTION_DROPOUT:g} only; "
            f"got attention_dropout={attention_dropout}."
        )


def _validate_flat_msa_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    metadata: _MSALaunchMetadata,
    *,
    softmax_scale: float,
) -> None:
    """Validate the fixed flat MSA launch contract.

    Args:
        q: Contiguous, 16-byte-aligned BF16 CUDA queries [T,64,128].
        k: Contiguous, 16-byte-aligned BF16 CUDA keys [T,4,128].
        v: Contiguous, 16-byte-aligned BF16 CUDA values [T,4,128].
        q2k: Contiguous, 16-byte-aligned int32 document-local support [4,T,16].
        metadata: Packed launch coordinates for the same T rows; field layouts
            are documented on :class:`_MSALaunchMetadata`.
        softmax_scale: Positive finite QK scale.

    Raises:
        ValueError: If a shape, dtype, device, contiguity, or scalar contract is violated.
        NotImplementedError: If the tensors are not on SM100.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "MiniMax M3 MSA requires flat rank-3 q/k/v tensors; got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )

    total_tokens, query_heads, head_dim = q.shape
    expected_kv = (total_tokens, _MSA_KV_HEADS, _MSA_HEAD_DIM)
    if (query_heads, head_dim) != (_MSA_QUERY_HEADS, _MSA_HEAD_DIM):
        raise ValueError(f"MiniMax M3 MSA requires q[T,64,128], got q={tuple(q.shape)}.")
    if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
        raise ValueError(
            "MiniMax M3 MSA requires matching k/v[T,4,128], got "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}, expected={expected_kv}."
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"MiniMax M3 MSA first supports BF16 q/k/v only; got q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError(
            "MiniMax M3 MSA requires CUDA q/k/v tensors on one SM100 device; got "
            f"q={q.device}, k={k.device}, v={v.device}."
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q/k/v must share one CUDA device, got q={q.device}, k={k.device}, v={v.device}.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("MiniMax M3 MSA requires contiguous flat q/k/v tensors.")
    misaligned = [name for name, tensor in (("q", q), ("k", k), ("v", v)) if tensor.data_ptr() % 16 != 0]
    if misaligned:
        raise ValueError(
            "MiniMax M3 MSA requires 16-byte-aligned q/k/v storage for vectorized kernel loads; "
            f"misaligned tensors={misaligned}."
        )
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (10, 0):
        raise NotImplementedError(
            "MiniMax M3 MSA first supports SM100 (compute capability 10.0) only; got compute capability "
            f"{capability[0]}.{capability[1]} on {q.device}. Use sparse_attn='generic' on this GPU."
        )

    metadata_tensors = (
        metadata.workspace_positions,
        metadata.document_workspace_starts,
        metadata.cu_seqlens,
    )
    if any(tensor.device != q.device for tensor in metadata_tensors):
        devices = tuple(tensor.device for tensor in metadata_tensors)
        raise ValueError(f"MSA packed-layout tensors must be on {q.device}, got devices={devices}.")
    if metadata.total_tokens != total_tokens:
        raise ValueError(
            f"MSA packed layout contains {metadata.total_tokens} tokens, but q/k/v contain {total_tokens}."
        )
    expected_q2k = (_MSA_KV_HEADS, total_tokens, _MSA_TOPK_BLOCKS)
    if tuple(q2k.shape) != expected_q2k:
        raise ValueError(f"q2k must have fixed layout [4,T,16], got {tuple(q2k.shape)}, expected={expected_q2k}.")
    if q2k.dtype != torch.int32 or not q2k.is_contiguous():
        raise ValueError(f"q2k must be contiguous int32, got dtype={q2k.dtype}, contiguous={q2k.is_contiguous()}.")
    if q2k.device != q.device:
        raise ValueError(f"q2k must be on {q.device}, got {q2k.device}.")
    metadata_and_support = (
        ("q2k", q2k),
        ("workspace_positions", metadata.workspace_positions),
        ("document_workspace_starts", metadata.document_workspace_starts),
        ("cu_seqlens", metadata.cu_seqlens),
    )
    misaligned = [name for name, tensor in metadata_and_support if tensor.data_ptr() % 16 != 0]
    if misaligned:
        raise ValueError(
            f"MiniMax M3 MSA requires 16-byte-aligned support and layout storage; misaligned tensors={misaligned}."
        )
    if metadata.workspace_positions.dtype != torch.int64:
        raise ValueError(
            f"MSA packed-layout workspace positions must be int64, got {metadata.workspace_positions.dtype}."
        )
    if metadata.document_workspace_starts.dtype != torch.int32:
        raise ValueError(
            "MSA packed-layout document workspace starts must be int32, got "
            f"{metadata.document_workspace_starts.dtype}."
        )
    if metadata.cu_seqlens.dtype != torch.int32 or not metadata.cu_seqlens.is_contiguous():
        raise ValueError(
            "MSA packed-layout cu_seqlens must be contiguous int32, got "
            f"dtype={metadata.cu_seqlens.dtype}, contiguous={metadata.cu_seqlens.is_contiguous()}."
        )
    if metadata.workspace_size <= 0 or metadata.workspace_size % _MSA_BLOCK_SIZE != 0:
        raise ValueError(
            f"MSA packed-layout workspace size must be a positive multiple of 128, got {metadata.workspace_size}."
        )
    if metadata.max_seqlen <= 0:
        raise ValueError(f"MSA packed-layout max sequence length must be positive, got {metadata.max_seqlen}.")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}.")


def _align_backward_tensor(
    compact: torch.Tensor,
    workspace_positions: torch.Tensor,
    workspace_size: int,
) -> torch.Tensor:
    """Create the zero-filled aligned K/V view used only by backward.

    Args:
        compact: Compact activation [T,H,D].
        workspace_positions: Int64 compact-to-workspace row map [T].
        workspace_size: Positive 128-aligned workspace length W.

    Returns:
        Contiguous tensor [W,H,D] with exact-zero alignment tails.

    Raises:
        ValueError: If shape, dtype, device, or alignment contracts disagree.
    """
    if compact.ndim != 3:
        raise ValueError(f"compact must have layout [T,H,D], got shape={tuple(compact.shape)}.")
    if workspace_positions.ndim != 1 or workspace_positions.shape[0] != compact.shape[0]:
        raise ValueError(
            "workspace_positions must have one entry per compact row, got "
            f"shape={tuple(workspace_positions.shape)} for compact={tuple(compact.shape)}."
        )
    if workspace_positions.dtype != torch.int64:
        raise ValueError(f"workspace_positions must be int64, got {workspace_positions.dtype}.")
    if workspace_positions.device != compact.device:
        raise ValueError(
            f"compact and workspace_positions must share a device, got {compact.device}/{workspace_positions.device}."
        )
    if workspace_size <= 0 or workspace_size % _MSA_BLOCK_SIZE != 0:
        raise ValueError(f"workspace_size must be a positive multiple of 128, got {workspace_size}.")
    workspace = compact.new_zeros((workspace_size, compact.shape[1], compact.shape[2]))
    return workspace.index_copy(0, workspace_positions, compact).contiguous()


class _MSASparseAttentionFunction(torch.autograd.Function):
    """Own flat forward state and the backward-only aligned workspace lifetime."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        metadata: _MSALaunchMetadata,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run the official MSA forward on compact Q/K/V.

        Args:
            ctx: PyTorch custom-autograd context.
            q: Compact, 16-byte-aligned queries [T,64,128] BF16 CUDA.
            k: Compact, 16-byte-aligned keys [T,4,128] BF16 CUDA.
            v: Compact, 16-byte-aligned values [T,4,128] BF16 CUDA.
            q2k: Canonical 16-byte-aligned document-local support [4,T,16] int32 CUDA.
            metadata: Packed launch coordinates for the same T compact rows;
                field layouts are documented on :class:`_MSALaunchMetadata`.
            softmax_scale: Positive finite QK scale.

        Returns:
            Compact MSA output [T,64,128] BF16 CUDA.
        """
        ctx.set_materialize_grads(False)
        cu_seqlens = metadata.cu_seqlens
        max_seqlen = int(metadata.max_seqlen)
        kernels = _require_msa()
        # The optional CSR extension obtains the current CUDA stream without a
        # CUDAGuard, so bind both external launches to the tensor's device.
        # Direct unit tests use CPU stand-ins to exercise this private autograd
        # seam; the public adapter rejects non-CUDA inputs before reaching it.
        device_context = torch.cuda.device(q.device) if q.is_cuda else nullcontext()
        with device_context:
            row_ptr, q_indices, schedule = kernels.build_k2q_csr(
                q2k,
                cu_seqlens,
                cu_seqlens,
                _MSA_BLOCK_SIZE,
                total_k=q.shape[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                total_rows=int(metadata.workspace_size) // _MSA_BLOCK_SIZE,
                qhead_per_kv=_MSA_QUERY_HEADS // _MSA_KV_HEADS,
                return_schedule=True,
            )
            out, lse = kernels.sparse_atten_func(
                q,
                k,
                v,
                row_ptr,
                q_indices,
                _MSA_TOPK_BLOCKS,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                blk_kv=_MSA_BLOCK_SIZE,
                causal=True,
                softmax_scale=float(softmax_scale),
                partial_dtype=torch.bfloat16,
                return_softmax_lse=True,
                schedule=schedule,
            )
        expected_out = (q.shape[0], _MSA_QUERY_HEADS, _MSA_HEAD_DIM)
        expected_lse = (q.shape[0], _MSA_QUERY_HEADS)
        if tuple(out.shape) != expected_out or out.dtype != torch.bfloat16:
            raise RuntimeError(
                f"MSA forward returned out {tuple(out.shape)}/{out.dtype}, expected {expected_out}/bf16."
            )
        if tuple(lse.shape) != expected_lse or lse.dtype != torch.float32:
            raise RuntimeError(
                f"MSA forward returned LSE {tuple(lse.shape)}/{lse.dtype}, expected {expected_lse}/fp32."
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            row_ptr,
            q_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            metadata.workspace_positions,
            metadata.document_workspace_starts,
            cu_seqlens,
        )
        ctx.workspace_size = int(metadata.workspace_size)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor | None) -> tuple[Any, ...]:
        """Run the vendored backward using forward-derived schedule metadata.

        Args:
            ctx: Autograd context populated by forward.
            grad_out: Compact upstream gradient [T,64,128] BF16 CUDA, or None.

        Returns:
            One gradient per forward argument in order; only the leading compact
            dQ [T,64,128], dK [T,4,128], and dV [T,4,128] are tensors, and the
            trailing ``q2k``, ``metadata``, and ``softmax_scale`` slots are None.
        """
        (
            q,
            k,
            v,
            out,
            lse,
            row_ptr,
            q_indices,
            scheduler_metadata,
            work_count,
            workspace_positions,
            document_workspace_starts,
            cu_seqlens,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros_like(out)

        k_aligned = _align_backward_tensor(k, workspace_positions, ctx.workspace_size)
        v_aligned = _align_backward_tensor(v, workspace_positions, ctx.workspace_size)
        schedule = _MSABackwardSchedule(
            row_ptr=row_ptr,
            q_indices=q_indices,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            cu_seqlens=cu_seqlens,
            document_workspace_starts=document_workspace_starts,
        )
        run_msa_backward = _require_msa_backward()
        device_context = torch.cuda.device(q.device) if q.is_cuda else nullcontext()
        with device_context:
            dq, dk_workspace, dv_workspace = run_msa_backward(
                q,
                k_aligned,
                v_aligned,
                grad_out.contiguous(),
                lse,
                out,
                schedule,
                softmax_scale=ctx.softmax_scale,
            )
        dk = dk_workspace.index_select(0, workspace_positions)
        dv = dv_workspace.index_select(0, workspace_positions)
        return dq, dk, dv, None, None, None


class _MSAFlatAttention(nn.Module):
    """MiniMax M3's model-private flat MSA training Adapter."""

    def __init__(self, softmax_scale: float) -> None:
        super().__init__()
        if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
            raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}.")
        self.softmax_scale = float(softmax_scale)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        *,
        layout: _MSAPackedLayout,
    ) -> torch.Tensor:
        """Run flat sparse prefill with a custom training backward.

        Args:
            q: Post-RoPE compact queries [T,64,128] in contiguous,
                16-byte-aligned BF16 CUDA storage.
            k: Post-RoPE compact keys [T,4,128] in contiguous,
                16-byte-aligned BF16 CUDA storage.
            v: Compact values [T,4,128] in contiguous, 16-byte-aligned BF16 CUDA storage.
            q2k: Canonical document-local support [4,T,16] in contiguous,
                16-byte-aligned int32 CUDA storage.
            layout: Opaque packed layout for the same compact token axis.

        Returns:
            Compact MSA output [T,64,128] BF16 CUDA.

        Raises:
            UnavailableError: If the optional MSA dependency is absent.
            TypeError: If layout is not model-owned packed metadata.
            ValueError: If an input violates the fixed launch contract.
            NotImplementedError: If the CUDA device is not SM100 or deterministic
                algorithms are enabled for this atomic-accumulation backward.
        """
        if not isinstance(layout, _MSAPackedLayout):
            raise TypeError(f"layout must be an _MSAPackedLayout, got {type(layout).__name__}.")
        if torch.are_deterministic_algorithms_enabled():
            raise NotImplementedError(
                "MiniMax M3 MSA backward uses global FP32 atomic accumulation and is not bitwise deterministic; "
                "disable torch deterministic algorithms or use sparse_attn='generic'."
            )
        metadata = layout.launch_metadata()
        _validate_flat_msa_inputs(q, k, v, q2k, metadata, softmax_scale=self.softmax_scale)
        return _MSASparseAttentionFunction.apply(q, k, v, q2k, metadata, self.softmax_scale)


_MSA_CACHE_ARGUMENTS = ("past_key_values", "cache_position", "page_table", "seqused_k", "prefix_cache")
_MSA_CROSS_ATTENTION_ARGUMENTS = ("encoder_hidden_states", "key_value_states")


def _reject_unsupported_msa_configuration(backend: BackendConfig) -> None:
    """Reject backend selections that cannot change after MSA construction."""
    if backend.te_fp8 is not None:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports BF16 projection only; set backend.te_fp8=None or use sparse_attn='generic'."
        )
    if backend.rope_fusion:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports rope_fusion=False only: the fused BSHD rotary path uses batch row 0's "
            "positions for every row (position_ids_to_freqs_cis), which corrupts packed per-document positions. "
            "Set backend.rope_fusion=False."
        )


def _msa_cp_enabled(owner: Any) -> bool:
    """Return whether context parallelism is active for *owner*.

    ``apply_cp`` sets ``_cp_enabled`` on the wrapper and the text module, and it
    runs only when the CP mesh has size > 1, so the flag is an exact stand-in for
    ``cp_size > 1``. Sparse attention layers instead receive the submesh itself
    through ``setup_cp_attention``. ``attn_kwargs['cp_size']`` is deliberately not
    consulted: it is injected only on the THD/TE batch paths, and MSA is BSHD-only.

    Args:
        owner: The module whose forward is applying the MSA runtime rules.

    Returns:
        True when this forward would run under context parallelism.
    """
    if getattr(owner, "_cp_enabled", False):
        return True
    cp_mesh = getattr(owner, "_cp_mesh", None)
    return cp_mesh is not None and cp_mesh.size() > 1


def _reject_unsupported_msa_runtime(attn_kwargs: Mapping[str, Any], *, cp_enabled: bool = False) -> None:
    """Reject runtime modes outside the first MiniMax M3 MSA delivery boundary.

    This is the single list of MSA runtime rules. The public causal-LM forward,
    the text-model forward, and the sparse attention forward all call it, so a
    direct caller of any seam is rejected by the same rules before document
    metadata is built or the optional dependency is imported. Structural gates
    for pipeline stages and MTP modules stay at the seams that own those modules.

    Args:
        attn_kwargs: Runtime attention metadata of one forward. Tensor values
            are tested for presence only; their layouts are not inspected.
        cp_enabled: Whether this forward runs under context parallelism; supply
            it from :func:`_msa_cp_enabled`. Defaults to False so that a direct
            caller with no CP state is not rejected.
    Returns:
        None.

    Raises:
        NotImplementedError: If context parallelism is active, or the metadata
            selects THD, a KV cache, paged KV, non-causal or windowed attention,
            cross-attention, or CUDA graph capture.
    """
    if cp_enabled:
        raise NotImplementedError(
            "MiniMax M3 MSA requires cp_size=1; disable context parallelism or set backend.sparse_attn='generic'."
        )
    qkv_format = attn_kwargs.get("qkv_format", "bshd")
    if qkv_format != "bshd":
        raise NotImplementedError(
            "MiniMax M3 MSA sparse attention supports BSHD (qkv_format='bshd') only; "
            f"got {qkv_format!r}. Set backend.sparse_attn='generic' for THD."
        )
    if attn_kwargs.get("use_cache", False):
        raise NotImplementedError("MiniMax M3 MSA supports cache-free prefill training only; set use_cache=False.")
    for cache_argument in _MSA_CACHE_ARGUMENTS:
        if attn_kwargs.get(cache_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA supports cache-free flat prefill only; "
                f"got non-None {cache_argument}. Remove cache metadata or use sparse_attn='generic'."
            )
    if attn_kwargs.get("is_causal", True) is not True:
        raise NotImplementedError("MiniMax M3 MSA first supports causal self-attention only; set is_causal=True.")
    window_size = attn_kwargs.get("window_size", (-1, 0))
    full_causal_window = window_size is None or (isinstance(window_size, int) and window_size == -1)
    if isinstance(window_size, (tuple, list)):
        full_causal_window = tuple(window_size) == (-1, 0)
    if not full_causal_window:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports full causal attention only; "
            f"got window_size={window_size!r}. Disable the sliding window."
        )
    for cross_attention_argument in _MSA_CROSS_ATTENTION_ARGUMENTS:
        if attn_kwargs.get(cross_attention_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA first supports causal self-attention only; "
                f"got non-None {cross_attention_argument}. Use sparse_attn='generic' for cross-attention."
            )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "MiniMax M3 MSA does not support CUDA graph capture in the first delivery boundary; "
            "run outside capture or use sparse_attn='generic'."
        )
