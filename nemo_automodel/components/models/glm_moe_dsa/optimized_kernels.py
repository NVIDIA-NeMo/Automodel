# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Optional GLM-5.2 DSA TileLang kernel dispatch.

The dense torch path in ``layers.py`` (``GlmMoeDsaIndexer`` / ``GlmMoeDsaMLA``)
is the numerical reference and the default. This module provides the optional
TileLang-backed sparse path, gated behind ``backend.attn == "tilelang"``:

* the fused **lighting indexer** (logits + top-k), and
* the gather-top-k **sparse MLA** attention (reads only the selected KV; no
  ``[T, T]`` mask is materialized).

The kernels are vendored from THUDM's slime GLM-5.2 plugin under
``nemo_automodel.components.models.glm_moe_dsa.kernels`` (see that package's
``__init__`` for attribution). They are imported with ``safe_import_from`` so
environments without ``tilelang`` still import the model and use the torch path.

Mirrors the structure of ``deepseek_v4/optimized_kernels.py``.
"""

from __future__ import annotations

from typing import Literal

import torch

from nemo_automodel.components.models.glm_moe_dsa.kernels._tilelang import HAS_TILELANG
from nemo_automodel.shared.import_utils import safe_import_from

# GLM-5.2 only ever resolves to "torch" or "tilelang" (see ``_dsa_kernel_backend``
# in ``layers.py``); "auto" is accepted for parity with the V4 dispatcher.
DsaIndexerBackend = Literal["torch", "tilelang", "auto"]
DsaSparseAttentionBackend = Literal["torch", "tilelang", "auto"]

_HAS_SLIME_INDEXER, _slime_lighting_indexer = safe_import_from(
    "nemo_automodel.components.models.glm_moe_dsa.kernels.indexer",
    "lighting_indexer",
    msg="Vendored slime GLM-5.2 indexer is unavailable. Install tilelang to use backend.attn='tilelang'.",
)
_HAS_SLIME_VARLEN, _slime_generate_varlen_mask_params = safe_import_from(
    "nemo_automodel.components.models.glm_moe_dsa.kernels.indexer",
    "generate_varlen_mask_params",
    msg="Vendored slime GLM-5.2 varlen-mask helper is unavailable. Install tilelang to use backend.attn='tilelang'.",
)
_HAS_SLIME_SPARSE_MLA, _slime_sparse_mla = safe_import_from(
    "nemo_automodel.components.models.glm_moe_dsa.kernels.sparse_mla",
    "SparseMLA",
    msg="Vendored slime GLM-5.2 sparse MLA is unavailable. Install tilelang to use backend.attn='tilelang'.",
)


def is_dsa_kernel_available(name: Literal["indexer", "sparse_attn"]) -> bool:
    """Return whether the optional TileLang kernel package for ``name`` is importable."""
    if name == "indexer":
        return HAS_TILELANG and _HAS_SLIME_INDEXER and _HAS_SLIME_VARLEN
    if name == "sparse_attn":
        return HAS_TILELANG and _HAS_SLIME_SPARSE_MLA
    raise ValueError(f"Unknown GLM-5.2 DSA kernel name: {name}")


def _all_cuda(*tensors: torch.Tensor) -> bool:
    return all(tensor.is_cuda for tensor in tensors)


def should_use_tilelang(
    backend: str,
    *,
    available: bool,
    kernel_name: str,
    tensors: tuple[torch.Tensor, ...],
    require_bf16: bool = False,
) -> bool:
    """Decide whether to run the TileLang kernel; raise if forced but unavailable.

    ``backend="tilelang"`` forces the kernel (and raises a clear error if it cannot run);
    ``backend="auto"`` silently falls back to torch when the kernel is unavailable;
    ``backend="torch"`` always uses the torch reference.
    """
    if backend == "torch":
        return False

    can_run = available and _all_cuda(*tensors)
    if require_bf16:
        can_run = can_run and all(tensor.dtype == torch.bfloat16 for tensor in tensors)

    if backend == "tilelang" and not can_run:
        requirement = "CUDA bfloat16 tensors" if require_bf16 else "CUDA tensors"
        raise RuntimeError(
            f"glm_moe_dsa {kernel_name} TileLang backend was requested, but the optional kernel is "
            f"unavailable or inputs do not satisfy {requirement}. Install tilelang (and run on GPU "
            "with bf16/THD inputs), or use backend.attn in {{te, sdpa}}."
        )
    return backend == "tilelang" or (backend == "auto" and can_run)


def tilelang_indexer_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    index_topk: int,
    query_indices: torch.Tensor | None = None,
    cu_seqlens_padded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused lighting-indexer top-k selection (THD/varlen).

    Args:
        index_q: ``[T, index_n_heads, index_head_dim]`` bf16 (rope already applied).
        index_k: ``[T, index_head_dim]`` bf16 (k_norm + rope already applied).
        head_weights: ``[T, index_n_heads]`` fp32; the caller must fold the index
            ``softmax_scale`` into the weight (the kernel computes ``relu(q·k) * w``
            with no internal scale), i.e. ``weights_proj(x) * index_n_heads**-0.5 *
            index_head_dim**-0.5``.
        cu_seqlens: ``[num_seq + 1]`` cumulative sequence lengths of the packed batch.
        index_topk: number of keys to keep (e.g. ``2048``).
        query_indices: Optional global THD token indices for the local query
            rows. Used by context parallelism when ``index_q`` is sharded but
            ``index_k`` has been all-gathered in global token order.
        cu_seqlens_padded: Optional cumulative lengths in the padded THD token
            layout. CP-packed datasets pad each document to a CP multiple, so
            local query indices address this padded layout rather than the
            compact ``cu_seqlens`` layout.

    Returns:
        ``topk_indices`` ``[T, 1, index_topk]`` int32 (``-1`` for invalid/causal-masked),
        matching the layout the sparse-MLA kernel expects.
    """
    if cu_seqlens_padded is not None and not torch.equal(cu_seqlens_padded, cu_seqlens):
        starts, ends = _generate_padded_varlen_mask_params(cu_seqlens, cu_seqlens_padded)
    else:
        starts, ends = _slime_generate_varlen_mask_params(cu_seqlens)
    if query_indices is not None:
        query_indices = query_indices.flatten().to(device=starts.device, dtype=torch.long)
        starts = starts.index_select(0, query_indices)
        ends = ends.index_select(0, query_indices)
    _, topk_indices = _slime_lighting_indexer(
        index_q,
        index_k,
        head_weights,
        starts.to(torch.int32),
        ends.to(torch.int32),
        index_topk,
    )
    return topk_indices.to(torch.int32).unsqueeze(1).contiguous()


def _generate_padded_varlen_mask_params(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-query key windows for padded THD layout.

    ``cu_seqlens`` stores real document lengths compactly. ``cu_seqlens_padded``
    stores their offsets in the actual flattened token tensor, including CP
    padding between documents. TileLang top-k indices refer to the flattened
    token tensor, so starts/ends must be in the padded coordinate space while
    still excluding CP padding keys for real query tokens.
    """
    cu_seqlens = cu_seqlens.to(torch.int32)
    cu_seqlens_padded = cu_seqlens_padded.to(torch.int32)
    padded_total = int(cu_seqlens_padded[-1].item())
    q_indices = torch.arange(0, padded_total, device=cu_seqlens_padded.device, dtype=torch.int32)
    seq_indices = torch.searchsorted(cu_seqlens_padded, q_indices, right=True) - 1
    seq_indices = seq_indices.clamp(min=0, max=cu_seqlens_padded.numel() - 2)

    starts = cu_seqlens_padded[seq_indices]
    real_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    real_ends = starts + real_lengths[seq_indices]
    ends = torch.minimum(q_indices + 1, real_ends)
    ends = torch.maximum(ends, starts + 1)
    return starts, ends


def tilelang_sparse_attention(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    topk_indices: torch.Tensor,
    w_vc: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Gather-top-k sparse MLA attention on the absorbed latent representation.

    Args:
        q: ``[T, n_heads, kv_lora_rank + qk_rope_head_dim]`` bf16 — the absorbed query
            ``cat([q_nope @ w_kc, q_pe], -1)`` (e.g. ``512 + 64 = 576``).
        kv_latent: ``[T, 1, kv_lora_rank + qk_rope_head_dim]`` bf16 — the latent KV
            ``cat([kv_compressed, k_pe], -1)``.
        topk_indices: ``[T, 1, index_topk]`` int32 (``-1`` sentinel).
        w_vc: ``[n_heads, v_head_dim, kv_lora_rank]`` — the value up-projection used to
            map the latent attention output back to ``v_head_dim``.
        softmax_scale: MLA attention scale ``mscale**2 / sqrt(qk_head_dim)`` (NOT the
            kernel's ``1/sqrt(dim+tail)`` default).

    Returns:
        ``attn_out`` ``[T, n_heads, v_head_dim]`` bf16.
    """
    out, _ = _slime_sparse_mla.apply(q, kv_latent, topk_indices, softmax_scale)
    # out: [T, n_heads, kv_lora_rank] -> map back to v_head_dim via the value up-proj.
    return torch.einsum("thc,hdc->thd", out, w_vc)
