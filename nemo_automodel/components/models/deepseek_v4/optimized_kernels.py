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

"""Optional DeepSeek V4 optimized kernel dispatch.

The torch implementations below are kept as the numerical reference.  Optional
TileLang-backed paths are sourced from:

* Sinkhorn: imported from DeepSeek TileKernels
  ``tile_kernels.modeling.mhc.ops.sinkhorn_normalize``.  No TileKernels source
  is vendored in AutoModel.  Upstream source:
  https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/modeling/mhc/ops/sinkhorn.py
  Upstream license: MIT, copyright 2026 DeepSeek.
* Sparse attention and indexer: vendored/adapted Miles DeepSeek V4 ops in
  ``nemo_automodel.components.models.deepseek_v4.kernels``.  Upstream source:
  https://github.com/yueming-yuan/miles/tree/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops
  Upstream license: Apache-2.0, copyright 2025 Zhipu AI.  See
  ``nemo_automodel/components/models/deepseek_v4/kernels/__init__.py`` for
  the per-file attribution.

Those packages are imported lazily so environments without TileLang still
import the model and use the existing torch path.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from nemo_automodel.shared.import_utils import safe_import_from

Dsv4SparseAttentionBackend = Literal["torch", "sparse_torch", "tilelang", "auto"]
Dsv4IndexerBackend = Literal["torch", "tilelang", "auto"]
Dsv4SinkhornBackend = Literal["torch", "tilelang", "auto"]

_OPTIONAL_IMPORTS: dict[tuple[str, str], tuple[bool, Any]] = {}


def _lazy_import_from(module: str, symbol: str, *, msg: str) -> tuple[bool, Any]:
    key = (module, symbol)
    if key not in _OPTIONAL_IMPORTS:
        _OPTIONAL_IMPORTS[key] = safe_import_from(module, symbol, msg=msg)
    return _OPTIONAL_IMPORTS[key]


def _load_tile_kernels_sinkhorn() -> tuple[bool, Any, bool, Any, bool, Any]:
    has_sinkhorn, sinkhorn = _lazy_import_from(
        "tile_kernels.modeling.mhc.ops",
        "sinkhorn_normalize",
        msg="TileKernels sinkhorn is unavailable. Install tile_kernels and tilelang to use backend.attn='tilelang'.",
    )
    has_fwd, fwd = _lazy_import_from(
        "tile_kernels.mhc.sinkhorn_kernel",
        "_mhc_sinkhorn_fwd",
        msg="TileKernels low-level sinkhorn forward kernel is unavailable.",
    )
    has_bwd, bwd = _lazy_import_from(
        "tile_kernels.mhc.sinkhorn_kernel",
        "_mhc_sinkhorn_bwd",
        msg="TileKernels low-level sinkhorn backward kernel is unavailable.",
    )
    return has_sinkhorn, sinkhorn, has_fwd, fwd, has_bwd, bwd


def _load_miles_sparse_attention() -> tuple[bool, Any, bool, Any]:
    has_sparse_attn, sparse_attn = _lazy_import_from(
        "nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention",
        "sparse_attn_tilelang",
        msg="Vendored Miles DeepSeek V4 sparse attention is unavailable. Install tilelang to use backend.attn='tilelang'.",
    )
    has_chunked, sparse_attn_chunked = _lazy_import_from(
        "nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention",
        "sparse_attn_tilelang_head_chunked",
        msg="Vendored Miles DeepSeek V4 chunked sparse attention is unavailable. Install tilelang to use "
        "backend.attn='tilelang'.",
    )
    return has_sparse_attn, sparse_attn, has_chunked, sparse_attn_chunked


def _load_miles_indexer() -> tuple[bool, Any, bool, Any, bool, Any]:
    has_indexer, batched_indexer_fwd = _lazy_import_from(
        "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer_fwd",
        "batched_indexer_fwd",
        msg="Vendored Miles DeepSeek V4 indexer is unavailable. Install tilelang to use backend.attn='tilelang'.",
    )
    has_cu_seqlens, make_causal_cu_seqlens = _lazy_import_from(
        "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer_fwd",
        "_make_causal_cu_seqlens",
        msg="Vendored Miles DeepSeek V4 indexer cu-seqlens helper is unavailable.",
    )
    has_indexer_autograd, v4_lighting_indexer = _lazy_import_from(
        "nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer",
        "v4_lighting_indexer",
        msg="Vendored Miles DeepSeek V4 autograd indexer is unavailable.",
    )
    return (
        has_indexer,
        batched_indexer_fwd,
        has_cu_seqlens,
        make_causal_cu_seqlens,
        has_indexer_autograd,
        v4_lighting_indexer,
    )


def is_dsv4_kernel_available(name: Literal["sinkhorn", "sparse_attn", "indexer"]) -> bool:
    """Return whether the optional TileLang kernel package for ``name`` is importable."""
    if name == "sinkhorn":
        has_sinkhorn, _, has_fwd, _, has_bwd, _ = _load_tile_kernels_sinkhorn()
        return has_sinkhorn and has_fwd and has_bwd
    if name == "sparse_attn":
        has_sparse_attn, _, _, _ = _load_miles_sparse_attention()
        return has_sparse_attn
    if name == "indexer":
        has_indexer, _, has_cu_seqlens, _, has_indexer_autograd, _ = _load_miles_indexer()
        return has_indexer and has_cu_seqlens and has_indexer_autograd
    raise ValueError(f"Unknown DeepSeek V4 kernel name: {name}")


def _all_cuda(*tensors: torch.Tensor) -> bool:
    return all(tensor.is_cuda for tensor in tensors)


def _tilelang_inputs_ready(tensors: tuple[torch.Tensor, ...], *, require_bf16: bool = False) -> bool:
    if not _all_cuda(*tensors):
        return False
    return not require_bf16 or all(tensor.dtype == torch.bfloat16 for tensor in tensors)


def _should_use_tilelang(
    backend: str,
    *,
    available: bool,
    kernel_name: str,
    tensors: tuple[torch.Tensor, ...],
    require_bf16: bool = False,
) -> bool:
    if backend == "torch" or backend == "sparse_torch":
        return False

    can_run = available and _all_cuda(*tensors)
    if require_bf16:
        can_run = can_run and all(tensor.dtype == torch.bfloat16 for tensor in tensors)

    if backend == "tilelang" and not can_run:
        requirement = "CUDA bfloat16 tensors" if require_bf16 else "CUDA tensors"
        raise RuntimeError(
            f"dsv4 {kernel_name} TileLang backend was requested, but the optional kernel is unavailable "
            f"or inputs do not satisfy {requirement}."
        )
    return backend == "tilelang" or (backend == "auto" and can_run)


def sinkhorn_normalize_torch(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Torch reference for TileKernels MHC Sinkhorn normalization."""
    x = x.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


class _Dsv4TileKernelsSinkhorn(torch.autograd.Function):
    """TileKernels Sinkhorn wrapper that accepts non-contiguous backward gradients.

    The upstream high-level wrapper launches the backward kernel with
    ``grad_output`` as-is. DSV4 consumes HC combinations through transposed
    matmul sites, so autograd can provide a transposed gradient layout. The
    low-level TileKernels backward kernel requires contiguous row-major inputs.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        flat_x = x.contiguous().view(-1, *x.shape[-2:])
        hidden_size = flat_x.shape[1]
        flat_output = torch.empty_like(flat_x)
        _, _, _, fwd_kernel_factory, _, bwd_kernel_factory = _load_tile_kernels_sinkhorn()
        fwd_kernel = fwd_kernel_factory(hidden_size, 1, repeat, eps)
        bwd_kernel = bwd_kernel_factory(hidden_size, 32, repeat, eps)
        fwd_kernel(flat_x, flat_output)
        ctx.save_for_backward(flat_x)
        ctx.bwd_kernel = bwd_kernel
        ctx.input_shape = x.shape
        return flat_output.view_as(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (flat_x,) = ctx.saved_tensors
        flat_grad_output = grad_output.contiguous().view_as(flat_x)
        flat_grad_input = torch.empty_like(flat_x)
        ctx.bwd_kernel(flat_grad_output, flat_x, flat_grad_input)
        return flat_grad_input.view(ctx.input_shape), None, None


def _tile_kernels_sinkhorn_contiguous_grad(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    has_sinkhorn, sinkhorn, has_fwd, _, has_bwd, _ = _load_tile_kernels_sinkhorn()
    if has_fwd and has_bwd:
        return _Dsv4TileKernelsSinkhorn.apply(x, repeat, eps)
    if has_sinkhorn:
        return sinkhorn(x.contiguous(), repeat=repeat, eps=eps)
    raise RuntimeError("TileKernels sinkhorn is unavailable")


def dsv4_sinkhorn_normalize(
    x: torch.Tensor,
    *,
    backend: Dsv4SinkhornBackend,
    repeat: int,
    eps: float,
) -> torch.Tensor:
    """Normalize HyperConnection combination logits with torch or TileKernels."""
    if backend == "torch":
        return sinkhorn_normalize_torch(x, repeat=repeat, eps=eps)

    inputs_ready = _tilelang_inputs_ready((x,))
    available = is_dsv4_kernel_available("sinkhorn") if inputs_ready else False
    if _should_use_tilelang(
        backend,
        available=available,
        kernel_name="sinkhorn",
        tensors=(x,),
    ):
        return _tile_kernels_sinkhorn_contiguous_grad(x, repeat=repeat, eps=eps)
    return sinkhorn_normalize_torch(x, repeat=repeat, eps=eps)


def build_dsv4_sparse_topk_indices(
    *,
    batch_size: int,
    seq_len: int,
    key_len: int,
    window_size: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    compress_ratio: int = 0,
    compressed_topk: torch.Tensor | None = None,
    n_pooled: int = 0,
    vanilla_key_len: int | None = None,
    q_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build Miles-style top-k key indices for DSV4 local-window + compressed KV attention."""
    vanilla_key_len = seq_len if vanilla_key_len is None else vanilla_key_len
    window = min(vanilla_key_len, window_size)
    if q_positions is None:
        q_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    else:
        if q_positions.dim() == 1:
            q_positions = q_positions.unsqueeze(0)
        q_pos = q_positions.to(device=device, dtype=torch.long)
        if q_pos.shape[0] == 1 and batch_size > 1:
            q_pos = q_pos.expand(batch_size, -1)
    offsets = torch.arange(window, device=device, dtype=torch.long)
    k_pos = (q_pos.unsqueeze(-1) - window_size + 1).clamp(min=0) + offsets.view(1, 1, window)
    topk = torch.where(k_pos > q_pos.unsqueeze(-1), torch.full_like(k_pos, -1), k_pos)

    if n_pooled > 0:
        if compressed_topk is not None:
            compressed = torch.where(
                compressed_topk >= 0,
                compressed_topk + vanilla_key_len,
                torch.full_like(compressed_topk, -1),
            )
        else:
            pooled_pos = torch.arange(n_pooled, device=device, dtype=torch.long).view(1, 1, n_pooled)
            threshold = ((q_pos + 1) // compress_ratio).unsqueeze(-1)
            compressed = torch.where(
                pooled_pos < threshold,
                pooled_pos + vanilla_key_len,
                torch.full_like(pooled_pos, -1),
            )
        topk = torch.cat([topk, compressed], dim=-1)

    topk = torch.where((topk >= 0) & (topk < key_len), topk, torch.full_like(topk, -1))

    if attention_mask is not None:
        if attention_mask.dim() != 4:
            raise ValueError(f"Expected 4D additive attention mask, got rank {attention_mask.dim()}")
        safe_topk = topk.clamp(min=0, max=key_len - 1)
        mask_values = torch.gather(attention_mask[:, 0, :, :key_len], dim=-1, index=safe_topk)
        topk = torch.where((topk < 0) | (mask_values < 0), torch.full_like(topk, -1), topk)

    return topk


def sparse_attention_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Miles sparse MQA torch reference.

    Args:
        q: Query tensor with shape ``[B, S, H, D]``.
        kv: Single-head KV tensor with shape ``[B, K, D]``.
        sinks: Per-head attention sink logits with shape ``[H]``.
        topk_idxs: Key indices with shape ``[B, S, K_top]``; ``-1`` masks an entry.
        sm_scale: Attention scaling factor.
    """
    q_float = q.float()
    kv_float = kv.float()
    batch, _, heads, _ = q.shape
    key_len = kv.shape[1]
    valid = (topk_idxs >= 0) & (topk_idxs < key_len)
    safe_idxs = topk_idxs.clamp(min=0, max=max(key_len - 1, 0))
    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1)
    kv_gathered = kv_float[batch_idx, safe_idxs]

    scores = torch.einsum("bshd,bskd->bshk", q_float, kv_gathered) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))

    numerator = torch.einsum("bshk,bskd->bshd", exp_scores, kv_gathered)
    denominator = exp_scores.sum(dim=-1) + torch.exp(sinks.float().view(1, 1, heads) - scores_max)
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)


def dense_attention_topk_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Dense torch oracle for the Miles top-k sparse-attention contract."""
    batch, seq_len, heads, _ = q.shape
    key_len = kv.shape[1]
    topk_len = topk_idxs.shape[-1]
    attn_mask = torch.zeros(batch, seq_len, key_len, dtype=torch.bool, device=q.device)
    valid = (topk_idxs >= 0) & (topk_idxs < key_len)
    safe_topk = topk_idxs.clamp(min=0, max=max(key_len - 1, 0))
    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1).expand(batch, seq_len, topk_len)
    seq_idx = torch.arange(seq_len, device=q.device).view(1, seq_len, 1).expand(batch, seq_len, topk_len)
    attn_mask[batch_idx[valid], seq_idx[valid], safe_topk[valid].long()] = True

    scores = torch.einsum("bshd,bkd->bshk", q.float(), kv.float()) * sm_scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))
    numerator = torch.einsum("bshk,bkd->bshd", exp_scores, kv.float())
    denominator = exp_scores.sum(dim=-1) + torch.exp(sinks.float().view(1, 1, heads) - scores_max)
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)


def dsv4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
    *,
    backend: Dsv4SparseAttentionBackend,
) -> torch.Tensor:
    """Run DSV4 sparse attention through Miles TileLang kernels or torch fallback."""
    if backend == "torch" or backend == "sparse_torch":
        return sparse_attention_torch(q, kv, sinks, topk_idxs.long(), sm_scale)

    inputs_ready = _tilelang_inputs_ready((q, kv), require_bf16=True)
    has_sparse_attn, sparse_attn_tilelang, has_chunked, sparse_attn_tilelang_head_chunked = (
        _load_miles_sparse_attention() if inputs_ready else (False, None, False, None)
    )
    use_tilelang = _should_use_tilelang(
        backend,
        available=has_sparse_attn,
        kernel_name="sparse attention",
        tensors=(q, kv),
        require_bf16=True,
    )
    if use_tilelang:
        q = q.contiguous()
        kv = kv.contiguous()
        sinks = sinks.float().contiguous()
        topk_idxs = topk_idxs.to(torch.int32).contiguous()
        original_heads = q.shape[2]
        if original_heads < 16:
            head_pad = 16 - original_heads
            q = torch.cat([q, q.new_zeros(*q.shape[:2], head_pad, q.shape[3])], dim=2).contiguous()
            sinks = torch.cat([sinks, sinks.new_zeros(head_pad)], dim=0).contiguous()

        # Miles runs this kernel under tensor parallelism, so the kernel sees a
        # small local head count. AutoModel's DSV4 recipe currently uses TP=1,
        # which would launch a single H=64, D=512 kernel with excessive shared
        # memory/register pressure. Chunking heads preserves the same TileLang
        # fwd/bwd kernels and lets autograd sum the per-chunk KV gradients.
        max_heads_per_kernel = 16 if q.shape[-1] >= 256 else 64
        if q.shape[2] > max_heads_per_kernel:
            if not has_chunked:
                raise RuntimeError("Chunked Miles DeepSeek V4 sparse attention is unavailable")
            output = sparse_attn_tilelang_head_chunked(q, kv, sinks, topk_idxs, max_heads_per_kernel, sm_scale)
        else:
            output = sparse_attn_tilelang(q, kv, sinks, topk_idxs, sm_scale)
        return output[:, :, :original_heads, :]
    return sparse_attention_torch(q, kv, sinks, topk_idxs.long(), sm_scale)


def indexer_scores_torch(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Torch reference for the Miles DSV4 C4 indexer score kernel."""
    scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.relu(scores) * softmax_scale
    return (scores * weights.float().unsqueeze(-1)).sum(dim=2)


def extract_indexer_topk_scores_torch(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Extract top-k score values, masking ``-1`` entries with ``-inf``."""
    valid = (topk_indices >= 0) & (topk_indices < logits.shape[-1])
    safe_indices = topk_indices.clamp(min=0, max=max(logits.shape[-1] - 1, 0)).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe_indices)
    return torch.where(valid, scores, torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device))


def dsv4_indexer_scores(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    *,
    compress_ratio: int,
    softmax_scale: float,
    backend: Dsv4IndexerBackend,
    query_start: int = 0,
    query_total_len: int | None = None,
) -> torch.Tensor:
    """Run DSV4 C4 indexer scores through Miles TileLang kernels or torch fallback."""
    if backend == "torch":
        return indexer_scores_torch(q, pooled_kv, weights, softmax_scale)

    inputs_ready = _tilelang_inputs_ready((q, pooled_kv), require_bf16=True)
    has_indexer, batched_indexer_fwd, has_cu_seqlens, make_causal_cu_seqlens, _, _ = (
        _load_miles_indexer() if inputs_ready else (False, None, False, None, False, None)
    )
    if _should_use_tilelang(
        backend,
        available=has_indexer and has_cu_seqlens,
        kernel_name="indexer",
        tensors=(q, pooled_kv),
        require_bf16=True,
    ):
        seq_len = q.shape[1]
        seq_len_kv = pooled_kv.shape[1]
        query_total_len = seq_len if query_total_len is None else query_total_len
        cu_ks, cu_ke = make_causal_cu_seqlens(query_total_len, seq_len_kv, compress_ratio, q.device)
        if query_start or query_total_len != seq_len:
            cu_ks = cu_ks[query_start : query_start + seq_len]
            cu_ke = cu_ke[query_start : query_start + seq_len]
        return batched_indexer_fwd(
            q.transpose(0, 1).contiguous(),
            pooled_kv.transpose(0, 1).contiguous(),
            (weights * softmax_scale).transpose(0, 1).contiguous(),
            cu_ks,
            cu_ke,
        )
    return indexer_scores_torch(q, pooled_kv, weights, softmax_scale)


def dsv4_indexer_topk_scores(
    q: torch.Tensor,
    pooled_kv: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    compress_ratio: int,
    softmax_scale: float,
    backend: Dsv4IndexerBackend,
) -> torch.Tensor:
    """Run DSV4 C4 top-k indexer scores through Miles autograd kernels or torch fallback."""
    if backend == "torch":
        logits = indexer_scores_torch(q, pooled_kv, weights, softmax_scale)
        return extract_indexer_topk_scores_torch(logits, topk_indices.long())

    inputs_ready = _tilelang_inputs_ready((q, pooled_kv), require_bf16=True)
    _, _, _, _, has_indexer_autograd, v4_lighting_indexer = (
        _load_miles_indexer() if inputs_ready else (False, None, False, None, False, None)
    )
    if _should_use_tilelang(
        backend,
        available=has_indexer_autograd,
        kernel_name="indexer autograd",
        tensors=(q, pooled_kv),
        require_bf16=True,
    ):
        scores, _ = v4_lighting_indexer(
            q.transpose(0, 1).contiguous(),
            pooled_kv.transpose(0, 1).contiguous(),
            (weights * softmax_scale).transpose(0, 1).contiguous(),
            compress_ratio,
            topk_indices.shape[-1],
            topk_indices.to(torch.int32).contiguous(),
        )
        return scores
    logits = indexer_scores_torch(q, pooled_kv, weights, softmax_scale)
    return extract_indexer_topk_scores_torch(logits, topk_indices.long())
