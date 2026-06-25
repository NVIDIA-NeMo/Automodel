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

"""Gemma4-specific context-parallel attention helpers."""

from __future__ import annotations

import contextlib
import logging
import math
import os
from dataclasses import dataclass, replace
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.utils.memory_profile import log_cuda_memory_profile

logger = logging.getLogger(__name__)
_GEMMA4_CP_FLEX_RING_OK_LOGGED = False
_GEMMA4_CP_PROFILE_NEXT_MODULE_ID = 0
_GEMMA4_CP_PACKED_DEBUG_SEEN: set[tuple[Any, ...]] = set()
_GEMMA4_CP_COMPARE_SDPA_SEEN: set[tuple[Any, ...]] = set()


def _patch_fsdp_accumulated_grad_guard() -> None:
    """Guard ``FSDPParam.to_accumulated_grad_if_needed`` against uninitialized params.

    On some torch builds that method reads ``self._unsharded_param`` (the lazily
    set unsharded tensor) without first checking it exists. In FSDP2 post-backward
    under fp32 grad-reduce, frozen / never-unsharded params (e.g. the frozen Gemma4
    vision tower and embeddings) have no ``_unsharded_param`` yet and it raises
    ``AttributeError``. Such params carry no grad to upcast anyway, so wrap the
    method to skip them when uninitialized. No-op once applied / on fixed builds.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return
    orig = FSDPParam.to_accumulated_grad_if_needed
    if getattr(orig, "_gemma4_guarded", False):
        return

    def guarded(self):
        if not hasattr(self, "_unsharded_param"):
            return
        return orig(self)

    guarded._gemma4_guarded = True
    FSDPParam.to_accumulated_grad_if_needed = guarded


_patch_fsdp_accumulated_grad_guard()


@dataclass(frozen=True)
class CPRingAttentionContext:
    """Inputs for Gemma4 manual ring CP attention (built by the run_cp_manual_attention seam)."""

    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cp_mesh: Any
    cp_group: Any
    cp_size: int
    cp_rank: int
    seq_local: int
    seq_full: int
    seq_global_start: int
    attn_mask: Any
    dropout_p: float
    is_causal: bool
    scale: Any
    enable_gqa: bool
    kwargs: dict[str, Any]
    metadata: dict[str, torch.Tensor | None]
    metadata_seq_dims: dict[str, int]


def gemma4_vision_group_ids(mm_token_type_ids: torch.Tensor) -> torch.Tensor:
    """Return per-image-block ids for Gemma4 vision tokens, or -1 for text/padding."""
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))


# Block masks built by create_block_mask are independent of the K/V *values*: they
# depend only on the attention type (sliding window), the local/chunk sequence
# geometry, and the position-level metadata (packing / padding / vision groups).
# Within a single training step the metadata content is fixed by *position*, so for
# a given (layer-type, chunk geometry) every one of E4B's 42 layers builds an
# identical mask -- and create_block_mask costs ~7.6ms/call, dominating short-seq CP
# step time (~66%). We cache on the position scalars only (never on tensor storage
# -- the CUDA allocator recycles addresses, so a data_ptr key could alias distinct
# content and return a stale mask). Correctness across steps is preserved by
# clearing the cache whenever the batch's metadata object changes; the object is
# held as the generation token so its identity can't be recycled while it is live.
_BLOCK_MASK_CACHE: dict = {}
# [data_ptr, held-tensor]: the metadata tensor is re-wrapped into a fresh Python
# object every layer but shares one storage within a step (and the detached
# backward metadata shares it too), so we detect a new batch by its data_ptr.
# The tensor is held so the allocator cannot recycle that address mid-step, which
# would otherwise let a later step's distinct content collide on the same pointer.
_BLOCK_MASK_GEN: list = [None, None]


def _block_mask_set_generation(gen_tensor) -> None:
    """Reset the per-step block-mask cache when a new batch (new metadata) arrives."""
    ptr = None if gen_tensor is None else gen_tensor.data_ptr()
    if ptr != _BLOCK_MASK_GEN[0]:
        _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_GEN[0] = ptr
        _BLOCK_MASK_GEN[1] = gen_tensor  # pin the storage for the duration of the step


def _cached_block_mask(key, build):
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = build()
    if len(_BLOCK_MASK_CACHE) >= 256:  # bound the gen=None (metadata-free) varying-seqlen case
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = mask
    return mask


def _cp_memory_profile_allowed(attention_module: torch.nn.Module) -> bool:
    layers = os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE_CP_LAYERS")
    if not layers:
        return True

    layer_idx = getattr(attention_module, "layer_idx", None)
    profile_id = getattr(attention_module, "_gemma4_cp_profile_id", None)
    allowed = set()
    for item in layers.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            allowed.add(int(item))
        except ValueError:
            continue
    return layer_idx in allowed or profile_id in allowed


def _log_cp_memory(
    attention_module: torch.nn.Module,
    tag: str,
    *,
    include_tensors: bool = False,
) -> None:
    if not _cp_memory_profile_allowed(attention_module):
        return
    layer_idx = getattr(attention_module, "layer_idx", None)
    profile_id = getattr(attention_module, "_gemma4_cp_profile_id", None)
    log_cuda_memory_profile(
        f"gemma4_cp layer={layer_idx} profile_id={profile_id} {tag}",
        logger=logger,
        include_tensors=include_tensors,
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _debug_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _debug_list_allows(value: int, env_name: str) -> bool:
    raw = os.environ.get(env_name)
    if not raw:
        return True
    raw = raw.strip().lower()
    if raw in {"all", "*"}:
        return True
    allowed = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            allowed.add(int(item))
        except ValueError:
            continue
    return value in allowed


def _debug_compare_allowed(attention_module: torch.nn.Module, ctx: Any, *, kv_global_start: int) -> bool:
    if not _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_SDPA"):
        return False
    rank = _debug_rank()
    if not _debug_list_allows(rank, "NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_SDPA_RANKS"):
        return False
    layer_idx = getattr(attention_module, "layer_idx", None)
    profile_id = getattr(attention_module, "_gemma4_cp_profile_id", -1)
    layer_or_profile = profile_id if layer_idx is None else layer_idx
    if not _debug_list_allows(int(layer_or_profile), "NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_SDPA_LAYERS"):
        return False
    seen_key = (rank, profile_id, ctx.cp_rank, int(ctx.seq_global_start), int(kv_global_start), int(ctx.seq_local))
    if seen_key in _GEMMA4_CP_COMPARE_SDPA_SEEN:
        return False
    _GEMMA4_CP_COMPARE_SDPA_SEEN.add(seen_key)
    return True


def _packed_id_segments(seq_ids: torch.Tensor, *, global_start: int, limit: int = 12) -> str:
    ids = seq_ids.detach().to("cpu").tolist()
    if not ids:
        return "[]"
    segments = []
    start = 0
    current = ids[0]
    for idx, value in enumerate(ids[1:], start=1):
        if value == current:
            continue
        if current > 0:
            segments.append((current, global_start + start, global_start + idx, idx - start))
        start = idx
        current = value
    if current > 0:
        segments.append((current, global_start + start, global_start + len(ids), len(ids) - start))

    rendered = [
        f"id={doc_id} global=[{seg_start},{seg_end}) len={length}"
        for doc_id, seg_start, seg_end, length in segments[:limit]
    ]
    suffix = "" if len(segments) <= limit else f"; ... total_segments={len(segments)}"
    return "[" + "; ".join(rendered) + suffix + "]"


def _boundary_allowed_mask(
    attention_module: torch.nn.Module,
    ctx: Any,
    *,
    packed_seq_ids_q: torch.Tensor,
    packed_seq_ids_kv: torch.Tensor,
    padding_mask_q: torch.Tensor | None,
    padding_mask_kv: torch.Tensor | None,
    vision_group_ids_q: torch.Tensor | None,
    vision_group_ids_kv: torch.Tensor | None,
    kv_global_start: int,
    use_vision_bidirectional: bool,
) -> torch.Tensor:
    q_global = torch.arange(ctx.seq_global_start, ctx.seq_global_start + ctx.seq_local, device=ctx.query.device)
    kv_global = torch.arange(kv_global_start, kv_global_start + packed_seq_ids_kv.shape[1], device=ctx.query.device)
    if ctx.is_causal:
        allowed = kv_global.view(1, 1, -1) <= q_global.view(1, -1, 1)
    else:
        allowed = torch.ones(
            (packed_seq_ids_q.shape[0], ctx.seq_local, packed_seq_ids_kv.shape[1]),
            dtype=torch.bool,
            device=ctx.query.device,
        )

    sliding_window = getattr(attention_module, "sliding_window", None)
    if sliding_window is not None:
        allowed = allowed & ((q_global.view(1, -1, 1) - kv_global.view(1, 1, -1)) < sliding_window)
    if use_vision_bidirectional and vision_group_ids_q is not None and vision_group_ids_kv is not None:
        same_vision_group = (vision_group_ids_q[:, :, None] == vision_group_ids_kv[:, None, :]) & (
            vision_group_ids_q[:, :, None] >= 0
        )
        allowed = allowed | same_vision_group

    same_doc = packed_seq_ids_q[:, :, None] == packed_seq_ids_kv[:, None, :]
    allowed = allowed & same_doc & (packed_seq_ids_q[:, :, None] > 0)
    allowed = torch.where(packed_seq_ids_q[:, :, None] <= 0, torch.zeros_like(allowed), allowed)
    if padding_mask_kv is not None:
        allowed = allowed & ~padding_mask_kv[:, None, :]
    if padding_mask_q is not None:
        allowed = torch.where(padding_mask_q[:, :, None], torch.zeros_like(allowed), allowed)
    return allowed


def _debug_packed_boundaries(
    attention_module: torch.nn.Module,
    ctx: Any,
    *,
    key_chunk: torch.Tensor,
    packed_seq_ids_q: torch.Tensor | None,
    packed_seq_ids_kv: torch.Tensor | None,
    padding_mask_q: torch.Tensor | None,
    padding_mask_kv: torch.Tensor | None,
    vision_group_ids_q: torch.Tensor | None,
    vision_group_ids_kv: torch.Tensor | None,
    kv_global_start: int,
    use_vision_bidirectional: bool,
) -> None:
    if not _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_PACKED"):
        return

    rank = _debug_rank()
    if not _debug_list_allows(rank, "NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_PACKED_RANKS"):
        return
    layer_idx = getattr(attention_module, "layer_idx", None)
    profile_id = getattr(attention_module, "_gemma4_cp_profile_id", -1)
    if layer_idx is not None:
        if not _debug_list_allows(int(layer_idx), "NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_PACKED_LAYERS"):
            return
    elif not _debug_list_allows(int(profile_id), "NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_PACKED_LAYERS"):
        return

    if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_PACKED_ONCE"):
        seen_key = (rank, profile_id, ctx.cp_rank, kv_global_start, ctx.seq_local, key_chunk.shape[2])
        if seen_key in _GEMMA4_CP_PACKED_DEBUG_SEEN:
            return
        _GEMMA4_CP_PACKED_DEBUG_SEEN.add(seen_key)

    prefix = (
        "[GEMMA4_CP_PACKED_DEBUG] "
        f"rank={rank} cp_rank={ctx.cp_rank} layer={layer_idx} profile_id={profile_id} "
        f"q_global=[{ctx.seq_global_start},{ctx.seq_global_start + ctx.seq_local}) "
        f"kv_global=[{kv_global_start},{kv_global_start + key_chunk.shape[2]}) "
        f"q_shape={tuple(ctx.query.shape)} kv_shape={tuple(key_chunk.shape)}"
    )

    if packed_seq_ids_q is None or packed_seq_ids_kv is None:
        message = f"{prefix} missing packed ids: q_present={packed_seq_ids_q is not None} kv_present={packed_seq_ids_kv is not None}"
        print(message, flush=True)
        if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_DEBUG_REQUIRE_PACKED"):
            raise RuntimeError(message)
        return

    q_ids = packed_seq_ids_q.contiguous()
    kv_ids = packed_seq_ids_kv.contiguous()
    allowed = _boundary_allowed_mask(
        attention_module,
        ctx,
        packed_seq_ids_q=q_ids,
        packed_seq_ids_kv=kv_ids,
        padding_mask_q=padding_mask_q,
        padding_mask_kv=padding_mask_kv,
        vision_group_ids_q=vision_group_ids_q,
        vision_group_ids_kv=vision_group_ids_kv,
        kv_global_start=kv_global_start,
        use_vision_bidirectional=use_vision_bidirectional,
    )
    valid_q = q_ids > 0 if padding_mask_q is None else (q_ids > 0) & ~padding_mask_q
    valid_kv = kv_ids > 0 if padding_mask_kv is None else (kv_ids > 0) & ~padding_mask_kv
    cross_doc = (q_ids[:, :, None] != kv_ids[:, None, :]) & valid_q[:, :, None] & valid_kv[:, None, :]
    cross_doc_allowed = (allowed & cross_doc).sum()

    q_unique = torch.unique(q_ids[q_ids > 0]).numel()
    kv_unique = torch.unique(kv_ids[kv_ids > 0]).numel()
    padding_q = int((~valid_q).sum().item())
    padding_kv = int((~valid_kv).sum().item())
    print(
        f"{prefix} q_ids_shape={tuple(q_ids.shape)} kv_ids_shape={tuple(kv_ids.shape)} "
        f"q_docs={int(q_unique)} kv_docs={int(kv_unique)} q_padding={padding_q} kv_padding={padding_kv} "
        f"allowed_pairs={int(allowed.sum().item())} cross_doc_allowed={int(cross_doc_allowed.item())}",
        flush=True,
    )
    print(
        f"{prefix} q_segments={_packed_id_segments(q_ids[0], global_start=ctx.seq_global_start)}",
        flush=True,
    )
    print(
        f"{prefix} kv_segments={_packed_id_segments(kv_ids[0], global_start=kv_global_start)}",
        flush=True,
    )
    if cross_doc_allowed.item() != 0:
        raise RuntimeError(
            f"{prefix} packed sequence boundary audit failed: cross_doc_allowed={cross_doc_allowed.item()}"
        )
    print(f"{prefix} packed sequence boundary audit PASS", flush=True)


def _compiled_flex_attention(attention_module: torch.nn.Module):
    compiled = getattr(attention_module, "_gemma4_cp_compiled_flex_attn", None)
    if compiled is None:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)
        attention_module._gemma4_cp_compiled_flex_attn = compiled
    return compiled


@contextlib.contextmanager
def _duck_shape_disabled():
    """Locally disable flex duck-shape specialization for the wrapped flex call.

    With variable-length (unpacked) batches the compiled flex kernel otherwise
    guards on incidental dim-equalities (e.g. ``block_mask.kv_indices.size()[2] ==
    key.size()[1]``) and recompiles on every new sequence length, collapsing
    throughput to ~warmup speed. ``use_duck_shape`` is read by dynamo at (re)trace
    time -- which happens inside the flex call -- so scoping it to the call window
    is sufficient and, unlike setting it once at compile time, does not leave the
    process-global ``torch.fx`` config mutated for unrelated ``torch.compile`` users.
    """
    from torch.fx.experimental import _config as _fx_config

    prev = _fx_config.use_duck_shape
    _fx_config.use_duck_shape = False
    try:
        yield
    finally:
        _fx_config.use_duck_shape = prev


def _base_gemma4_cp_mask(attention_module: torch.nn.Module, ctx: Any, q_idx, kv_idx, kv_global_start: int = 0):
    q_global_idx = q_idx + ctx.seq_global_start
    kv_global_idx = kv_idx + kv_global_start
    if not ctx.is_causal:
        allowed = torch.ones_like(q_global_idx >= kv_global_idx)
    else:
        allowed = kv_global_idx <= q_global_idx

    sliding_window = getattr(attention_module, "sliding_window", None)
    if sliding_window is not None:
        allowed = allowed & ((q_global_idx - kv_global_idx) < sliding_window)
    return allowed


def _metadata_like(metadata: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {name: torch.empty_like(value) if value is not None else None for name, value in metadata.items()}


def _detach_metadata(metadata: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {name: value.detach().contiguous() if value is not None else None for name, value in metadata.items()}


def _repeat_kv_for_gqa(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to query-head count, matching HF SDPA with an explicit mask."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _ring_exchange(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    cp_group: Any,
    cp_rank: int,
    cp_size: int,
) -> None:
    if not tensors:
        return

    ranks = torch.distributed.get_process_group_ranks(cp_group)
    send_dst = ranks[(cp_rank + 1) % cp_size]
    recv_src = ranks[(cp_rank - 1) % cp_size]

    send_ops = [
        torch.distributed.P2POp(torch.distributed.isend, send_tensor.contiguous(), send_dst, cp_group)
        for send_tensor, _ in tensors
    ]
    recv_ops = [
        torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, recv_src, cp_group) for _, recv_tensor in tensors
    ]
    ops = send_ops + recv_ops if cp_rank % 2 == 0 else recv_ops + send_ops
    for req in torch.distributed.batch_isend_irecv(ops):
        req.wait()


def _direct_exchange(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    cp_group: Any,
    cp_rank: int,
    send_cp_rank: int,
    recv_cp_rank: int,
) -> None:
    if not tensors:
        return

    ranks = torch.distributed.get_process_group_ranks(cp_group)
    send_dst = ranks[send_cp_rank]
    recv_src = ranks[recv_cp_rank]

    send_ops = [
        torch.distributed.P2POp(torch.distributed.isend, send_tensor.contiguous(), send_dst, cp_group)
        for send_tensor, _ in tensors
    ]
    recv_ops = [
        torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, recv_src, cp_group) for _, recv_tensor in tensors
    ]
    ops = send_ops + recv_ops if cp_rank % 2 == 0 else recv_ops + send_ops
    for req in torch.distributed.batch_isend_irecv(ops):
        req.wait()


def _merge_flex_chunk(
    out_acc: torch.Tensor | None,
    lse_acc: torch.Tensor | None,
    out_step: torch.Tensor,
    lse_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out_acc is None or lse_acc is None:
        return out_step, lse_step
    lse_next = torch.logaddexp(lse_acc, lse_step)
    empty_rows = torch.isneginf(lse_next)
    old_scale = torch.where(empty_rows, torch.zeros_like(lse_next), torch.exp(lse_acc - lse_next)).unsqueeze(-1)
    new_scale = torch.where(empty_rows, torch.zeros_like(lse_next), torch.exp(lse_step - lse_next)).unsqueeze(-1)
    return out_acc * old_scale + out_step * new_scale, lse_next


def _zero_masked_attention_rows(
    tensor: torch.Tensor,
    *,
    empty_query_rows: torch.Tensor | None,
    chunk_empty_query_rows: torch.Tensor,
) -> torch.Tensor:
    if empty_query_rows is not None and empty_query_rows.any():
        tensor = tensor.masked_fill(empty_query_rows[:, None, :, None], 0)
    if chunk_empty_query_rows.any():
        tensor = tensor.masked_fill(chunk_empty_query_rows[:, None, :, None], 0)
    return tensor


def _debug_compare_flex_sdpa(
    attention_module: torch.nn.Module,
    ctx: Any,
    *,
    query_for_flex: torch.Tensor,
    key_for_flex: torch.Tensor,
    value_for_flex: torch.Tensor,
    block_mask: Any,
    cp_score_mod: Any,
    dense_allowed_mask: torch.Tensor,
    empty_query_rows: torch.Tensor | None,
    chunk_empty_query_rows: torch.Tensor,
    flex_scale: float | None,
    flex_kwargs: dict[str, Any],
    flex_out: torch.Tensor,
    kv_global_start: int,
) -> None:
    if not _debug_compare_allowed(attention_module, ctx, kv_global_start=kv_global_start):
        return

    rank = _debug_rank()
    layer_idx = getattr(attention_module, "layer_idx", None)
    profile_id = getattr(attention_module, "_gemma4_cp_profile_id", -1)
    prefix = (
        "[GEMMA4_CP_SDPA_COMPARE] "
        f"rank={rank} cp_rank={ctx.cp_rank} layer={layer_idx} profile_id={profile_id} "
        f"q_global=[{ctx.seq_global_start},{ctx.seq_global_start + ctx.seq_local}) "
        f"kv_global=[{kv_global_start},{kv_global_start + key_for_flex.shape[2]})"
    )

    if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_FULL_SDPA"):
        compare_kwargs = {key: value for key, value in flex_kwargs.items() if key != "enable_gqa"}
        compare_kwargs["enable_gqa"] = False
        compare_kwargs["score_mod"] = cp_score_mod
        compare_kwargs["block_mask"] = block_mask
        compare_kwargs["scale"] = flex_scale

        dense_mask = dense_allowed_mask[:, None, :, :]
        compare_grads = _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_SDPA_GRADS")
        with torch.enable_grad():
            q = query_for_flex.detach().contiguous().requires_grad_(compare_grads)
            k = key_for_flex.detach().contiguous().requires_grad_(compare_grads)
            v = value_for_flex.detach().contiguous().requires_grad_(compare_grads)
            ref_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=dense_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=flex_scale,
                enable_gqa=False,
            )
            ref_out = _zero_masked_attention_rows(
                ref_out,
                empty_query_rows=empty_query_rows,
                chunk_empty_query_rows=chunk_empty_query_rows,
            )
            flex_cmp, _ = _compiled_flex_attention(attention_module)(
                q,
                k,
                v,
                return_lse=True,
                **compare_kwargs,
            )
            flex_cmp = _zero_masked_attention_rows(
                flex_cmp,
                empty_query_rows=empty_query_rows,
                chunk_empty_query_rows=chunk_empty_query_rows,
            )
            if compare_grads:
                upstream = torch.randn_like(ref_out)
                ref_grads = torch.autograd.grad(ref_out, (q, k, v), upstream, retain_graph=True)
                flex_grads = torch.autograd.grad(flex_cmp, (q, k, v), upstream)
            else:
                ref_grads = flex_grads = None

        out_diff = (flex_cmp.detach() - ref_out.detach()).float()
        prod_out_diff = (flex_out.detach() - ref_out.detach()).float()
        ref_norm = ref_out.detach().float().norm()
        flex_norm = flex_cmp.detach().float().norm()
        out_max = out_diff.abs().max()
        out_mean = out_diff.abs().mean()
        prod_out_max = prod_out_diff.abs().max()
        math_sdpa_message = ""
        if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_MATH_SDPA"):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                math_out = F.scaled_dot_product_attention(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    attn_mask=dense_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=flex_scale,
                    enable_gqa=False,
                )
            math_out = _zero_masked_attention_rows(
                math_out,
                empty_query_rows=empty_query_rows,
                chunk_empty_query_rows=chunk_empty_query_rows,
            )
            math_flex_diff = (flex_cmp.detach() - math_out.detach()).float()
            math_default_diff = (ref_out.detach() - math_out.detach()).float()
            math_sdpa_message = (
                f" math_sdpa_norm={math_out.detach().float().norm().item():.6e}"
                f" flex_math_max={math_flex_diff.abs().max().item():.6e}"
                f" flex_math_mean={math_flex_diff.abs().mean().item():.6e}"
                f" default_math_max={math_default_diff.abs().max().item():.6e}"
                f" default_math_mean={math_default_diff.abs().mean().item():.6e}"
            )
    else:
        ref_grads = flex_grads = None
        scale = flex_scale if flex_scale is not None else (1.0 / math.sqrt(query_for_flex.shape[-1]))
        block_q = int(os.environ.get("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_MANUAL_BLOCK_Q", "128"))
        abs_sum = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        sq_sum = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        ref_sq_sum = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        flex_sq_sum = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        out_max = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        prod_out_max = torch.zeros((), device=query_for_flex.device, dtype=torch.float32)
        numel = 0
        key_t = key_for_flex.transpose(-2, -1)
        for q_start in range(0, query_for_flex.shape[2], block_q):
            q_end = min(q_start + block_q, query_for_flex.shape[2])
            scores = torch.matmul(query_for_flex[:, :, q_start:q_end, :], key_t) * scale
            scores = scores.masked_fill(~dense_allowed_mask[:, None, q_start:q_end, :], -float("inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value_for_flex.dtype)
            ref_block = torch.matmul(probs, value_for_flex)
            flex_block = flex_out[:, :, q_start:q_end, :]
            if empty_query_rows is not None:
                ref_block = ref_block.masked_fill(empty_query_rows[:, None, q_start:q_end, None], 0)
            if chunk_empty_query_rows.any():
                ref_block = ref_block.masked_fill(chunk_empty_query_rows[:, None, q_start:q_end, None], 0)
            diff = (flex_block.detach() - ref_block.detach()).float()
            abs_diff = diff.abs()
            abs_sum = abs_sum + abs_diff.sum()
            sq_sum = sq_sum + (diff * diff).sum()
            ref_float = ref_block.detach().float()
            flex_float = flex_block.detach().float()
            ref_sq_sum = ref_sq_sum + (ref_float * ref_float).sum()
            flex_sq_sum = flex_sq_sum + (flex_float * flex_float).sum()
            out_max = torch.maximum(out_max, abs_diff.max())
            prod_out_max = out_max
            numel += diff.numel()
        out_mean = abs_sum / max(numel, 1)
        ref_norm = torch.sqrt(ref_sq_sum)
        flex_norm = torch.sqrt(flex_sq_sum)
    allowed_pairs = int(dense_allowed_mask.sum().item())
    empty_rows = int(chunk_empty_query_rows.sum().item())
    empty_query_count = 0 if empty_query_rows is None else int(empty_query_rows.sum().item())
    ref_name = "sdpa" if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_FULL_SDPA") else "manual"
    message = (
        f"{prefix} allowed_pairs={allowed_pairs} empty_rows={empty_rows} "
        f"empty_query_rows={empty_query_count} ref={ref_name} scale={flex_scale} "
        f"q_dtype={query_for_flex.dtype} k_dtype={key_for_flex.dtype} v_dtype={value_for_flex.dtype} "
        f"q_stride={tuple(query_for_flex.stride())} k_stride={tuple(key_for_flex.stride())} "
        f"v_stride={tuple(value_for_flex.stride())} "
        f"q_absmax={query_for_flex.detach().abs().max().item():.6e} "
        f"k_absmax={key_for_flex.detach().abs().max().item():.6e} "
        f"v_absmax={value_for_flex.detach().abs().max().item():.6e} "
        f"ref_norm={ref_norm.item():.6e} "
        f"flex_norm={flex_norm.item():.6e} "
        f"out_max={out_max.item():.6e} out_mean={out_mean.item():.6e} "
        f"prod_out_max={prod_out_max.item():.6e}"
    )
    if _env_flag("NEMO_AUTOMODEL_GEMMA4_CP_COMPARE_FULL_SDPA"):
        message += math_sdpa_message
    if ref_grads is not None and flex_grads is not None:
        grad_diffs = [(fg.detach() - sg.detach()).float() for fg, sg in zip(flex_grads, ref_grads)]
        message += (
            f" q_grad_max={grad_diffs[0].abs().max().item():.6e} q_grad_norm={grad_diffs[0].norm().item():.6e}"
            f" k_grad_max={grad_diffs[1].abs().max().item():.6e} k_grad_norm={grad_diffs[1].norm().item():.6e}"
            f" v_grad_max={grad_diffs[2].abs().max().item():.6e} v_grad_norm={grad_diffs[2].norm().item():.6e}"
        )
    print(message, flush=True)


def _run_gemma4_flex_chunk(
    attention_module: torch.nn.Module,
    ctx: Any,
    *,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    metadata_chunk: dict[str, torch.Tensor | None],
    kv_global_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    query = ctx.query
    orig_head_dim = query.shape[-1]
    padded_head_dim = 1 << (orig_head_dim - 1).bit_length()
    use_small_flex_blocks = padded_head_dim > 256
    flex_block_size = (32, 32) if use_small_flex_blocks else 128

    packed_seq_ids_q = ctx.metadata.get("_packed_seq_ids")
    packed_seq_ids_kv = metadata_chunk.get("_packed_seq_ids")
    padding_mask_q = ctx.metadata.get("padding_mask")
    padding_mask_kv = metadata_chunk.get("padding_mask")
    vision_group_ids_q = ctx.metadata.get("_gemma4_vision_group_ids")
    vision_group_ids_kv = metadata_chunk.get("_gemma4_vision_group_ids")
    if vision_group_ids_q is None and ctx.metadata.get("mm_token_type_ids") is not None:
        vision_group_ids_q = gemma4_vision_group_ids(ctx.metadata["mm_token_type_ids"])
    if vision_group_ids_kv is None and metadata_chunk.get("mm_token_type_ids") is not None:
        vision_group_ids_kv = gemma4_vision_group_ids(metadata_chunk["mm_token_type_ids"])

    sliding_window = getattr(attention_module, "sliding_window", None)
    config_uses_vision_bidir = (
        getattr(getattr(attention_module, "config", None), "use_bidirectional_attention", None) == "vision"
    )
    use_vision_bidirectional = (
        sliding_window is not None
        and config_uses_vision_bidir
        and vision_group_ids_q is not None
        and vision_group_ids_kv is not None
    )
    _debug_packed_boundaries(
        attention_module,
        ctx,
        key_chunk=key_chunk,
        packed_seq_ids_q=packed_seq_ids_q,
        packed_seq_ids_kv=packed_seq_ids_kv,
        padding_mask_q=padding_mask_q,
        padding_mask_kv=padding_mask_kv,
        vision_group_ids_q=vision_group_ids_q,
        vision_group_ids_kv=vision_group_ids_kv,
        kv_global_start=kv_global_start,
        use_vision_bidirectional=use_vision_bidirectional,
    )

    empty_query_rows = None
    if packed_seq_ids_q is not None:
        empty_query_rows = packed_seq_ids_q <= 0
    if padding_mask_q is not None:
        padding_query_rows = padding_mask_q
        empty_query_rows = padding_query_rows if empty_query_rows is None else empty_query_rows | padding_query_rows

    try:
        from torch.nn.attention.flex_attention import create_block_mask

        if (
            use_vision_bidirectional
            or packed_seq_ids_q is not None
            or packed_seq_ids_kv is not None
            or padding_mask_q is not None
            or padding_mask_kv is not None
        ):

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                allowed = _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx, kv_global_start)
                if use_vision_bidirectional:
                    q_group = vision_group_ids_q[batch_idx, q_idx]
                    kv_group = vision_group_ids_kv[batch_idx, kv_idx]
                    same_vision_group = (q_group == kv_group) & (q_group >= 0)
                    allowed = allowed | same_vision_group
                if packed_seq_ids_q is not None and packed_seq_ids_kv is not None:
                    q_pack_id = packed_seq_ids_q[batch_idx, q_idx]
                    kv_pack_id = packed_seq_ids_kv[batch_idx, kv_idx]
                    allowed = allowed & (q_pack_id == kv_pack_id) & (q_pack_id > 0)
                    allowed = torch.where(q_pack_id <= 0, kv_idx == 0, allowed)
                if padding_mask_kv is not None:
                    kv_is_padding = padding_mask_kv[batch_idx, kv_idx]
                    allowed = allowed & ~kv_is_padding
                if padding_mask_q is not None:
                    q_is_padding = padding_mask_q[batch_idx, q_idx]
                    allowed = torch.where(q_is_padding, kv_idx == 0, allowed)
                return allowed

            block_mask_batch = query.shape[0]
        else:

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                return _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx, kv_global_start)

            block_mask_batch = None

        def _build_block_mask():
            return create_block_mask(
                cp_mask,
                B=block_mask_batch,
                H=None,
                Q_LEN=ctx.seq_local,
                KV_LEN=key_chunk.shape[2],
                device=query.device,
                BLOCK_SIZE=flex_block_size,
            )

        # Reset the cache when the batch changes (new metadata object => new step),
        # then key purely on the position scalars. Within a step the metadata content
        # is a function of position, so (layer-type, query/chunk geometry) uniquely
        # identifies the mask -- shared by all 42 same-type layers, both ring chunks,
        # and (since both pin the same captured metadata) forward and backward.
        gen_obj = None
        for _gen_key in ("mm_token_type_ids", "_packed_seq_ids", "padding_mask"):
            _cand = ctx.metadata.get(_gen_key)
            if _cand is not None:
                gen_obj = _cand
                break
        _block_mask_set_generation(gen_obj)
        block_mask_key = (
            getattr(attention_module, "sliding_window", None),
            bool(ctx.is_causal),
            int(ctx.seq_global_start),
            int(kv_global_start),
            int(ctx.seq_local),
            int(key_chunk.shape[2]),
            flex_block_size,
            block_mask_batch,
            bool(use_vision_bidirectional),
            query.device.type,
            query.device.index,
        )
        block_mask = _cached_block_mask(block_mask_key, _build_block_mask)

        batch_idx = torch.arange(query.shape[0], device=query.device).view(-1, 1, 1)
        head_idx = torch.zeros_like(batch_idx)
        q_idx = torch.arange(ctx.seq_local, device=query.device).view(1, -1, 1)
        kv_idx = torch.arange(key_chunk.shape[2], device=query.device).view(1, 1, -1)
        chunk_allowed = cp_mask(batch_idx, head_idx, q_idx, kv_idx)
        if chunk_allowed.shape[0] == 1 and query.shape[0] > 1:
            chunk_allowed = chunk_allowed.expand(query.shape[0], -1, -1)
        chunk_empty_query_rows = ~chunk_allowed.any(dim=-1)

        query_for_flex = query
        key_for_flex = key_chunk
        value_for_flex = value_chunk
        flex_scale = ctx.scale
        if padded_head_dim != orig_head_dim:
            pad_len = padded_head_dim - orig_head_dim
            query_for_flex = F.pad(query_for_flex, (0, pad_len))
            key_for_flex = F.pad(key_for_flex, (0, pad_len))
            value_for_flex = F.pad(value_for_flex, (0, pad_len))
            if flex_scale is None:
                flex_scale = 1.0 / math.sqrt(orig_head_dim)

        def cp_score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
            allowed = cp_mask(batch_idx, head_idx, q_idx, kv_idx)
            return torch.where(allowed, score, torch.full_like(score, -float("inf")))

        flex_kwargs = {
            "score_mod": cp_score_mod,
            "block_mask": block_mask,
            "scale": flex_scale,
            "enable_gqa": ctx.enable_gqa,
        }
        if use_small_flex_blocks:
            flex_kwargs["kernel_options"] = {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
                "num_stages": 1,
                "num_warps": 4,
            }

        with _duck_shape_disabled():
            try:
                out, lse = _compiled_flex_attention(attention_module)(
                    query_for_flex.contiguous(), key_for_flex, value_for_flex, return_lse=True, **flex_kwargs
                )
            except TypeError as exc:
                if "kernel_options" in str(exc) and "kernel_options" in flex_kwargs:
                    flex_kwargs.pop("kernel_options")
                    out, lse = _compiled_flex_attention(attention_module)(
                        query_for_flex.contiguous(), key_for_flex, value_for_flex, return_lse=True, **flex_kwargs
                    )
                else:
                    raise

        if empty_query_rows is not None and empty_query_rows.any():
            out = out.masked_fill(empty_query_rows[:, None, :, None], 0)
        if chunk_empty_query_rows.any():
            out = out.masked_fill(chunk_empty_query_rows[:, None, :, None], 0)
            lse = lse.masked_fill(chunk_empty_query_rows[:, None, :], -float("inf"))
        _debug_compare_flex_sdpa(
            attention_module,
            ctx,
            query_for_flex=query_for_flex,
            key_for_flex=key_for_flex,
            value_for_flex=value_for_flex,
            block_mask=block_mask,
            cp_score_mod=cp_score_mod,
            dense_allowed_mask=chunk_allowed,
            empty_query_rows=empty_query_rows,
            chunk_empty_query_rows=chunk_empty_query_rows,
            flex_scale=flex_scale,
            flex_kwargs=flex_kwargs,
            flex_out=out,
            kv_global_start=kv_global_start,
        )
        if padded_head_dim != orig_head_dim:
            out = out[..., :orig_head_dim]

        return out, lse, empty_query_rows, padded_head_dim
    except Exception as flex_err:
        raise RuntimeError(
            "Gemma4 CP ring requires FlexAttention for local-query/ring-KV attention. "
            f"FlexAttention failed for Q={tuple(query.shape)} K={tuple(key_chunk.shape)} "
            f"V={tuple(value_chunk.shape)} cp_rank={ctx.cp_rank} seq_local={ctx.seq_local} "
            f"kv_global_start={kv_global_start}."
        ) from flex_err


def _collect_ring_kv_chunks(ctx: Any) -> list[tuple[int, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]]:
    current_key = ctx.key.contiguous()
    current_value = ctx.value.contiguous()
    current_metadata = {name: value.contiguous() if value is not None else None for name, value in ctx.metadata.items()}
    current_owner = ctx.cp_rank
    chunks = []

    for step in range(ctx.cp_size):
        chunks.append((current_owner, current_key, current_value, current_metadata))

        if step == ctx.cp_size - 1:
            break

        recv_key = torch.empty_like(current_key)
        recv_value = torch.empty_like(current_value)
        recv_metadata = _metadata_like(current_metadata)
        exchange_tensors = [(current_key, recv_key), (current_value, recv_value)]
        exchange_tensors.extend(
            (current_metadata[name], recv_metadata[name])
            for name in sorted(current_metadata)
            if current_metadata[name] is not None
        )
        _ring_exchange(exchange_tensors, cp_group=ctx.cp_group, cp_rank=ctx.cp_rank, cp_size=ctx.cp_size)
        current_key = recv_key
        current_value = recv_value
        current_metadata = recv_metadata
        current_owner = (current_owner - 1) % ctx.cp_size

    return chunks


def _run_gemma4_cp_ring_attention_forward(attention_module: torch.nn.Module, ctx: Any) -> torch.Tensor:
    """Run Gemma4 local-query/ring-key CP attention forward with FlexAttention."""
    if ctx.dropout_p:
        raise NotImplementedError("Gemma4 FlexAttention ring CP does not support attention dropout.")

    out_acc = None
    lse_acc = None
    empty_query_rows = None
    padded_head_dim = ctx.query.shape[-1]

    _log_cp_memory(attention_module, "forward_start")
    for current_owner, current_key, current_value, current_metadata in _collect_ring_kv_chunks(ctx):
        kv_global_start = current_owner * ctx.seq_local
        out_step, lse_step, empty_query_rows, padded_head_dim = _run_gemma4_flex_chunk(
            attention_module,
            ctx,
            key_chunk=current_key,
            value_chunk=current_value,
            metadata_chunk=current_metadata,
            kv_global_start=kv_global_start,
        )
        out_acc, lse_acc = _merge_flex_chunk(out_acc, lse_acc, out_step, lse_step)
        if os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE_CP_VERBOSE", "").lower() in {"1", "true", "yes", "on"}:
            _log_cp_memory(attention_module, f"forward_after_owner={current_owner}")

    if out_acc is None:
        raise RuntimeError("Gemma4 CP ring attention produced no output chunks.")
    if empty_query_rows is not None and empty_query_rows.any():
        out_acc = out_acc.masked_fill(empty_query_rows[:, None, :, None], 0)

    global _GEMMA4_CP_FLEX_RING_OK_LOGGED
    if not _GEMMA4_CP_FLEX_RING_OK_LOGGED:
        logger.info(
            "Gemma4 CP using compiled flex_attention p2p ring. Q=%s K_local=%s head_dim=%s->%s cp_rank=%s cp_size=%s",
            tuple(ctx.query.shape),
            tuple(ctx.key.shape),
            ctx.query.shape[-1],
            padded_head_dim,
            ctx.cp_rank,
            ctx.cp_size,
        )
        _GEMMA4_CP_FLEX_RING_OK_LOGGED = True
    _log_cp_memory(attention_module, "forward_end")
    return out_acc.to(ctx.query.dtype)


def _zero_if_none(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(like) if grad is None else grad


class _Gemma4FlexRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(autograd_ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring_ctx: Any):
        _log_cp_memory(ring_ctx.module, "autograd_forward_start")
        runtime_ctx = replace(ring_ctx, query=query, key=key, value=value)
        out = _run_gemma4_cp_ring_attention_forward(ring_ctx.module, runtime_ctx)
        autograd_ctx.save_for_backward(query, key, value)
        autograd_ctx.ring_ctx = replace(
            ring_ctx,
            query=None,
            key=None,
            value=None,
            metadata=_detach_metadata(ring_ctx.metadata),
        )
        _log_cp_memory(ring_ctx.module, "autograd_forward_end")
        return out

    @staticmethod
    def backward(autograd_ctx, grad_output: torch.Tensor):
        query, key, value = autograd_ctx.saved_tensors
        ring_ctx = autograd_ctx.ring_ctx
        _log_cp_memory(ring_ctx.module, "backward_start")

        with torch.enable_grad():
            query_req = query.detach().requires_grad_(True)
            collect_ctx = replace(
                ring_ctx,
                query=query_req,
                key=key.detach(),
                value=value.detach(),
                metadata=_detach_metadata(ring_ctx.metadata),
            )
            chunks = _collect_ring_kv_chunks(collect_ctx)
            _log_cp_memory(ring_ctx.module, "backward_after_collect")

            runtime_ctx = replace(collect_ctx, query=query_req)
            key_reqs = []
            value_reqs = []
            owners = []
            out_acc = None
            lse_acc = None
            empty_query_rows = None

            for owner, key_chunk, value_chunk, metadata_chunk in chunks:
                key_req = key_chunk.detach().requires_grad_(True)
                value_req = value_chunk.detach().requires_grad_(True)
                key_reqs.append(key_req)
                value_reqs.append(value_req)
                owners.append(owner)

                out_step, lse_step, empty_query_rows, _ = _run_gemma4_flex_chunk(
                    ring_ctx.module,
                    runtime_ctx,
                    key_chunk=key_req,
                    value_chunk=value_req,
                    metadata_chunk=metadata_chunk,
                    kv_global_start=owner * ring_ctx.seq_local,
                )
                out_acc, lse_acc = _merge_flex_chunk(out_acc, lse_acc, out_step, lse_step)
                if os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE_CP_VERBOSE", "").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    _log_cp_memory(ring_ctx.module, f"backward_after_recompute_owner={owner}")

            if out_acc is None:
                raise RuntimeError("Gemma4 CP ring attention backward produced no output chunks.")
            if empty_query_rows is not None and empty_query_rows.any():
                out_acc = out_acc.masked_fill(empty_query_rows[:, None, :, None], 0)

            _log_cp_memory(ring_ctx.module, "backward_before_autograd_grad")
            grad_targets = [query_req, *key_reqs, *value_reqs]
            grads = torch.autograd.grad(out_acc, grad_targets, grad_output, allow_unused=True)
            _log_cp_memory(ring_ctx.module, "backward_after_autograd_grad", include_tensors=True)

        grad_query = _zero_if_none(grads[0], query)
        num_chunks = len(chunks)
        grad_key_by_owner = {owner: _zero_if_none(grads[1 + idx], key_reqs[idx]) for idx, owner in enumerate(owners)}
        grad_value_by_owner = {
            owner: _zero_if_none(grads[1 + num_chunks + idx], value_reqs[idx]) for idx, owner in enumerate(owners)
        }

        grad_key = grad_key_by_owner[ring_ctx.cp_rank].contiguous()
        grad_value = grad_value_by_owner[ring_ctx.cp_rank].contiguous()
        for distance in range(1, ring_ctx.cp_size):
            send_owner = (ring_ctx.cp_rank - distance) % ring_ctx.cp_size
            recv_query_rank = (ring_ctx.cp_rank + distance) % ring_ctx.cp_size
            recv_grad_key = torch.empty_like(grad_key)
            recv_grad_value = torch.empty_like(grad_value)
            _direct_exchange(
                [
                    (grad_key_by_owner[send_owner], recv_grad_key),
                    (grad_value_by_owner[send_owner], recv_grad_value),
                ],
                cp_group=ring_ctx.cp_group,
                cp_rank=ring_ctx.cp_rank,
                send_cp_rank=send_owner,
                recv_cp_rank=recv_query_rank,
            )
            grad_key = grad_key + recv_grad_key
            grad_value = grad_value + recv_grad_value
        _log_cp_memory(ring_ctx.module, "backward_end", include_tensors=True)

        return grad_query, grad_key, grad_value, None


def _run_gemma4_cp_ring_attention(attention_module: torch.nn.Module, ctx: Any) -> torch.Tensor:
    """Run Gemma4 local-query/ring-key CP attention with FlexAttention."""
    return _Gemma4FlexRingAttention.apply(ctx.query, ctx.key, ctx.value, ctx)


def _gemma4_cp_manual_attention(
    attention_module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cp_mesh,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    kwargs,
) -> torch.Tensor:
    """Gemma4-owned manual ring CP attention entry.

    Plugs into cp_utils' generic ``run_cp_manual_attention`` seam: receives the
    raw local (un-gathered) Q/K/V plus ``cp_mesh``, builds the ring context, and
    runs the p2p ring FlexAttention. K/V are rotated across CP ranks inside the
    ring autograd function -- they are never all-gathered.
    """
    group = cp_mesh.get_group()
    size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=group)
    seq_local = key.shape[2]
    seq_global_start = cp_rank * seq_local
    seq_full = seq_local * size
    if query.shape[1] != key.shape[1]:
        num_key_value_groups = query.shape[1] // key.shape[1]
        key = _repeat_kv_for_gqa(key, num_key_value_groups)
        value = _repeat_kv_for_gqa(value, num_key_value_groups)
        enable_gqa = False

    local_metadata = getattr(attention_module, "_cp_manual_metadata", {})
    metadata_seq_dims = getattr(attention_module, "_cp_manual_metadata_seq_dims", {})
    ctx = CPRingAttentionContext(
        module=attention_module,
        query=query,
        key=key,
        value=value,
        cp_mesh=cp_mesh,
        cp_group=group,
        cp_size=size,
        cp_rank=cp_rank,
        seq_local=seq_local,
        seq_full=seq_full,
        seq_global_start=seq_global_start,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        kwargs=kwargs,
        metadata=local_metadata,
        metadata_seq_dims=metadata_seq_dims,
    )
    return _run_gemma4_cp_ring_attention(attention_module, ctx)


def _install_gemma4_cp_ring_sdpa(attention_module: torch.nn.Module, cp_mesh) -> None:
    """Swap ``F.scaled_dot_product_attention`` -> Gemma4 ring CP attention on this module.

    Gemma4 owns its CP attention end-to-end (it does not use cp_utils' generic CP
    SDPA hooks). It installs its own ``@torch._dynamo.disable`` SDPA wrapper -- on
    the inner attention module so it also fires during gradient-checkpointing
    recompute -- that runs the p2p ring FlexAttention. The per-forward attention
    kwargs the ring needs (mm_token_type_ids, packed-seq ids, padding/vision masks)
    are captured off the forward kwargs into ``_cp_manual_metadata`` here, since the
    swapped SDPA only receives Q/K/V.
    """
    import torch.nn.functional as F_module

    original_sdpa = F_module.scaled_dot_product_attention
    metadata_keys = getattr(attention_module, "_cp_manual_metadata_keys", ())

    @torch._dynamo.disable
    def _ring_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        return _gemma4_cp_manual_attention(
            attention_module,
            query,
            key,
            value,
            cp_mesh=cp_mesh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            kwargs=kwargs,
        )

    def _pre_hook(module, args, kwargs):
        captured = {name: kwargs.pop(name, None) for name in metadata_keys}
        # Dense Gemma4 routes through HF's decoder layers, which do not thread the
        # CP/vision metadata down to self_attn kwargs (the MoE backend does). The
        # dense forward stashes it on the module as ``_cp_dense_metadata``; fall
        # back to that for any key the caller didn't pass so the ring can still
        # build the vision-bidirectional / packed masks. Persisted across the step
        # (not cleared by the post-hook) so it survives activation-checkpoint recompute.
        fallback = getattr(module, "_cp_dense_metadata", None)
        if fallback:
            for name in metadata_keys:
                if captured.get(name) is None:
                    captured[name] = fallback.get(name)
        module._cp_manual_metadata = captured
        # Own the CP mask handling (instead of relying on the generic
        # attach_context_parallel_hooks mask-strip): drop any full-sequence 4D
        # attention_mask -- invalid once the sequence is CP-sharded -- and force
        # the causal path, which the ring honors via run_cp_manual_attention.
        kwargs["attention_mask"] = None
        kwargs["is_causal"] = True
        F_module.scaled_dot_product_attention = _ring_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        module._cp_manual_metadata = {}
        F_module.scaled_dot_product_attention = original_sdpa

    attention_module._cp_uses_attention_hook = True
    attention_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    attention_module.register_forward_hook(_post_hook, always_call=True)


def attach_gemma4_cp_ring_attention(attention_module: torch.nn.Module) -> None:
    """Register Gemma4's model-owned p2p ring CP attention on a self-attention module.

    Declares the metadata keys the ring needs and exposes ``setup_cp_attention(cp_mesh)``
    -- the model-owned CP-attention seam the parallelizer calls (with the CP mesh)
    instead of cp_utils' generic SDPA hooks. ``run_cp_manual_attention`` is also bound
    as the ring entry point.
    """
    global _GEMMA4_CP_PROFILE_NEXT_MODULE_ID
    if not hasattr(attention_module, "_gemma4_cp_profile_id"):
        attention_module._gemma4_cp_profile_id = _GEMMA4_CP_PROFILE_NEXT_MODULE_ID
        _GEMMA4_CP_PROFILE_NEXT_MODULE_ID += 1
    attention_module._cp_manual_metadata_keys = (
        "mm_token_type_ids",
        "_packed_seq_ids",
        "padding_mask",
        "_gemma4_vision_group_ids",
    )
    attention_module._cp_manual_metadata_seq_dims = {
        "mm_token_type_ids": 1,
        "_packed_seq_ids": 1,
        "padding_mask": 1,
        "_gemma4_vision_group_ids": 1,
    }
    attention_module.run_cp_manual_attention = MethodType(_gemma4_cp_manual_attention, attention_module)
    attention_module.setup_cp_attention = MethodType(_install_gemma4_cp_ring_sdpa, attention_module)
