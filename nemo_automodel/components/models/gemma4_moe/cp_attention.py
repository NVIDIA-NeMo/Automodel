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

import logging
import math
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_GEMMA4_CP_FLEX_OK_LOGGED = False


def gemma4_vision_group_ids(mm_token_type_ids: torch.Tensor) -> torch.Tensor:
    """Return per-image-block ids for Gemma4 vision tokens, or -1 for text/padding."""
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))


def _compiled_flex_attention(attention_module: torch.nn.Module):
    compiled = getattr(attention_module, "_gemma4_cp_compiled_flex_attn", None)
    if compiled is None:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)
        attention_module._gemma4_cp_compiled_flex_attn = compiled
    return compiled


def _base_gemma4_cp_mask(attention_module: torch.nn.Module, ctx: Any, q_idx, kv_idx):
    q_global_idx = q_idx + ctx.seq_global_start
    if not ctx.is_causal:
        allowed = torch.ones_like(q_global_idx >= kv_idx)
    else:
        allowed = kv_idx <= q_global_idx

    sliding_window = getattr(attention_module, "sliding_window", None)
    if sliding_window is not None:
        allowed = allowed & ((q_global_idx - kv_idx) < sliding_window)
    return allowed


def _run_gemma4_cp_allgather_attention(attention_module: torch.nn.Module, ctx: Any) -> torch.Tensor:
    """Run Gemma4 local-query/global-key CP attention with FlexAttention."""
    query = ctx.query
    key_full = ctx.key_full
    value_full = ctx.value_full
    orig_head_dim = query.shape[-1]
    padded_head_dim = 1 << (orig_head_dim - 1).bit_length()
    use_small_flex_blocks = padded_head_dim > 256
    flex_block_size = (32, 32) if use_small_flex_blocks else 128

    mm_token_type_ids_full = ctx.metadata.get("mm_token_type_ids")
    packed_seq_ids_full = ctx.metadata.get("_packed_seq_ids")
    padding_mask_full = ctx.metadata.get("padding_mask")
    vision_group_ids = gemma4_vision_group_ids(mm_token_type_ids_full) if mm_token_type_ids_full is not None else None

    sliding_window = getattr(attention_module, "sliding_window", None)
    config_uses_vision_bidir = (
        getattr(getattr(attention_module, "config", None), "use_bidirectional_attention", None) == "vision"
    )
    has_vision_tokens = vision_group_ids is not None and bool((vision_group_ids >= 0).any().item())
    use_vision_bidirectional = sliding_window is not None and config_uses_vision_bidir and has_vision_tokens

    q_indices = torch.arange(ctx.seq_local, device=query.device) + ctx.seq_global_start
    empty_query_rows = None
    if packed_seq_ids_full is not None:
        empty_query_rows = packed_seq_ids_full[:, q_indices] <= 0
    if padding_mask_full is not None:
        padding_query_rows = padding_mask_full[:, q_indices]
        empty_query_rows = padding_query_rows if empty_query_rows is None else empty_query_rows | padding_query_rows

    try:
        from torch.nn.attention.flex_attention import create_block_mask

        if use_vision_bidirectional or packed_seq_ids_full is not None or padding_mask_full is not None:

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                q_global_idx = q_idx + ctx.seq_global_start
                allowed = _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx)
                if use_vision_bidirectional:
                    q_group = vision_group_ids[batch_idx, q_global_idx]
                    kv_group = vision_group_ids[batch_idx, kv_idx]
                    same_vision_group = (q_group == kv_group) & (q_group >= 0)
                    allowed = allowed | same_vision_group
                if packed_seq_ids_full is not None:
                    q_pack_id = packed_seq_ids_full[batch_idx, q_global_idx]
                    kv_pack_id = packed_seq_ids_full[batch_idx, kv_idx]
                    allowed = allowed & (q_pack_id == kv_pack_id) & (q_pack_id > 0)
                    allowed = torch.where(q_pack_id <= 0, kv_idx == 0, allowed)
                if padding_mask_full is not None:
                    q_is_padding = padding_mask_full[batch_idx, q_global_idx]
                    kv_is_padding = padding_mask_full[batch_idx, kv_idx]
                    allowed = allowed & ~kv_is_padding
                    allowed = torch.where(q_is_padding, kv_idx == 0, allowed)
                return allowed

            block_mask_batch = query.shape[0]
        else:

            def cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                return _base_gemma4_cp_mask(attention_module, ctx, q_idx, kv_idx)

            block_mask_batch = None

        block_mask = create_block_mask(
            cp_mask,
            B=block_mask_batch,
            H=None,
            Q_LEN=ctx.seq_local,
            KV_LEN=ctx.seq_full,
            device=query.device,
            BLOCK_SIZE=flex_block_size,
        )

        query_for_flex = query
        key_for_flex = key_full
        value_for_flex = value_full
        flex_scale = ctx.scale
        if padded_head_dim != orig_head_dim:
            pad_len = padded_head_dim - orig_head_dim
            query_for_flex = F.pad(query_for_flex, (0, pad_len))
            key_for_flex = F.pad(key_for_flex, (0, pad_len))
            value_for_flex = F.pad(value_for_flex, (0, pad_len))
            if flex_scale is None:
                flex_scale = 1.0 / math.sqrt(orig_head_dim)

        flex_kwargs = {"block_mask": block_mask, "scale": flex_scale, "enable_gqa": ctx.enable_gqa}
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

        try:
            out = _compiled_flex_attention(attention_module)(
                query_for_flex.contiguous(), key_for_flex, value_for_flex, **flex_kwargs
            )
        except TypeError as exc:
            if "kernel_options" in str(exc) and "kernel_options" in flex_kwargs:
                flex_kwargs.pop("kernel_options")
                out = _compiled_flex_attention(attention_module)(
                    query_for_flex.contiguous(), key_for_flex, value_for_flex, **flex_kwargs
                )
            else:
                raise

        if empty_query_rows is not None and empty_query_rows.any():
            out = out.masked_fill(empty_query_rows[:, None, :, None], 0)
        if padded_head_dim != orig_head_dim:
            out = out[..., :orig_head_dim]

        global _GEMMA4_CP_FLEX_OK_LOGGED
        if not _GEMMA4_CP_FLEX_OK_LOGGED:
            logger.info(
                "Gemma4 CP using compiled flex_attention all-gather. Q=%s K=%s head_dim=%s->%s cp_rank=%s",
                tuple(query.shape),
                tuple(key_full.shape),
                orig_head_dim,
                padded_head_dim,
                ctx.cp_rank,
            )
            _GEMMA4_CP_FLEX_OK_LOGGED = True
        return out
    except Exception as flex_err:
        raise RuntimeError(
            "Gemma4 CP all-gather requires FlexAttention for local-query/global-key attention. "
            f"FlexAttention failed for Q={tuple(query.shape)} K={tuple(key_full.shape)} "
            f"V={tuple(value_full.shape)} cp_rank={ctx.cp_rank} seq_local={ctx.seq_local} seq_full={ctx.seq_full}."
        ) from flex_err


@dataclass(frozen=True)
class CPAllGatherAttentionContext:
    """Inputs for Gemma4 manual all-gather CP attention."""

    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_full: torch.Tensor
    value_full: torch.Tensor
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


def _cp_all_gather_seq(tensor: torch.Tensor, group, size: int) -> torch.Tensor:
    """All-gather an attention tensor along the sequence dim (dim=2), autograd-aware when available."""
    tensor = tensor.contiguous()
    try:
        from torch.distributed.nn.functional import all_gather as dist_all_gather

        parts = dist_all_gather(tensor, group=group)
    except (ImportError, AttributeError):
        parts = [torch.empty_like(tensor) for _ in range(size)]
        torch.distributed.all_gather(parts, tensor, group=group)
    return torch.cat(tuple(parts), dim=2)


def _cp_all_gather_seq_metadata(
    metadata: torch.Tensor | None, group, size: int, seq_dim: int = 1
) -> torch.Tensor | None:
    """All-gather non-differentiable per-token metadata (masks / packed ids) along ``seq_dim``."""
    if metadata is None:
        return None
    local = metadata.contiguous()
    parts = [torch.empty_like(local) for _ in range(size)]
    torch.distributed.all_gather(parts, local, group=group)
    return torch.cat(parts, dim=seq_dim)


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
    """Gemma4-owned manual all-gather CP transport.

    Each CP rank keeps its local query shard; K/V (and per-token metadata) are
    all-gathered to the full sequence, then the local-query/global-key FlexAttention
    in :func:`_run_gemma4_cp_allgather_attention` consumes them. This is the transport
    that ``cp_utils.attach_cp_attention_hooks`` invokes via the generic
    ``module.run_cp_manual_attention`` seam, so ``cp_utils`` stays model-agnostic.
    """
    group = cp_mesh.get_group()
    size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=group)
    seq_local = key.shape[2]
    seq_global_start = cp_rank * seq_local

    key_full = _cp_all_gather_seq(key, group, size)
    value_full = _cp_all_gather_seq(value, group, size)
    seq_full = key_full.shape[2]
    if query.shape[1] != key_full.shape[1]:
        enable_gqa = True

    local_metadata = getattr(attention_module, "_cp_allgather_metadata", {})
    metadata_seq_dims = getattr(attention_module, "_cp_allgather_metadata_seq_dims", {})
    metadata_full = {
        name: _cp_all_gather_seq_metadata(value_, group, size, metadata_seq_dims.get(name, 1))
        for name, value_ in local_metadata.items()
    }

    ctx = CPAllGatherAttentionContext(
        module=attention_module,
        query=query,
        key=key,
        value=value,
        key_full=key_full,
        value_full=value_full,
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
        metadata=metadata_full,
    )
    return _run_gemma4_cp_allgather_attention(attention_module, ctx)


def attach_gemma4_cp_allgather_attention(attention_module: torch.nn.Module) -> None:
    """Register Gemma4's model-specific manual all-gather CP attention handler.

    Gemma4 owns the whole CP attention (transport + FlexAttention compute). It plugs
    into the generic ``cp_utils`` SDPA-swap via the ``run_cp_manual_attention`` seam,
    so ``cp_utils`` carries no all-gather-specific code.
    """
    attention_module._cp_allgather_metadata_keys = ("mm_token_type_ids", "_packed_seq_ids", "padding_mask")
    attention_module._cp_allgather_metadata_seq_dims = {
        "mm_token_type_ids": 1,
        "_packed_seq_ids": 1,
        "padding_mask": 1,
    }
    attention_module.run_cp_manual_attention = MethodType(_gemma4_cp_manual_attention, attention_module)
