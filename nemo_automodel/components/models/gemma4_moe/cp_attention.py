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
from dataclasses import dataclass, replace
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_GEMMA4_CP_FLEX_RING_OK_LOGGED = False


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


def _compiled_flex_attention(attention_module: torch.nn.Module):
    compiled = getattr(attention_module, "_gemma4_cp_compiled_flex_attn", None)
    if compiled is None:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)
        attention_module._gemma4_cp_compiled_flex_attn = compiled
    return compiled


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
    old_scale = torch.exp(lse_acc - lse_next).unsqueeze(-1)
    new_scale = torch.exp(lse_step - lse_next).unsqueeze(-1)
    return out_acc * old_scale + out_step * new_scale, lse_next


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

        block_mask = create_block_mask(
            cp_mask,
            B=block_mask_batch,
            H=None,
            Q_LEN=ctx.seq_local,
            KV_LEN=key_chunk.shape[2],
            device=query.device,
            BLOCK_SIZE=flex_block_size,
        )

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
    return out_acc.to(ctx.query.dtype)


def _zero_if_none(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(like) if grad is None else grad


class _Gemma4FlexRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(autograd_ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ring_ctx: Any):
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
        return out

    @staticmethod
    def backward(autograd_ctx, grad_output: torch.Tensor):
        query, key, value = autograd_ctx.saved_tensors
        ring_ctx = autograd_ctx.ring_ctx

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

            if out_acc is None:
                raise RuntimeError("Gemma4 CP ring attention backward produced no output chunks.")
            if empty_query_rows is not None and empty_query_rows.any():
                out_acc = out_acc.masked_fill(empty_query_rows[:, None, :, None], 0)

            grad_targets = [query_req, *key_reqs, *value_reqs]
            grads = torch.autograd.grad(out_acc, grad_targets, grad_output, allow_unused=True)

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
        enable_gqa = True

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
        module._cp_manual_metadata = {name: kwargs.pop(name, None) for name in metadata_keys}
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
