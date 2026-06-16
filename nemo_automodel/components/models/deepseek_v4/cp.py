# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Context-parallel helpers for the DeepSeek V4 custom model.

This implements the Miles-style training path: each CP rank owns a contiguous
query shard, while K/V and compressed K/V are all-gathered with autograd-aware
collectives before DSV4 sparse attention consumes them.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else max(a, b)


def dsv4_cp_local_seq_multiple(model_or_config) -> int:
    """Required per-CP-rank sequence-length multiple for DSV4 Miles-style CP.

    Compress-ratio layers constrain how the sequence may be split across CP ranks:
    a ratio-R layer needs each local shard divisible by R, and ratio-4 layers use
    cross-window overlap so they need ``2*R``. The returned value is the LCM across
    all configured ``compress_ratios`` (1 when none are configured).
    """
    config = getattr(model_or_config, "config", model_or_config)
    ratios = [int(r) for r in (getattr(config, "compress_ratios", None) or []) if int(r) > 0]
    multiple = 1
    for ratio in ratios:
        required = 2 * ratio if ratio == 4 else ratio
        multiple = _lcm(multiple, required)
    return max(multiple, 1)


def dsv4_cp_enabled(cp_group) -> bool:
    """Return whether a real CP process group is active."""
    return (
        cp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=cp_group) > 1
    )


def dsv4_cp_rank(cp_group) -> int:
    """Return this rank's index in the DSV4 CP group, or 0 without CP."""
    return dist.get_rank(group=cp_group) if dsv4_cp_enabled(cp_group) else 0


def dsv4_cp_size(cp_group) -> int:
    """Return the DSV4 CP group size, or 1 without CP."""
    return dist.get_world_size(group=cp_group) if dsv4_cp_enabled(cp_group) else 1


def dsv4_cp_all_gather(tensor: torch.Tensor, *, dim: int, cp_group) -> torch.Tensor:
    """All-gather activation tensors across CP ranks and concatenate on ``dim``.

    The distributed.nn functional collective preserves autograd, so backward
    routes gradients for gathered remote slices back to their owning ranks.
    """
    if not dsv4_cp_enabled(cp_group):
        return tensor

    try:
        from torch.distributed.nn.functional import all_gather
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version guard
        raise RuntimeError(
            "DeepSeek V4 context parallelism requires torch.distributed.nn.functional.all_gather "
            "for differentiable activation gathers."
        ) from exc

    parts = all_gather(tensor.contiguous(), group=cp_group)
    return torch.cat(tuple(parts), dim=dim)


def dsv4_cp_all_gather_metadata(tensor: torch.Tensor | None, *, dim: int, cp_group) -> torch.Tensor | None:
    """All-gather non-differentiable metadata such as padding masks."""
    if tensor is None or not dsv4_cp_enabled(cp_group):
        return tensor
    local = tensor.contiguous()
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(parts, local, group=cp_group)
    return torch.cat(parts, dim=dim)


def build_dsv4_cp_causal_padding_mask(
    *,
    position_ids: torch.Tensor,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
    cp_group,
    padding_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build local-query/global-key additive mask for Miles-style DSV4 CP.

    ``position_ids`` are the local query positions after contiguous CP slicing.
    Keys are in global sequence order because DSV4 gathers K/V along sequence.
    ``padding_mask`` follows the internal convention True=padding.
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    q_pos = position_ids.to(device=device, dtype=torch.long)
    batch, seq_len = q_pos.shape
    k_pos = torch.arange(key_len, device=device, dtype=torch.long)

    allowed = k_pos.view(1, 1, key_len) <= q_pos.view(batch, seq_len, 1)
    if sliding_window is not None:
        allowed = allowed & ((q_pos.view(batch, seq_len, 1) - k_pos.view(1, 1, key_len)) < sliding_window)

    padding_mask_full = dsv4_cp_all_gather_metadata(padding_mask, dim=1, cp_group=cp_group)
    if padding_mask_full is not None:
        allowed = allowed & ~padding_mask_full.to(device=device, dtype=torch.bool).view(batch, 1, key_len)

    min_value = torch.finfo(dtype).min
    return torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_value, dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Model-owned CP batch sharding (Miles-style contiguous query shard).
#
# main's ``cp_utils.make_cp_batch_and_ctx`` delegates manual all-gather CP to the
# model via a ``_cp_make_batch_fn`` callable attached to the batch (see Gemma4's
# ``make_contiguous_shard_cp_batch_and_ctx``). DSV4 attaches the function below:
# it pads + contiguously shards the sequence per CP rank and hands the CP process
# group to the forward (``_dsv4_cp_group``) so DSV4 attention can all-gather K/V.
# ---------------------------------------------------------------------------


def _pad_tensor_seq_dim_(tensor: torch.Tensor, seq_dim: int, pad_len: int, value) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids_seq_dim_(position_ids: torch.Tensor, seq_dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    inc_shape = [1] * position_ids.ndim
    inc_shape[seq_dim] = pad_len
    inc = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(inc_shape)
    return torch.cat((position_ids, last + inc), dim=seq_dim)


def make_dsv4_contiguous_shard_cp_batch_and_ctx(
    cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id: int = 0, pad_multiple: int | None = None
):
    """Contiguously shard a batch for DeepSeek V4 Miles-style context parallelism.

    Attached to the batch as ``_cp_make_batch_fn`` (via ``functools.partial`` to bind
    ``pad_multiple``) and invoked by ``cp_utils.make_cp_batch_and_ctx``. Each CP rank
    keeps one ``seq_start:seq_end`` slice; DSV4 attention all-gathers K/V across CP
    ranks during forward. No collective happens here -- this is the batch-side
    counterpart of ``cp.py``'s activation gathers. Returns ``(nullcontext, batch)``.

    ``pad_multiple`` is the required *per-CP-rank* shard multiple (from
    ``dsv4_cp_local_seq_multiple``); the global sequence is padded so it is divisible
    by ``cp_size`` and each local shard is divisible by ``pad_multiple`` (>= 2).
    """
    import contextlib

    if "cu_seqlens" in batch or batch.get("qkv_format") == "thd":
        raise NotImplementedError("DeepSeek V4 context parallelism with packed sequences is not implemented yet.")

    cp_size = cp_mesh.size()
    divisor = cp_size * max(int(pad_multiple or 2), 2)

    # attention_mask -> padding_mask (True == pad) so CP attention can rebuild the
    # local-query/global-key mask after K/V is all-gathered.
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is not None and "padding_mask" not in batch:
        if attention_mask.ndim == 4:
            diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
            batch["padding_mask"] = diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
        else:
            batch["padding_mask"] = attention_mask.bool().logical_not()

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_dsv4_contiguous_shard_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    seq_len = batch[primary_key].shape[1]
    batch_size = batch[primary_key].shape[0]

    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=batch[primary_key].device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    position_ids = batch["position_ids"]
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch.get("labels")
    if labels is None and loss_mask is not None:
        labels, loss_mask = loss_mask, None
    if labels is None:
        raise KeyError("DSV4 context parallelism requires `labels` in the batch, or labels passed as `loss_mask`.")

    pad_len = (-seq_len) % divisor
    if pad_len:
        if "input_ids" in batch:
            batch["input_ids"] = _pad_tensor_seq_dim_(batch["input_ids"], 1, pad_len, padding_token_id)
        if "inputs_embeds" in batch:
            batch["inputs_embeds"] = _pad_tensor_seq_dim_(batch["inputs_embeds"], 1, pad_len, 0)
        labels = _pad_tensor_seq_dim_(labels, 1, pad_len, -100)
        position_ids = _pad_position_ids_seq_dim_(position_ids, pos_seq_dim, pad_len)
        batch["position_ids"] = position_ids
        if "padding_mask" in batch:
            batch["padding_mask"] = _pad_tensor_seq_dim_(batch["padding_mask"], 1, pad_len, True)
        if loss_mask is not None:
            loss_mask = _pad_tensor_seq_dim_(loss_mask, 1, pad_len, 0)

    batch["labels"] = labels

    if dist.is_available() and dist.is_initialized():
        cp_rank = dist.get_rank(group=cp_mesh.get_group())
    else:
        cp_rank = getattr(cp_mesh, "get_local_rank", lambda: 0)()

    seq_len = batch[primary_key].shape[1]
    if seq_len % cp_size != 0:
        raise ValueError(
            f"DSV4 CP sequence length must be divisible by cp_size after padding, got {seq_len=} {cp_size=}"
        )
    local_seq_len = seq_len // cp_size
    seq_start = cp_rank * local_seq_len
    seq_end = seq_start + local_seq_len

    def _slice_seq(key: str, seq_dim: int = 1) -> None:
        if key not in batch:
            return
        slices = [slice(None)] * batch[key].ndim
        slices[seq_dim] = slice(seq_start, seq_end)
        batch[key] = batch[key][tuple(slices)].contiguous()

    _slice_seq("input_ids", 1)
    _slice_seq("inputs_embeds", 1)
    _slice_seq("labels", 1)
    _slice_seq("position_ids", pos_seq_dim)
    _slice_seq("padding_mask", 1)
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    # Hand the CP process group to the model forward (read from attn_kwargs) so
    # DSV4 attention all-gathers K/V across CP ranks. Not a tensor -> not sharded.
    batch["_dsv4_cp_group"] = cp_mesh.get_group()
    return contextlib.nullcontext, batch
