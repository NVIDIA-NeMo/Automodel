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
# ``cp_utils.make_cp_batch_and_ctx`` delegates manual all-gather CP to the model
# via the ``CPSharder`` returned by ``prepare_model_inputs_for_cp``. DSV4's
# sharder pads + contiguously shards the sequence per CP rank (via the shared
# contiguous implementation in ``components/distributed/cp_sharder.py``) and
# hands the CP process group to the forward (``_dsv4_cp_group``) so DSV4
# attention can all-gather K/V.
# ---------------------------------------------------------------------------


def make_dsv4_contiguous_shard_cp_batch_and_ctx(
    cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id: int = 0, pad_multiple: int | None = None
):
    """Contiguously shard a batch for DeepSeek V4 Miles-style context parallelism.

    Exposed as ``CPSharder.shard_batch`` (via ``functools.partial`` to bind
    ``pad_multiple``) and invoked by ``cp_utils.make_cp_batch_and_ctx``. Each CP rank
    keeps one ``seq_start:seq_end`` slice; DSV4 attention all-gathers K/V across CP
    ranks during forward. No collective happens here -- this is the batch-side
    counterpart of ``cp.py``'s activation gathers. Returns ``(nullcontext, batch)``.

    ``pad_multiple`` is the required *per-CP-rank* shard multiple (from
    ``dsv4_cp_local_seq_multiple``); the global sequence is padded so it is divisible
    by ``cp_size`` and each local shard is divisible by ``pad_multiple`` (>= 2).
    """
    from nemo_automodel.components.distributed.cp_sharder import shard_batch_contiguous  # noqa: PLC0415

    if "cu_seqlens" in batch or batch.get("qkv_format") == "thd":
        raise NotImplementedError("DeepSeek V4 context parallelism with packed sequences is not implemented yet.")

    ctx, batch = shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        pad_multiple=int(pad_multiple or 2),
        synthesize_packed_seq_ids=False,
    )
    # Hand the CP process group to the model forward (read from attn_kwargs) so
    # DSV4 attention all-gathers K/V across CP ranks. Not a tensor -> not sharded.
    batch["_dsv4_cp_group"] = cp_mesh.get_group()
    return ctx, batch
