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

"""Context-parallel batch-sharding contract.

Every CP backend is a :class:`CPSharder`. A model that owns its CP batch
sharding and attention transport returns one from
``prepare_model_inputs_for_cp`` under the ``"cp_sharder"`` batch key; the
framework constructs its own for the remaining backends (torch
``context_parallel`` round-robin, TE/THD, MagiAttention) so
``cp_utils.make_cp_batch_and_ctx`` reduces to resolving a sharder and calling
``shard_batch``. This replaces the retired private batch keys
(``_cp_make_batch_fn``, ``_cp_metadata_seq_dims``, ``_cp_metadata_pad_values``,
``_cp_full_logits_grad_touch``).

The contract is a closed verb set with open implementations: the dataclass
slots are plain callables filled with functions from the owning model's
directory or from the framework. ``local_token_global_indices`` — the global
position of every local token — is the universal layout coordinate system: the
default token-tensor shard/gather are synthesized from it, so a sharder only
overrides them when it has a cheaper communication pattern. Layouts that are a
pure function of ``(cp_mesh, padded_seq_len)`` (contiguous, round-robin)
provide it; data-dependent layouts (THD ``cu_seqlens`` partitioning, magi's
dispatch solver) set it to None. ``layout`` is a diagnostic string; no
framework code may branch on it.

This module also hosts the framework's ``shard_batch`` implementations: the
shared contiguous-shard batch prep used by models whose CP ranks own contiguous
sequence slices (Gemma4, DeepSeek V4), and the torch ``context_parallel``
round-robin load-balanced prep with its index map. The TE/THD and magi preps
live with their dependencies (``cp_utils``, ``magi_attn_utils``); the
dispatcher wraps them into sharders at resolution time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def _cp_rank(cp_mesh) -> int:
    """Resolve this rank's index within the CP submesh (0 without distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=cp_mesh.get_group())
    return getattr(cp_mesh, "get_local_rank", lambda: 0)()


def contiguous_local_indices(cp_mesh, padded_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Global positions owned by this rank under contiguous rank-ordered sharding.

    Rank ``r`` owns ``[r * L, (r + 1) * L)`` with ``L = padded_seq_len // cp_size``.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        padded_seq_len: Global sequence length after CP padding; must be
            divisible by the CP size.
        device: Device for the returned index tensor.

    Returns:
        A 1-D int64 tensor of this rank's global token positions.
    """
    cp_size = cp_mesh.size()
    if padded_seq_len % cp_size != 0:
        raise ValueError(f"padded_seq_len must be divisible by cp_size, got {padded_seq_len=} {cp_size=}")
    local_len = padded_seq_len // cp_size
    start = _cp_rank(cp_mesh) * local_len
    return torch.arange(start, start + local_len, device=device, dtype=torch.long)


def round_robin_local_indices(cp_mesh, padded_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Global positions owned by this rank under torch ``context_parallel`` load balancing.

    The sequence splits into ``2 * cp_size`` chunks and rank ``r`` owns chunks
    ``r`` and ``2 * cp_size - 1 - r`` (head-tail pairing, so causal-attention
    work stays even across ranks).

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        padded_seq_len: Global sequence length after CP padding; must be
            divisible by ``2 * cp_size``.
        device: Device for the returned index tensor.

    Returns:
        A 1-D int64 tensor of this rank's global token positions, in local
        shard order (head chunk then tail chunk).
    """
    cp_size = cp_mesh.size()
    if padded_seq_len % (2 * cp_size) != 0:
        raise ValueError(f"padded_seq_len must be divisible by 2 * cp_size, got {padded_seq_len=} {cp_size=}")
    chunk_len = padded_seq_len // (2 * cp_size)
    rank = _cp_rank(cp_mesh)
    head = torch.arange(rank * chunk_len, (rank + 1) * chunk_len, device=device, dtype=torch.long)
    tail_start = (2 * cp_size - 1 - rank) * chunk_len
    tail = torch.arange(tail_start, tail_start + chunk_len, device=device, dtype=torch.long)
    return torch.cat((head, tail))


def shard_token_tensor_by_indices(tensor: torch.Tensor, local_indices: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
    """Keep this rank's slice of a full-length token-aligned tensor.

    Args:
        tensor: Full (padded) sequence tensor, e.g. ``[B, S]`` advantages.
        local_indices: This rank's global token positions.
        seq_dim: The sequence dimension of ``tensor``.

    Returns:
        The local shard, laid out identically to the model inputs.
    """
    return tensor.index_select(seq_dim, local_indices.to(tensor.device)).contiguous()


def _reorder_gathered_token_tensor(
    parts: list[torch.Tensor], index_parts: list[torch.Tensor], seq_dim: int = 1
) -> torch.Tensor:
    """Reassemble rank-order gathered shards into global sequence order.

    Args:
        parts: Per-rank local shards, in rank order.
        index_parts: Per-rank global token positions, in the same order.
        seq_dim: The sequence dimension of the shards.

    Returns:
        The full-sequence tensor with position ``i`` holding the token whose
        global index is ``i``.
    """
    full = torch.cat(parts, dim=seq_dim)
    full_indices = torch.cat([p.to(full.device) for p in index_parts])
    return full.index_select(seq_dim, torch.argsort(full_indices)).contiguous()


def gather_token_tensor_by_indices(
    cp_mesh, tensor: torch.Tensor, local_indices: torch.Tensor, seq_dim: int = 1
) -> torch.Tensor:
    """Differentiably gather a token-aligned local shard back to the full sequence.

    Uses ``torch.distributed.nn.functional.all_gather`` so gradients route back
    to the owning ranks, then restores global sequence order from the gathered
    per-rank indices. Identity when CP is inactive.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        tensor: This rank's local shard, e.g. ``[B, S/cp]`` token logprobs.
        local_indices: This rank's global token positions.
        seq_dim: The sequence dimension of ``tensor``.

    Returns:
        The full-sequence tensor in global order (padding not trimmed).
    """
    if cp_mesh is None or cp_mesh.size() <= 1 or not (dist.is_available() and dist.is_initialized()):
        return tensor

    from torch.distributed.nn.functional import all_gather  # noqa: PLC0415

    group = cp_mesh.get_group()
    parts = list(all_gather(tensor.contiguous(), group=group))
    local_indices = local_indices.to(tensor.device)
    index_parts = [torch.empty_like(local_indices) for _ in range(cp_mesh.size())]
    dist.all_gather(index_parts, local_indices.contiguous(), group=group)
    return _reorder_gathered_token_tensor(parts, index_parts, seq_dim=seq_dim)


@dataclass
class CPSharder:
    """CP backend description: how a batch is sharded and where local tokens live.

    Attributes:
        shard_batch: ``(cp_mesh, tp_mesh, batch, *, loss_mask=None,
            padding_token_id=0) -> (ctx_factory, batch)``. Pads and shards the
            batch and installs any backend-owned attention transport; returns
            the forward context factory and the sharded batch.
        local_token_global_indices: ``(cp_mesh, padded_seq_len, device) ->
            LongTensor`` with the global position of each local token. None for
            layouts that depend on batch content (THD ``cu_seqlens``
            partitioning, magi's dispatch solver), where the token-tensor
            shard/gather verbs are unavailable.
        layout: Diagnostic label; never branched on by framework code.
    """

    shard_batch: Callable[..., tuple[Callable, dict[str, Any]]]
    local_token_global_indices: Callable[..., torch.Tensor] | None
    layout: str = "custom"

    def _indices(self, cp_mesh, padded_seq_len: int, device) -> torch.Tensor:
        if self.local_token_global_indices is None:
            raise NotImplementedError(
                f"CPSharder(layout={self.layout!r}) has a data-dependent token layout and does not expose "
                "local_token_global_indices; token-tensor shard/gather are unavailable."
            )
        return self.local_token_global_indices(cp_mesh, padded_seq_len, device)

    def shard_token_tensor(self, cp_mesh, tensor: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """Shard a full-length token-aligned tensor exactly like the model inputs.

        ``tensor`` must already be padded to the CP-padded sequence length.
        """
        indices = self._indices(cp_mesh, tensor.shape[seq_dim], tensor.device)
        return shard_token_tensor_by_indices(tensor, indices, seq_dim=seq_dim)

    def gather_token_tensor(self, cp_mesh, tensor: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """Differentiably gather a token-aligned local shard to global order."""
        padded_seq_len = tensor.shape[seq_dim] * cp_mesh.size()
        indices = self._indices(cp_mesh, padded_seq_len, tensor.device)
        return gather_token_tensor_by_indices(cp_mesh, tensor, indices, seq_dim=seq_dim)


# ---------------------------------------------------------------------------
# Shared contiguous-shard batch implementation.
#
# Each CP rank keeps one contiguous ``seq_start:seq_end`` slice; no collective
# happens here — the transport lives in the owning model's attention (Gemma4
# ring FlexAttention, DSV4 K/V all-gather). This is the non-load-balanced peer
# of the ``context_parallel`` and TE/THD batch shardings.
# ---------------------------------------------------------------------------


def _pad_tensor_seq_dim_(tensor: torch.Tensor, seq_dim: int, pad_len: int, value: float | int = 0) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids_seq_dim_(position_ids: torch.Tensor, seq_dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last_position = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    increment_shape = [1] * position_ids.ndim
    increment_shape[seq_dim] = pad_len
    increments = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(
        increment_shape
    )
    return torch.cat((position_ids, last_position + increments), dim=seq_dim)


def convert_attention_mask_to_padding_mask(batch: dict) -> None:
    """Pop ``attention_mask`` and derive ``padding_mask`` (True == pad) in place.

    Preserves padding semantics for modules such as MoE routers after the
    attention mask is stripped for CP. Idempotent: a no-op when
    ``padding_mask`` already exists or there is no ``attention_mask``.
    """
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is not None and "padding_mask" not in batch:
        if attention_mask.ndim == 4:
            diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
            batch["padding_mask"] = diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
        else:
            batch["padding_mask"] = attention_mask.bool().logical_not()


def _prepare_manual_cp_batch(cp_mesh, tp_mesh, batch, loss_mask):
    """Pre-shard prep for the model-owned CP path.

    Converts ``attention_mask`` to a ``padding_mask`` (preserving padding
    semantics for modules such as MoE), selects the primary sequence tensor,
    injects/normalizes ``position_ids``, and resolves ``labels`` (falling back
    to ``loss_mask``).
    """
    convert_attention_mask_to_padding_mask(batch)

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = batch[primary_key]
    seq_len = primary_seq_tensor.shape[1]

    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    elif "position_ids" in batch:
        position_ids = batch["position_ids"]
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = batch["position_ids"]
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch.get("labels")
    if labels is None and loss_mask is not None:
        labels = loss_mask
        loss_mask = None
    if labels is None:
        raise KeyError("Context parallelism requires `labels` in the batch, or labels passed as `loss_mask`.")

    return primary_key, seq_len, labels, position_ids, pos_seq_dim, loss_mask


def shard_batch_contiguous(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    pad_multiple: int = 1,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
):
    """Prepare and contiguously shard a batch for model-owned CP.

    Runs the shared pre-shard prep, pads the sequence to
    ``cp_size * max(pad_multiple, 2)``, then keeps one contiguous sequence
    slice per CP rank.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        tp_mesh: The tensor-parallel device (sub)mesh (or None).
        batch: The full-sequence batch; mutated and sharded in place.
        loss_mask: Optional per-token loss mask; used as labels when the batch
            has none, otherwise sharded alongside the batch.
        padding_token_id: Pad sentinel for ``input_ids``.
        pad_multiple: Required per-CP-rank shard length multiple (e.g. DSV4's
            compress-ratio LCM). The effective divisor is
            ``cp_size * max(pad_multiple, 2)``.
        extra_seq_keys: Model-specific per-token batch keys to pad and shard,
            mapped to their sequence dim (e.g. Gemma4 vision group ids).
        extra_pad_values: Pad sentinels for ``extra_seq_keys`` (default 0).

    Returns:
        ``(contextlib.nullcontext, batch)`` — transport lives in the model's
        own attention, so no CP context manager is needed.
    """
    primary_key, seq_len, labels, position_ids, pos_seq_dim, loss_mask = _prepare_manual_cp_batch(
        cp_mesh, tp_mesh, batch, loss_mask
    )
    return _make_contiguous_shard_cp_batch(
        cp_mesh,
        batch,
        primary_key=primary_key,
        seq_len=seq_len,
        labels=labels,
        position_ids=position_ids,
        pos_seq_dim=pos_seq_dim,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        pad_multiple=pad_multiple,
        extra_seq_keys=extra_seq_keys,
        extra_pad_values=extra_pad_values,
    )


def _make_contiguous_shard_cp_batch(
    cp_mesh,
    batch,
    *,
    primary_key,
    seq_len,
    labels,
    position_ids,
    pos_seq_dim,
    loss_mask,
    padding_token_id,
    pad_multiple: int = 1,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
):
    cp_size = cp_mesh.size()
    # Batch-resident sequence tensors, each sharded on seq dim 1 with its own pad
    # sentinel. ``position_ids``/``labels``/``loss_mask`` are sharded too but handled
    # separately below (special pad logic, or carried as locals rather than in the
    # batch dict during padding). This single table drives padding, slicing, and the
    # known-key set so a new field is declared in one place.
    seq_pad_values = {
        "input_ids": padding_token_id,
        "inputs_embeds": 0,
        "padding_mask": True,
    }
    known_sequence_keys = set(seq_pad_values) | {"labels", "position_ids", "loss_mask"}

    # Extra per-token metadata (e.g. Gemma4 vision group ids) is sharded like the
    # known sequence tensors, using model-provided seq dims / pad values. The
    # legacy batch-key spelling (`_cp_metadata_*`) is still honored for direct
    # callers; explicit arguments win on key collisions.
    metadata_seq_dims = {**batch.pop("_cp_metadata_seq_dims", {}), **(extra_seq_keys or {})}
    metadata_pad_values = {**batch.pop("_cp_metadata_pad_values", {}), **(extra_pad_values or {})}
    extra_metadata_keys = [key for key in metadata_seq_dims if key in batch and key not in known_sequence_keys]

    divisor = cp_size * max(int(pad_multiple or 1), 2)
    pad_len = (-seq_len) % divisor
    if pad_len:
        for key, pad_val in seq_pad_values.items():
            if key in batch:
                batch[key] = _pad_tensor_seq_dim_(batch[key], 1, pad_len, pad_val)
        labels = _pad_tensor_seq_dim_(labels, 1, pad_len, -100)
        position_ids = _pad_position_ids_seq_dim_(position_ids, pos_seq_dim, pad_len)
        batch["position_ids"] = position_ids
        if loss_mask is not None:
            loss_mask = _pad_tensor_seq_dim_(loss_mask, 1, pad_len, 0)
        for key in extra_metadata_keys:
            batch[key] = _pad_tensor_seq_dim_(
                batch[key],
                metadata_seq_dims[key],
                pad_len,
                metadata_pad_values.get(key, 0),
            )

    # Manual sequence slicing. Every CP rank in the same CP group starts from
    # the same full batch, then keeps one contiguous sequence shard.
    batch["labels"] = labels
    cp_rank = _cp_rank(cp_mesh)

    seq_len = batch[primary_key].shape[1]
    if seq_len % cp_size != 0:
        raise ValueError(f"CP sequence length must be divisible by cp_size after padding, got {seq_len=} {cp_size=}")
    local_seq_len = seq_len // cp_size
    seq_start = cp_rank * local_seq_len
    seq_end = seq_start + local_seq_len

    def _slice_seq(key: str, seq_dim: int = 1) -> None:
        if key not in batch:
            return
        slices = [slice(None)] * batch[key].ndim
        slices[seq_dim] = slice(seq_start, seq_end)
        batch[key] = batch[key][tuple(slices)].contiguous()

    for key in seq_pad_values:
        _slice_seq(key, 1)
    _slice_seq("labels", 1)
    _slice_seq("position_ids", pos_seq_dim)
    for key in extra_metadata_keys:
        _slice_seq(key, metadata_seq_dims[key])
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    return contextlib.nullcontext, batch


# ---------------------------------------------------------------------------
# Round-robin load-balanced shard batch implementation.
#
# The framework's default CP path: torch ``context_parallel`` shards every
# buffer into 2*cp head-tail-paired chunks (see ``round_robin_local_indices``)
# and the transport is ring SDPA installed by the returned context.
# ---------------------------------------------------------------------------


def shard_batch_load_balanced(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id: int = 0):
    """Shard a batch with torch ``context_parallel`` round-robin load balancing.

    ``CPSharder.shard_batch`` implementation for the default framework-owned CP
    path (layout ``"round_robin"``, indices from
    :func:`round_robin_local_indices`). Assumes an active CP mesh (size > 1).
    ``padding_token_id`` is accepted per the contract but unused: CP-pad slots
    are zero-filled, sit after every real token under the causal mask, and
    carry -100 labels.

    Returns:
        ``(ctx_factory, batch)`` where entering ``ctx_factory()`` installs the
        SDPA-kernel + ``context_parallel`` context for the forward.
    """
    # Call-time import: the torch-CP transport machinery stays in cp_utils
    # (NeMo-RL imports it from there), and cp_utils imports this module.
    from nemo_automodel.components.distributed.cp_utils import (  # noqa: PLC0415
        create_context_parallel_ctx,
        get_train_context,
    )

    # Remove attention_mask from the batch so the model does not attempt to
    # build a 4D causal mask (which would have mismatched shapes with
    # DTensor-sharded Q/K/V).  Each self_attn module's forward_pre_hook
    # (registered by attach_context_parallel_hooks) will set is_causal=True
    # so that SDPA handles causal masking internally.
    batch.pop("attention_mask", None)

    # Determine the primary sequence tensor: inputs_embeds (VLM with CP, where
    # multimodal token replacement happened pre-shard) or input_ids (standard LLM).
    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    if has_inputs_embeds:
        primary_seq_tensor = batch["inputs_embeds"]
    else:
        primary_seq_tensor = batch["input_ids"]
    seq_len = primary_seq_tensor.shape[1]

    # Skip 1D injection if position_ids already in batch (e.g. mRoPE pre-computed)
    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    else:
        position_ids = batch["position_ids"]
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = batch["position_ids"]

    # Determine correct seq dim for CP sharding
    # mRoPE: [3, B, S] → shard on dim 2; standard: [B, S] → shard on dim 1
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch["labels"]

    # Collect all available tensors for context parallel.  We track each
    # cp_buffer's batch key (when sourced from ``batch``) so the padding pass
    # below can pick the semantically-correct fill sentinel and mirror the
    # padded tensor back into ``batch``.  ``loss_mask`` is passed as an arg
    # (not in batch) so it has no key.
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    cp_buffers = [primary_seq_tensor, labels, position_ids]
    # inputs_embeds is [B, S, H] → seq_dim=1; input_ids is [B, S] → seq_dim=1
    cp_seq_dims = [1, 1, pos_seq_dim]
    cp_no_restore_buffers = {primary_seq_tensor, labels}
    batch_buffer_keys: dict[int, str] = {0: primary_key, 1: "labels", 2: "position_ids"}

    # Add loss_mask if available (passed as arg, not in batch -> no key)
    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    # Add padding_mask if available in batch
    if "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(padding_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(padding_mask)

    # Pad sequence length to be divisible by 2 * cp_size (required by
    # context_parallel load balancing). The inputs_embeds path can hit
    # arbitrary seq lengths from the VLM collator, so we pad here rather
    # than relying on dataset-side padding.
    #
    # Per-buffer pad sentinels: each tensor's "ignore" value is semantic, not
    # dtype-derived.  ``labels``/``padding_mask``/``attention_mask`` are all
    # int/bool but have different ignore conventions.  Falling through to 0
    # for ``padding_mask`` (== False == "real token") would tell the MoE
    # router to route the cp-pad slots to experts -- silently wasting capacity
    # and skewing load-balance loss.
    PAD_FILL = {
        "labels": -100,  # CE ignore_index
        "padding_mask": True,  # bool: True == "this position is pad, ignore"
        "attention_mask": False,  # HF: 0 == "this position is pad, ignore"
        # everything else (input_ids, position_ids, ...) -> 0
    }
    cp_divisor = cp_mesh.size() * 2
    if seq_len % cp_divisor != 0:
        pad_len = cp_divisor - (seq_len % cp_divisor)
        new_no_restore = set()
        for i, (buf, dim) in enumerate(zip(cp_buffers, cp_seq_dims)):
            pad_shape = list(buf.shape)
            pad_shape[dim] = pad_len
            if buf.dtype.is_floating_point:
                pad_val = torch.zeros(pad_shape, dtype=buf.dtype, device=buf.device)
            else:
                fill_val = PAD_FILL.get(batch_buffer_keys.get(i), 0)
                pad_val = torch.full(pad_shape, fill_val, dtype=buf.dtype, device=buf.device)
            old_buf = buf
            cp_buffers[i] = torch.cat([buf, pad_val], dim=dim)
            if old_buf in cp_no_restore_buffers:
                new_no_restore.add(cp_buffers[i])
        cp_no_restore_buffers = new_no_restore
        # Mirror every batch-sourced cp_buffer back into ``batch`` so any
        # downstream consumer reading from the dict sees the padded shape.
        for idx, key in batch_buffer_keys.items():
            batch[key] = cp_buffers[idx]

    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    # TODO(@akoumparouli): surface these in the future.
    enable_loss_parallel: bool = False
    enable_compiled_autograd: bool = False
    return get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_ctx), batch
