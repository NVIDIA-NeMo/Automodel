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

"""Model-neutral contiguous context-parallel batch sharding."""

from __future__ import annotations

import contextlib
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.shared.cp_contracts import ContextFactory, CPBatch


def _batch_tensor(batch: CPBatch, key: str) -> torch.Tensor:
    """Read one tensor-valued batch field.

    Args:
        batch: Mapping whose tensor fields may use arbitrary model-owned
            layouts; this helper does not reshape or mutate them.
        key: Field name to resolve.

    Returns:
        The exact tensor object stored at ``key`` with shape/dtype unchanged.
    """
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"CP batch field {key!r} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _cp_rank(cp_mesh: DeviceMesh) -> int:
    """Resolve this rank's index within ``cp_mesh``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=cp_mesh.get_group())
    return cp_mesh.get_local_rank()


def _pad_tensor_seq_dim(
    tensor: torch.Tensor,
    seq_dim: int,
    pad_len: int,
    value: float | int = 0,
) -> torch.Tensor:
    """Right-pad one tensor sequence axis.

    Args:
        tensor: Tensor with arbitrary rank and sequence axis ``S`` at
            ``seq_dim``.
        seq_dim: Axis to extend.
        pad_len: Number of positions to append.
        value: Scalar fill value converted to ``tensor`` dtype and device.

    Returns:
        ``tensor`` itself when ``pad_len <= 0``; otherwise a new contiguous-by-
        construction tensor with sequence length ``S + pad_len``.
    """
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids_seq_dim(
    position_ids: torch.Tensor,
    seq_dim: int,
    pad_len: int,
) -> torch.Tensor:
    """Right-pad position IDs with monotonically increasing positions.

    Args:
        position_ids: Standard positions ``[B, S]`` or mRoPE positions
            ``[R, B, S]`` where ``R`` is the rotary-axis count.
        seq_dim: Sequence axis: ``1`` for standard positions or ``2`` for
            mRoPE positions.
        pad_len: Number of positions to append.

    Returns:
        Original tensor when no padding is requested; otherwise a new tensor
        with ``pad_len`` monotonically increasing positions appended.
    """
    if pad_len <= 0:
        return position_ids
    last_position = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    increment_shape = [1] * position_ids.ndim
    increment_shape[seq_dim] = pad_len
    increments = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(
        increment_shape
    )
    return torch.cat((position_ids, last_position + increments), dim=seq_dim)


def convert_attention_mask_to_padding_mask(batch: CPBatch) -> None:
    """Replace an attention mask with a token padding mask in place.

    Args:
        batch: Batch containing an optional attention mask ``[B, S]`` or
            broadcast mask ``[B, 1, S, S]``. Boolean masks use ``True`` for
            visible positions; numeric broadcast masks use zero for visible
            positions. The emitted ``padding_mask`` is ``[B, S]`` with
            ``True`` for padding. Here ``B`` is batch size and ``S`` is
            sequence length.

    Returns:
        ``None``. ``batch`` is mutated by removing ``attention_mask`` and, when
        absent, adding ``padding_mask``.
    """
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is None or "padding_mask" in batch:
        return
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
        batch["padding_mask"] = diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
    else:
        batch["padding_mask"] = attention_mask.bool().logical_not()


def _prepare_contiguous_cp_batch(
    batch: CPBatch,
    loss_mask: torch.Tensor | None,
) -> tuple[str, int, torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    """Normalize a full batch before contiguous CP sharding.

    Args:
        batch: Full batch containing exactly one of token IDs ``[B, S]`` or
            embeddings ``[B, S, H]`` and optional positions/masks. It is
            mutated to normalize masks and positions. Here ``B`` is batch
            size, ``S`` is sequence length, and ``H`` is hidden size.
        loss_mask: Optional token mask ``[B, S]`` used as labels only when the
            batch has no labels.

    Returns:
        Primary key, global sequence length ``S``, labels ``[B, S]``, positions
        ``[B, S]`` or ``[R, B, S]``, position sequence axis, and remaining loss
        mask.
    """
    convert_attention_mask_to_padding_mask(batch)

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    if has_inputs_embeds == has_input_ids:
        raise ValueError("CP batch requires exactly one of 'inputs_embeds' or 'input_ids'")
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = _batch_tensor(batch, primary_key)
    seq_len = primary_seq_tensor.shape[1]

    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    else:
        position_ids = _batch_tensor(batch, "position_ids")
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = _batch_tensor(batch, "position_ids")
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels_value = batch.get("labels")
    labels = labels_value if isinstance(labels_value, torch.Tensor) else None
    if labels is None and loss_mask is not None:
        labels, loss_mask = loss_mask, None
    if labels is None:
        raise KeyError("Context parallelism requires `labels` in the batch, or labels passed as `loss_mask`.")

    return primary_key, seq_len, labels, position_ids, pos_seq_dim, loss_mask


def shard_batch_contiguous(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: CPBatch,
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    pad_multiple: int = 1,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, object] | None = None,
) -> tuple[ContextFactory, CPBatch]:
    """Pad and contiguously shard a full model-owned CP batch.

    Args:
        cp_mesh: Active CP mesh. Rank ``r`` receives one contiguous
            ``S_local = S_padded / CP`` slice.
        tp_mesh: Optional TP mesh; unused but retained by the sharder contract.
        batch: Full batch containing IDs ``[B, S]`` or embeddings ``[B, S, H]``,
            labels ``[B, S]``, and optional token-aligned tensors. It is mutated
            in place to contain local tensors. Here ``B`` is batch size, ``S``
            is global sequence length, and ``H`` is hidden size.
        loss_mask: Optional token mask ``[B, S]``.
        padding_token_id: Fill value for token IDs.
        pad_multiple: Required multiple of each local sequence length.
        extra_seq_keys: Model-specific tensor keys mapped to their sequence axes.
        extra_pad_values: Scalar fills for model-specific keys.

    Returns:
        ``(nullcontext, batch)`` with token tensors padded and replaced by
        contiguous local shards. The returned batch aliases the input mapping.
    """
    del tp_mesh
    if cp_mesh is None:
        raise ValueError("contiguous CP sharding requires a context-parallel mesh")
    primary_key, seq_len, labels, position_ids, pos_seq_dim, loss_mask = _prepare_contiguous_cp_batch(batch, loss_mask)
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
    cp_mesh: DeviceMesh,
    batch: CPBatch,
    *,
    primary_key: str,
    seq_len: int,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    pos_seq_dim: int,
    loss_mask: torch.Tensor | None,
    padding_token_id: int,
    pad_multiple: int = 1,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, object] | None = None,
) -> tuple[ContextFactory, CPBatch]:
    """Apply padding and one contiguous rank slice to normalized tensors.

    Args:
        cp_mesh: Active CP mesh.
        batch: Normalized full batch mutated in place. Tensor fields use batch
            size ``B``, global sequence length ``S``, and optional hidden size
            ``H`` as documented by :func:`shard_batch_contiguous`.
        primary_key: ``input_ids`` or ``inputs_embeds``.
        seq_len: Global pre-pad sequence length ``S``.
        labels: Labels ``[B, S]``.
        position_ids: Positions ``[B, S]`` or ``[R, B, S]``.
        pos_seq_dim: Position sequence axis.
        loss_mask: Optional token mask ``[B, S]``.
        padding_token_id: Fill value for token IDs.
        pad_multiple: Required local-sequence multiple.
        extra_seq_keys: Additional tensor keys and sequence axes.
        extra_pad_values: Scalar fills for additional tensors.

    Returns:
        ``(nullcontext, batch)`` with local contiguous tensors. The mapping is
        the same object passed by the caller.
    """
    cp_size = cp_mesh.size()
    seq_pad_values: dict[str, object] = {
        "input_ids": padding_token_id,
        "inputs_embeds": 0,
        "padding_mask": True,
    }
    known_sequence_keys = set(seq_pad_values) | {"labels", "position_ids", "loss_mask"}
    metadata_seq_dims = dict(extra_seq_keys or {})
    metadata_pad_values = dict(extra_pad_values or {})
    extra_metadata_keys = [key for key in metadata_seq_dims if key in batch and key not in known_sequence_keys]

    divisor = cp_size * max(int(pad_multiple or 1), 2)
    pad_len = (-seq_len) % divisor
    if pad_len:
        for key, pad_value in seq_pad_values.items():
            if key in batch:
                batch[key] = _pad_tensor_seq_dim(_batch_tensor(batch, key), 1, pad_len, cast(float | int, pad_value))
        labels = _pad_tensor_seq_dim(labels, 1, pad_len, -100)
        position_ids = _pad_position_ids_seq_dim(position_ids, pos_seq_dim, pad_len)
        batch["position_ids"] = position_ids
        if loss_mask is not None:
            loss_mask = _pad_tensor_seq_dim(loss_mask, 1, pad_len, 0)
        for key in extra_metadata_keys:
            batch[key] = _pad_tensor_seq_dim(
                _batch_tensor(batch, key),
                metadata_seq_dims[key],
                pad_len,
                cast(float | int, metadata_pad_values.get(key, 0)),
            )

    batch["labels"] = labels
    cp_rank = _cp_rank(cp_mesh)
    padded_seq_len = _batch_tensor(batch, primary_key).shape[1]
    if padded_seq_len % cp_size != 0:
        raise ValueError(
            f"CP sequence length must be divisible by cp_size after padding, got {padded_seq_len=} {cp_size=}"
        )
    local_seq_len = padded_seq_len // cp_size
    seq_start = cp_rank * local_seq_len
    seq_end = seq_start + local_seq_len

    def _slice_seq(key: str, seq_dim: int = 1) -> None:
        """Replace one token tensor by its contiguous local sequence slice."""
        if key not in batch:
            return
        tensor = _batch_tensor(batch, key)
        slices = [slice(None)] * tensor.ndim
        slices[seq_dim] = slice(seq_start, seq_end)
        batch[key] = tensor[tuple(slices)].contiguous()

    for key in seq_pad_values:
        _slice_seq(key)
    _slice_seq("labels")
    _slice_seq("position_ids", pos_seq_dim)
    for key in extra_metadata_keys:
        _slice_seq(key, metadata_seq_dims[key])
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    return contextlib.nullcontext, batch
