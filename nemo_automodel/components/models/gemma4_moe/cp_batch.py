# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Gemma4's contiguous-shard context-parallel batch sharding.

Contiguously shards the sequence across CP ranks (each rank keeps one
``seq_start:seq_end`` slice) so Gemma4 can run its own p2p ring FlexAttention
over the shards. It performs no collective -- the transport lives in Gemma4's
attention (see ``cp_attention.py``); this is its batch-side counterpart, and the
non-load-balanced peer of the ``context_parallel`` and TE/THD batch shardings.

Gemma4's ``prepare_model_inputs_for_cp`` attaches
``make_contiguous_shard_cp_batch_and_ctx`` to the batch as ``_cp_make_batch_fn``;
``cp_utils.make_cp_batch_and_ctx`` then invokes that callable (model-agnostically)
in place of the default load-balanced ``context_parallel`` path.
"""

import contextlib

import torch


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


def _synthesize_single_document_seq_ids(batch: dict, primary_key: str, seq_len: int) -> None:
    """Materialize the trivial single-document ``_packed_seq_ids`` map for the manual CP path.

    The VLM/LLM collates emit ``_packed_seq_ids`` only when 2+ documents are
    packed (``attention_mask.max() > 1``), so a single, unpacked sequence arrives
    without it. The manual CP attention mask builder needs document boundaries
    even for one document, so synthesize the trivial map here (1 = real token, 0 =
    pad) instead of lowering each collate's threshold -- which would change
    behavior for every non-CP ``_packed_seq_ids`` consumer (e.g. SqrtCrossEntropy).
    Derived from ``padding_mask`` when present, else all-ones.

    A no-op when ``_packed_seq_ids`` is already present (genuinely packed input).

    Args:
        batch: The CP batch dict; mutated in place to add ``_packed_seq_ids``.
        primary_key: ``"input_ids"`` or ``"inputs_embeds"`` (selects batch / device).
        seq_len: The pre-pad sequence length.
    """
    if "_packed_seq_ids" in batch:
        return
    primary = batch[primary_key]
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        # padding_mask is [B, S], True == pad -> real-token map is its inverse.
        batch["_packed_seq_ids"] = (~padding_mask.bool()).to(torch.long)
    else:
        batch["_packed_seq_ids"] = torch.ones((primary.shape[0], seq_len), dtype=torch.long, device=primary.device)


def _prepare_manual_cp_batch(cp_mesh, tp_mesh, batch, loss_mask):
    """Pre-shard prep for the model-owned CP path.

    Kept here (rather than in ``cp_utils.make_cp_batch_and_ctx``) so that
    function's default, load-balanced path stays identical to upstream. Converts
    ``attention_mask`` to a ``padding_mask`` (preserving padding semantics for
    modules such as MoE), selects the primary sequence tensor, injects/normalizes
    ``position_ids``, and resolves ``labels`` (falling back to ``loss_mask``).
    """
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is not None and "padding_mask" not in batch:
        mask_for_padding = attention_mask.get("full_attention") if isinstance(attention_mask, dict) else attention_mask
        if mask_for_padding is not None and mask_for_padding.ndim == 4:
            diagonal = torch.diagonal(mask_for_padding[:, 0], dim1=-2, dim2=-1)
            batch["padding_mask"] = diagonal.logical_not() if mask_for_padding.dtype == torch.bool else diagonal != 0
        elif mask_for_padding is not None:
            batch["padding_mask"] = mask_for_padding.bool().logical_not()

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = batch[primary_key]
    seq_len = primary_seq_tensor.shape[1]

    def _mesh_size(mesh):
        return 0 if mesh is None else mesh.size()

    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch and (_mesh_size(cp_mesh) > 1 or _mesh_size(tp_mesh) > 1):
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


def make_contiguous_shard_cp_batch_and_ctx(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
    """Prepare and contiguously shard a batch for Gemma4's ring CP.

    Gemma4 attaches this callable to the batch (as ``_cp_make_batch_fn``) in its
    pre-embed; ``cp_utils.make_cp_batch_and_ctx`` invokes it. Runs the shared
    pre-shard prep, then keeps one contiguous sequence slice per CP rank (no
    collective; the transport lives in Gemma4's own ring attention).
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
):
    cp_size = cp_mesh.size()
    # The manual CP attention mask builder needs per-document boundaries
    # (`_packed_seq_ids`) even for a single sequence; the collates emit it only
    # for 2+ packed docs, so synthesize the trivial one-document map here.
    _synthesize_single_document_seq_ids(batch, primary_key, seq_len)

    # Batch-resident sequence tensors, each sharded on seq dim 1 with its own pad
    # sentinel. ``position_ids``/``labels``/``loss_mask`` are sharded too but handled
    # separately below (special pad logic, or carried as locals rather than in the
    # batch dict during padding). This single table drives padding, slicing, and the
    # known-key set so a new field is declared in one place.
    seq_pad_values = {
        "input_ids": padding_token_id,
        "inputs_embeds": 0,
        "mm_token_type_ids": 0,
        "_packed_seq_ids": 0,
        "per_layer_inputs": 0,
        "padding_mask": True,
    }
    known_sequence_keys = set(seq_pad_values) | {"labels", "position_ids", "loss_mask"}

    # Extra per-token metadata (e.g. Gemma4 vision group ids) is sharded like the
    # known sequence tensors, using model-provided seq dims / pad values.
    metadata_seq_dims = batch.pop("_cp_metadata_seq_dims", {})
    metadata_pad_values = batch.pop("_cp_metadata_pad_values", {})
    extra_metadata_keys = [key for key in metadata_seq_dims if key in batch and key not in known_sequence_keys]

    pad_len = (-seq_len) % (2 * cp_size)
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
    else:
        cp_rank = getattr(cp_mesh, "get_local_rank", lambda: 0)()

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
