# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Contiguous-shard context-parallel batch sharding (model-owned CP path).

Contiguously shards the sequence across CP ranks (each rank keeps one
``seq_start:seq_end`` slice) so a model can run its own CP attention over the
shards (e.g. Gemma4's p2p ring FlexAttention). It performs no collective — the
transport (e.g. Gemma4's ring) lives entirely in the model's attention. This is the
batch-side counterpart of the ``run_cp_manual_attention`` seam, and the
non-load-balanced peer of the ``context_parallel`` and TE/THD batch shardings.

Selected by the ``_cp_manual`` batch flag and dispatched from
``cp_utils.make_cp_batch_and_ctx``.
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
    Models that key CP masks on something else (e.g. DeepSeek V4 uses
    ``position_ids``) never read the synthesized tensor and simply ignore it.

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
    # Extra per-token metadata (e.g. Gemma4 vision group ids) is sharded like the
    # known sequence tensors, using model-provided seq dims / pad values.
    metadata_seq_dims = batch.pop("_cp_metadata_seq_dims", {})
    metadata_pad_values = batch.pop("_cp_metadata_pad_values", {})
    known_sequence_keys = {
        "input_ids",
        "inputs_embeds",
        "labels",
        "position_ids",
        "mm_token_type_ids",
        "_packed_seq_ids",
        "per_layer_inputs",
        "padding_mask",
        "loss_mask",
    }
    extra_metadata_keys = [key for key in metadata_seq_dims if key in batch and key not in known_sequence_keys]
    pad_len = (-seq_len) % (2 * cp_size)
    if pad_len:
        if "input_ids" in batch:
            batch["input_ids"] = _pad_tensor_seq_dim_(batch["input_ids"], 1, pad_len, padding_token_id)
        if "inputs_embeds" in batch:
            batch["inputs_embeds"] = _pad_tensor_seq_dim_(batch["inputs_embeds"], 1, pad_len, 0)
        labels = _pad_tensor_seq_dim_(labels, 1, pad_len, -100)
        position_ids = _pad_position_ids_seq_dim_(position_ids, pos_seq_dim, pad_len)
        batch["position_ids"] = position_ids
        if "mm_token_type_ids" in batch:
            batch["mm_token_type_ids"] = _pad_tensor_seq_dim_(batch["mm_token_type_ids"], 1, pad_len, 0)
        if "_packed_seq_ids" in batch:
            batch["_packed_seq_ids"] = _pad_tensor_seq_dim_(batch["_packed_seq_ids"], 1, pad_len, 0)
        if "per_layer_inputs" in batch:
            batch["per_layer_inputs"] = _pad_tensor_seq_dim_(batch["per_layer_inputs"], 1, pad_len, 0)
        if loss_mask is not None:
            loss_mask = _pad_tensor_seq_dim_(loss_mask, 1, pad_len, 0)
        if "padding_mask" in batch:
            batch["padding_mask"] = _pad_tensor_seq_dim_(batch["padding_mask"], 1, pad_len, True)
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

    _slice_seq("input_ids", 1)
    _slice_seq("inputs_embeds", 1)
    _slice_seq("labels", 1)
    _slice_seq("position_ids", pos_seq_dim)
    _slice_seq("mm_token_type_ids", 1)
    _slice_seq("_packed_seq_ids", 1)
    _slice_seq("per_layer_inputs", 1)
    _slice_seq("padding_mask", 1)
    for key in extra_metadata_keys:
        _slice_seq(key, metadata_seq_dims[key])
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    return contextlib.nullcontext, batch
