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

"""Context-parallel batch sharding and setup for Inkling.

Sharding is contiguous -- rank ``r`` owns global positions ``[r * L, (r + 1) * L)`` --
rather than the framework's default load-balanced round-robin layout. Two reasons: the
UPipe all-to-all reassembles the sequence in rank order, and the block short convolutions
will need contiguous neighbors once their halo exchange lands.

``attention_mask`` is deliberately left at full length. The short convolutions inside
attention run *after* the all-to-all, on the complete sequence, so they need the padding
map for positions this rank does not own. It is one bit per token, so replicating it
costs nothing next to the activations.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout

_SHARDED_KEYS = ("input_ids", "labels", "position_ids", "padding_mask")
_PAD_VALUES = {"input_ids": None, "labels": -100, "padding_mask": True, "attention_mask": 0}


def _pad_seq_dim(tensor: torch.Tensor, pad_len: int, value) -> torch.Tensor:
    """Right-pad a ``[batch, sequence, ...]`` tensor along the sequence axis."""
    if pad_len <= 0:
        return tensor
    pad = [0, 0] * (tensor.ndim - 2) + [0, pad_len]
    return F.pad(tensor, pad, value=value)


def shard_batch_for_inkling_cp(cp_mesh, tp_mesh, batch: dict, *, loss_mask=None, padding_token_id: int = 0):
    """Shard a batch contiguously across the context-parallel mesh for Inkling.

    Exposed through the :class:`ContextParallelSharder` returned by
    ``InklingForConditionalGeneration.prepare_model_inputs_for_cp``. Every rank starts
    from the same full batch and keeps its ``[seq_start, seq_end)`` slice of each
    sequence-aligned tensor, except ``attention_mask``, which stays full length so the
    post-all-to-all convolutions can see padding outside this rank's shard.

    Args:
        cp_mesh: One-dimensional context-parallel mesh, or ``None``.
        tp_mesh: Tensor-parallel mesh; unused, accepted for interface parity.
        batch: Batch mapping containing ``input_ids`` of shape ``[batch, sequence]``.
        loss_mask: Optional ``[batch, sequence]`` tensor sharded with the labels.
        padding_token_id: Token id used when padding ``input_ids``.

    Returns:
        tuple: ``(context_factory, batch, layout)``. The context factory is a null
        context because the transport lives in the Inkling attention modules themselves.
    """
    del tp_mesh
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    input_ids = batch["input_ids"]
    original_seq_len = input_ids.shape[1]
    seq_len = original_seq_len
    device = input_ids.device

    # A 4D or otherwise expanded mask cannot survive sharding; only the 2D padding map is
    # meaningful downstream, and attention masking is rebuilt from global indices anyway.
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim != 2:
        attention_mask = None
    for key in ("seq_lens", "seq_lens_padded", "cu_seqlens", "cu_seqlens_padded", "max_seqlen", "qkv_format"):
        batch.pop(key, None)

    pad_len = (-seq_len) % cp_size
    if pad_len:
        for key in _SHARDED_KEYS:
            if key in batch:
                value = padding_token_id if key == "input_ids" else _PAD_VALUES[key]
                batch[key] = _pad_seq_dim(batch[key], pad_len, value)
        if attention_mask is not None:
            attention_mask = _pad_seq_dim(attention_mask, pad_len, 0)
        if loss_mask is not None:
            loss_mask = _pad_seq_dim(loss_mask, pad_len, 0)
        seq_len += pad_len

    if attention_mask is None and pad_len:
        # Padding introduced positions with no real token; the convolutions must not mix
        # them into the tail of the sequence.
        attention_mask = torch.ones(input_ids.shape[0], seq_len, dtype=torch.bool, device=device)
        attention_mask[:, original_seq_len:] = False
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask
    else:
        batch.pop("attention_mask", None)

    if cp_size > 1:
        local_seq_len = seq_len // cp_size
        seq_start = cp_mesh.get_local_rank() * local_seq_len
        seq_end = seq_start + local_seq_len
        for key in _SHARDED_KEYS:
            if key in batch:
                batch[key] = batch[key][:, seq_start:seq_end].contiguous()
        if loss_mask is not None:
            batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()
    elif loss_mask is not None:
        batch["loss_mask"] = loss_mask

    layout = ShardLayout(original_seq_len=original_seq_len, padded_seq_len=seq_len)
    return contextlib.nullcontext, batch, layout


def setup_inkling_cp(model: torch.nn.Module, cp_mesh: Any) -> None:
    """Install UPipe attention and mark the block convolutions as sequence-sharded.

    Two kinds of short convolution live in an Inkling layer and they need opposite
    treatment. ``k_sconv`` / ``v_sconv`` run inside attention *after* the all-to-all, on
    the full sequence, and consume the full-length mask as-is. ``attn_sconv`` /
    ``mlp_sconv`` run on the residual stream, which stays sequence-sharded, so they are
    told their rank in order to slice the same full-length mask down to their own window.

    Idempotent: the parallelizer may also reach ``setup_cp_attention`` directly.
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    for layer in _text_layers(model):
        setup = getattr(layer.self_attn, "setup_cp_attention", None)
        if setup is not None:
            setup(cp_mesh)
        for name in ("attn_sconv", "mlp_sconv"):
            sconv = getattr(layer, name, None)
            if sconv is not None:
                sconv.set_cp_shard(cp_rank, cp_size)


def _text_layers(model: torch.nn.Module):
    """Yield the Inkling text decoder layers, tolerating the VLM and PP wrappers."""
    inner = getattr(model, "model", model)
    language_model = getattr(inner, "language_model", inner)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return []
    return layers.values() if hasattr(layers, "values") else layers
