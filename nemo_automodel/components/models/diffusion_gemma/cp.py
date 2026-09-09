# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context-parallel input layout for DiffusionGemma's two sequence streams."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    round_robin_local_indices,
    shard_token_tensor_by_indices,
)


def _ceil_multiple(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _pad(tensor: torch.Tensor, dim: int, length: int, value: int | float | bool) -> torch.Tensor:
    if tensor.shape[dim] >= length:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = length - tensor.shape[dim]
    tail = torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, tail), dim=dim)


def _pad_positions(position_ids: torch.Tensor, length: int) -> torch.Tensor:
    if position_ids.shape[1] >= length:
        return position_ids
    count = length - position_ids.shape[1]
    start = position_ids[:, -1:] + 1
    offsets = torch.arange(count, device=position_ids.device, dtype=position_ids.dtype)[None]
    return torch.cat((position_ids, start + offsets), dim=1)


def _pad_decoder_bias(mask: torch.Tensor, query_len: int, key_len: int, encoder_len: int) -> torch.Tensor:
    """Pad a global additive decoder bias while keeping pad query rows finite."""
    # TE P2P CP evaluates K/V blocks incrementally. A partial block may be fully
    # masked even though the complete attention row is not; dtype-min biases can
    # make that partial softmax produce NaNs. -1e4 still underflows masked
    # probabilities to zero in FP16/BF16 while keeping every partial block finite.
    neg = -1.0e4
    padded = torch.full((*mask.shape[:-2], query_len, key_len), neg, dtype=mask.dtype, device=mask.device)
    padded[..., : mask.shape[-2], : mask.shape[-1]] = mask.clamp_min(neg)
    rows = torch.arange(query_len, device=mask.device)
    padded[..., rows, encoder_len + rows] = 0
    return padded


def shard_diffusion_gemma_batch(
    cp_mesh,
    tp_mesh,
    batch: dict[str, Any],
    *,
    loss_mask=None,
    padding_token_id: int = 0,
):
    """Shard encoder-KV and decoder-query streams with independent TE layouts.

    The encoder is extended with one dummy slot per padded canvas token. During
    decoder attention those slots are replaced with the decoder layer's actual
    K/V. Thus TE sees one ordinary CP-partitioned cross-attention K/V stream
    ``[encoder ; canvas]`` without replicating the 100K+ encoder cache.
    """
    del tp_mesh, loss_mask
    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    divisor = 2 * cp_size

    input_ids = batch["input_ids"]
    canvas_ids = batch["canvas_ids"]
    encoder_len = input_ids.shape[1]
    canvas_len = canvas_ids.shape[1]
    canvas_padded_len = _ceil_multiple(canvas_len, divisor)
    combined_padded_len = _ceil_multiple(encoder_len + canvas_padded_len, divisor)

    device = input_ids.device
    encoder_indices = round_robin_local_indices(cp_mesh, combined_padded_len, device=device)
    canvas_indices = round_robin_local_indices(cp_mesh, canvas_padded_len, device=device)

    # Extended encoder layout: real clean tokens, replaceable canvas slots, tail pad.
    extended_ids = _pad(input_ids, 1, combined_padded_len, padding_token_id)
    encoder_positions = batch["encoder_position_ids"]
    decoder_positions = _pad_positions(batch["decoder_position_ids"], canvas_padded_len)
    extended_positions = torch.cat((encoder_positions, decoder_positions), dim=1)
    extended_positions = _pad_positions(extended_positions, combined_padded_len)

    encoder_padding = batch.get("encoder_padding_mask")
    if encoder_padding is None:
        encoder_padding = torch.zeros_like(input_ids, dtype=torch.bool)
    encoder_padding = _pad(encoder_padding, 1, combined_padded_len, True)

    batch["input_ids"] = shard_token_tensor_by_indices(extended_ids, encoder_indices)
    batch["encoder_position_ids"] = shard_token_tensor_by_indices(extended_positions, encoder_indices)
    batch["encoder_padding_mask"] = shard_token_tensor_by_indices(encoder_padding, encoder_indices)

    batch["canvas_ids"] = shard_token_tensor_by_indices(
        _pad(canvas_ids, 1, canvas_padded_len, padding_token_id), canvas_indices
    )
    batch["decoder_position_ids"] = shard_token_tensor_by_indices(decoder_positions, canvas_indices)
    decoder_padding = batch.get("decoder_padding_mask")
    if decoder_padding is None:
        decoder_padding = torch.zeros_like(canvas_ids, dtype=torch.bool)
    batch["decoder_padding_mask"] = shard_token_tensor_by_indices(
        _pad(decoder_padding, 1, canvas_padded_len, True), canvas_indices
    )

    labels = batch.get("encoder_labels")
    if labels is not None:
        labels = _pad(labels, 1, combined_padded_len, -100)
        batch["encoder_labels"] = shard_token_tensor_by_indices(labels, encoder_indices)

    masks = batch["decoder_attention_mask"]
    # TE p2p CP consumes local query rows but the global K/V bias axis; its ring
    # slices the latter as K/V blocks circulate.
    batch["decoder_attention_mask"] = {
        name: _pad_decoder_bias(mask, canvas_padded_len, combined_padded_len, encoder_len).index_select(
            -2, canvas_indices
        )
        for name, mask in masks.items()
    }

    # Tensor metadata survives filter_forward_kwargs and is safe under compile.
    batch["cp_encoder_indices"] = encoder_indices
    batch["cp_canvas_indices"] = canvas_indices
    batch["cp_encoder_length"] = encoder_len
    batch["cp_canvas_padded_length"] = canvas_padded_len

    layout = ShardLayout(
        local_token_global_indices=canvas_indices,
        original_seq_len=canvas_len,
        padded_seq_len=canvas_padded_len,
    )
    return contextlib.nullcontext, batch, layout


def diffusion_gemma_cp_sharder() -> ContextParallelSharder:
    """Return the model-owned sharder selected by CP dispatch."""
    return ContextParallelSharder(
        shard_batch=shard_diffusion_gemma_batch,
        local_token_global_indices=round_robin_local_indices,
    )
