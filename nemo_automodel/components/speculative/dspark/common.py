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
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DSparkForwardOutput:
    """Outputs for one DSpark training forward.

    Shape symbols:
        batch_size: number of samples in the batch
        seq_len: source sequence length
        num_anchors: sampled anchor blocks per sample
        block_size: number of draft positions per anchor
        vocab_size: vocabulary size

    The sampler keeps anchors whose first draft target is enabled by
    ``loss_mask``. Later slots are supervised only while they remain inside
    ``seq_len`` and form a contiguous enabled prefix. Dummy anchors can still
    appear when a sample has too few valid anchors; they are masked out by
    ``block_keep_mask`` and ``eval_mask``.
    """

    # [batch_size, num_anchors, block_size, vocab_size]
    draft_logits: torch.Tensor
    # [batch_size, num_anchors, block_size]
    target_ids: torch.Tensor
    # [batch_size, num_anchors, block_size]
    eval_mask: torch.Tensor
    # [batch_size, num_anchors]
    block_keep_mask: torch.Tensor
    # [batch_size, num_anchors, block_size]
    confidence_pred: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size, vocab_size]
    aligned_target_logits: Optional[torch.Tensor] = None


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    assert layer_ids, "target_layer_ids must not be empty."
    start = 0
    end = int(num_target_layers) - 1
    previous = None
    for layer_id in layer_ids:
        assert layer_id == -1 or start <= layer_id <= end, (
            f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
            f"for num_target_layers={num_target_layers}. "
            "-1 denotes the embedding output."
        )
        assert previous is None or layer_id > previous, "target_layer_ids must be strictly increasing."
        previous = layer_id
    return layer_ids


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    return anchor_valid & first_target_valid


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    )
    valid_counts = valid.sum(dim=1)
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = (
        torch.arange(num_candidates, device=device)
        .unsqueeze(0)
        .expand(
            bsz,
            -1,
        )
    )
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    random_vals = torch.rand(bsz, num_candidates, device=device)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (valid_counts.unsqueeze(1).clamp(max=max_n))
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,
    safe_label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()


def create_position_ids(
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + offsets).view(
        bsz,
        num_blocks * block_size,
    )


def create_noise_embed(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device
    noise_ids = torch.full(
        (bsz, num_blocks * block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_starts = torch.arange(num_blocks, device=device) * block_size
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)
    flat_batch_idx = (
        torch.arange(bsz, device=device)
        .unsqueeze(1)
        .expand(
            bsz,
            num_blocks,
        )
    )
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        block_keep_mask,
        anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long, device=device),
    )
    return embed_tokens(noise_ids)


def pin_rope_inv_freq_fp32(rotary_emb: Optional[nn.Module]) -> None:
    """Keep a RoPE module's ``inv_freq`` buffer in fp32 after a dtype cast.

    ``module.to(bfloat16)`` (the training build path) rounds the rotary
    ``inv_freq`` buffer to bf16. The rounded frequencies dephase with absolute
    position, so the train/inference RoPE diverges (worse with longer context)
    and draft acceptance erodes, while the serving runtime keeps an fp32 RoPE
    cache. A bf16 round-trip cannot be undone by upcasting, so recompute fresh
    fp32 frequencies from the rotary config (the same values HF derives on the
    fp32 paths). No-op when ``inv_freq`` is already fp32 or on a meta device.
    """
    if rotary_emb is None:
        return
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None or not inv_freq.is_floating_point() or inv_freq.is_meta or inv_freq.dtype == torch.float32:
        return
    rope_init_fn = getattr(rotary_emb, "rope_init_fn", None)
    config = getattr(rotary_emb, "config", None)
    if rope_init_fn is None or config is None:
        return
    rope_kwargs = getattr(rotary_emb, "rope_kwargs", None) or {}
    fresh, _ = rope_init_fn(config, inv_freq.device, **rope_kwargs)
    rotary_emb.inv_freq = fresh.to(device=inv_freq.device, dtype=torch.float32)
    if getattr(rotary_emb, "original_inv_freq", None) is not None:
        rotary_emb.original_inv_freq = rotary_emb.inv_freq.clone()


__all__ = [
    "DSparkForwardOutput",
    "AcceptRatePredictor",
    "extract_context_feature",
    "validate_target_layer_ids",
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "build_eval_mask",
    "create_position_ids",
    "create_noise_embed",
    "pin_rope_inv_freq_fp32",
]
