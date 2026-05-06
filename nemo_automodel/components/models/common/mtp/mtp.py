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

"""Model-agnostic Multi-Token Prediction (MTP) scaffolding.

The MTP module owns the depth-iteration loop, token rolling, fusion of the
previous-depth hidden state with the future-token embedding, and per-depth
loss computation. Model-specific construction of the inner decoder block is
delegated to the caller via a sublayer factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

# Layer-type symbols used by Megatron's hybrid layer specs (see
# ``megatron/core/models/hybrid/hybrid_layer_allocation.py``). HF's
# Nemotron-H ``configuration_nemotron_h.py`` uses the same characters in
# ``hybrid_override_pattern`` and ``mtp_hybrid_override_pattern``.
_SYMBOL_TO_BLOCK = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
    "E": "moe",
}


def parse_mtp_layer_pattern(pattern: str) -> list[str]:
    """Parse an MTP layer pattern (e.g. ``"*E"``) into a list of block types.

    Args:
        pattern: Pattern string using Megatron-style symbols ``M``, ``*``, ``-``, ``E``.

    Returns:
        List of block type names (``"mamba"``, ``"attention"``, ``"mlp"``, ``"moe"``).

    Raises:
        ValueError: If the pattern is empty or contains unknown symbols.
    """
    if not pattern:
        raise ValueError("MTP layer pattern is empty")
    blocks: list[str] = []
    for ch in pattern:
        if ch not in _SYMBOL_TO_BLOCK:
            raise ValueError(
                f"Unknown MTP layer symbol {ch!r} in pattern {pattern!r}; "
                f"valid symbols are {sorted(_SYMBOL_TO_BLOCK.keys())}"
            )
        blocks.append(_SYMBOL_TO_BLOCK[ch])
    return blocks


# ---------------------------------------------------------------------------
# Token rolling
# ---------------------------------------------------------------------------


def roll_tensor(t: torch.Tensor, shifts: int = -1, dim: int = -1) -> torch.Tensor:
    """Roll a tensor along ``dim`` by ``shifts`` and zero the wrapped slice.

    Matches Megatron's ``roll_tensor`` (single-GPU, non-CP, non-packed path)
    in ``multi_token_prediction.py``. Used to shift ``input_ids`` /
    ``position_ids`` / ``labels`` left by one position per MTP depth.

    Args:
        t: Input tensor.
        shifts: Number of positions to shift (negative = left shift).
        dim: Dimension to roll along.

    Returns:
        New tensor with the trailing ``|shifts|`` positions along ``dim``
        zero-filled (i.e. no real wrap-around).
    """
    rolled = torch.roll(t, shifts=shifts, dims=dim)
    if shifts == 0 or t.shape[dim] == 0:
        return rolled
    n = abs(shifts)
    if shifts < 0:
        idx = torch.arange(t.shape[dim] - n, t.shape[dim], device=t.device)
    else:
        idx = torch.arange(0, n, device=t.device)
    rolled = rolled.index_fill(dim, idx, 0)
    return rolled


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MTPConfig:
    """Runtime configuration for the MTP block.

    Attributes:
        num_layers: Number of MTP depths (D). ``0`` disables MTP.
        layer_pattern: Per-depth inner-block pattern, e.g. ``"*E"`` for one
            attention + one MoE sublayer per depth.
        loss_scaling_factor: Coefficient applied to the summed per-depth CE
            loss (Megatron's ``--mtp-loss-scaling-factor``, default 0.1).
            The effective per-depth weight is
            ``loss_scaling_factor / num_layers``.
    """

    num_layers: int = 0
    layer_pattern: str = ""
    loss_scaling_factor: float = 0.1

    @property
    def pattern_length(self) -> int:
        return len(self.layer_pattern)

    @property
    def total_sublayers(self) -> int:
        return self.num_layers * self.pattern_length

    @property
    def enabled(self) -> bool:
        return self.num_layers > 0 and self.pattern_length > 0


# ---------------------------------------------------------------------------
# MTP module
# ---------------------------------------------------------------------------


class MTPModule(nn.Module):
    """Multi-Token Prediction block.

    Holds a flat :class:`nn.ModuleList` of sublayers (length
    ``num_layers * pattern_length``) where the first sublayer of each depth
    carries the fusion modules (``enorm``, ``hnorm``, ``eh_proj``) and the
    last sublayer of each depth carries ``final_layernorm``. This flat layout
    matches the HuggingFace export format used by Nemotron-V3
    (``mtp.layers.{i}.*``).

    The model-specific sublayer construction (which decoder block to use, how
    to handle MoE / attention / Mamba) is delegated to the caller via
    ``sublayer_factory``.

    Args:
        mtp_config: :class:`MTPConfig` describing depth and pattern.
        sublayer_factory: Callable
            ``factory(global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm) -> nn.Module``
            constructing one sublayer. The returned module must be callable
            as ``sublayer(hidden_states, **kwargs) -> Tensor`` and, when
            ``has_fusion=True``, expose attributes ``enorm``, ``hnorm``,
            ``eh_proj``. When ``has_final_norm=True`` it must expose
            ``final_layernorm``.
    """

    def __init__(
        self,
        mtp_config: MTPConfig,
        sublayer_factory: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("MTPModule constructed with disabled MTPConfig")
        self.mtp_config = mtp_config
        block_types = parse_mtp_layer_pattern(mtp_config.layer_pattern)
        P = mtp_config.pattern_length
        D = mtp_config.num_layers
        layers: list[nn.Module] = []
        for d in range(D):
            for s in range(P):
                global_idx = d * P + s
                layers.append(
                    sublayer_factory(
                        global_idx=global_idx,
                        depth=d,
                        sublayer_idx=s,
                        block_type=block_types[s],
                        has_fusion=(s == 0),
                        has_final_norm=(s == P - 1),
                    )
                )
        self.layers = nn.ModuleList(layers)

    @property
    def num_depths(self) -> int:
        return self.mtp_config.num_layers

    @property
    def pattern_length(self) -> int:
        return self.mtp_config.pattern_length

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        hidden_states: torch.Tensor,
        embed_fn: Callable[[torch.LongTensor], torch.Tensor],
        **block_kwargs,
    ) -> list[torch.Tensor]:
        """Iterate over MTP depths and return per-depth hidden states.

        Args:
            input_ids: Token ids ``[B, S]`` (or ``[T]`` in THD). Rolled
                cumulatively left by 1 per depth.
            position_ids: Position ids matching ``input_ids``, or ``None``.
            hidden_states: Output of the main model's final norm (``h_0``);
                shape matches the model's residual stream.
            embed_fn: Callable applied to rolled ``input_ids`` to produce the
                future-token embedding (typically the model's input embedding
                layer).
            **block_kwargs: Forwarded to each sublayer's ``__call__`` (e.g.
                ``attention_mask``).

        Returns:
            List of length ``num_depths`` containing the hidden state
            produced at each depth.
        """
        D = self.num_depths
        P = self.pattern_length
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        cur_position_ids = position_ids
        for d in range(D):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            if cur_position_ids is not None:
                cur_position_ids = roll_tensor(cur_position_ids, shifts=-1, dim=-1)

            for s in range(P):
                sublayer = self.layers[d * P + s]
                if s == 0:
                    decoder_input = embed_fn(cur_input_ids)
                    e = sublayer.enorm(decoder_input)
                    h = sublayer.hnorm(hidden_states)
                    fused = torch.cat([e, h], dim=-1)
                    hidden_states = sublayer.eh_proj(fused)
                hidden_states = sublayer(hidden_states, **block_kwargs)
                if s == P - 1:
                    hidden_states = sublayer.final_layernorm(hidden_states)
            per_depth_h.append(hidden_states)
        return per_depth_h


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_mtp_loss(
    per_depth_hidden_states: list[torch.Tensor],
    labels: torch.LongTensor,
    lm_head: nn.Linear,
    loss_scaling_factor: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the cumulative MTP cross-entropy loss across depths.

    For depth ``k``: cumulatively roll ``labels`` left by ``k`` positions
    (mirroring the input-side roll), project ``h_k`` to logits via the shared
    ``lm_head``, and accumulate masked CE. Final loss is
    ``loss_scaling_factor / D * sum_k(CE_k)`` where ``D`` is the number of
    depths. Mirrors Megatron's ``process_mtp_loss``
    (``multi_token_prediction.py:615``).

    Args:
        per_depth_hidden_states: List of length ``D`` with the per-depth
            hidden states from :meth:`MTPModule.forward`.
        labels: Original (unshifted) labels, ``[B, S]`` or ``[T]``.
        lm_head: Shared output projection (typically ``model.lm_head``).
        loss_scaling_factor: Scalar multiplier on the summed loss (default
            ``0.1`` matches Megatron's default).
        ignore_index: Label value masked out of the CE loss.

    Returns:
        Scalar tensor with ``requires_grad=True`` (provided
        ``per_depth_hidden_states[*].requires_grad``).
    """
    D = len(per_depth_hidden_states)
    if D == 0:
        raise ValueError("per_depth_hidden_states must be non-empty")

    cur_labels = labels
    total = per_depth_hidden_states[0].new_zeros(())
    for k, h_k in enumerate(per_depth_hidden_states):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        # ``roll_tensor`` zero-fills wrapped positions with 0 (a valid token
        # id). Override those trailing positions with ``ignore_index`` so
        # they do not contribute to CE.
        masked_labels = cur_labels.clone()
        seq_dim = masked_labels.dim() - 1
        if masked_labels.shape[seq_dim] > 0:
            n_invalid = min(k + 1, masked_labels.shape[seq_dim])
            idx = torch.arange(
                masked_labels.shape[seq_dim] - n_invalid,
                masked_labels.shape[seq_dim],
                device=masked_labels.device,
            )
            masked_labels = masked_labels.index_fill(seq_dim, idx, ignore_index)

        logits_k = lm_head(h_k)
        loss_k = F.cross_entropy(
            logits_k.reshape(-1, logits_k.size(-1)),
            masked_labels.reshape(-1),
            ignore_index=ignore_index,
        )
        total = total + loss_k

    return total * (loss_scaling_factor / D)
