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

"""Model-agnostic MTP scaffolding: depth iteration, token rolling, and loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


def roll_tensor(t: torch.Tensor, shifts: int = -1, dim: int = -1) -> torch.Tensor:
    """Roll a tensor along ``dim`` by ``shifts`` and zero the wrapped slice.

    Used to shift ``input_ids`` / ``position_ids`` / ``labels`` left by one
    position per MTP depth. Single-GPU path only (no CP / packed-sequence
    handling).

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


def get_mtp_loss_scaling_factor(model: nn.Module, default: float = 0.1) -> float:
    """Return the model's configured MTP auxiliary-loss scaling factor."""
    mtp_config = getattr(model, "mtp_config", None)
    if mtp_config is not None:
        return float(getattr(mtp_config, "loss_scaling_factor", default))
    return default


@dataclass
class MTPConfig:
    """Runtime configuration for the MTP block.

    Attributes:
        num_layers: Number of MTP forward iterations (D). ``0`` disables MTP.
            Equivalent to Megatron's ``--mtp-num-layers``.
        layer_pattern: Per-depth inner-block pattern, e.g. ``"*E"`` for one
            attention + one MoE sublayer per depth.
        loss_scaling_factor: Coefficient applied to the summed per-depth CE
            loss (default ``0.1``). The effective per-depth weight is
            ``loss_scaling_factor / num_layers``.
        use_repeated_layer: When ``True``, build a single physical depth's
            worth of sublayers and reuse it for all ``num_layers`` forward
            iterations (weight-tied across depths). Equivalent to Megatron's
            ``--mtp-use-repeated-layer``.
    """

    num_layers: int = 0
    layer_pattern: str = ""
    loss_scaling_factor: float = 0.1
    use_repeated_layer: bool = False

    @property
    def pattern_length(self) -> int:
        return len(self.layer_pattern)

    @property
    def num_physical_depths(self) -> int:
        return 1 if self.use_repeated_layer else self.num_layers

    @property
    def total_sublayers(self) -> int:
        return self.num_physical_depths * self.pattern_length

    @property
    def enabled(self) -> bool:
        return self.num_layers > 0 and self.pattern_length > 0


class MTPModule(nn.Module):
    """Multi-Token Prediction block.

    Holds a flat :class:`nn.ModuleList` of sublayers (length
    ``num_physical_depths * pattern_length``) where the first sublayer of
    each physical depth carries the fusion modules (``enorm``, ``hnorm``,
    ``eh_proj``) and the last sublayer of each physical depth carries
    ``final_layernorm``. This flat layout matches the HuggingFace export
    format used by Nemotron-V3 (``mtp.layers.{i}.*``).

    The model-specific sublayer construction (which decoder block to use, how
    to handle MoE / attention / Mamba) is delegated to the caller via
    ``sublayer_factory``.

    Args:
        mtp_config: :class:`MTPConfig` describing depth and pattern.
        block_types_per_sublayer: List of block-type strings (one per inner
            sublayer position), length must equal ``mtp_config.pattern_length``.
            Caller is responsible for parsing the model-specific symbol
            convention; this module does not interpret symbols.
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
        block_types_per_sublayer: list[str],
        sublayer_factory: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("MTPModule constructed with disabled MTPConfig")
        if len(block_types_per_sublayer) != mtp_config.pattern_length:
            raise ValueError(
                f"len(block_types_per_sublayer)={len(block_types_per_sublayer)} "
                f"!= mtp_config.pattern_length={mtp_config.pattern_length}"
            )
        self.mtp_config = mtp_config
        num_sublayers_per_depth = mtp_config.pattern_length
        num_physical_depths = mtp_config.num_physical_depths
        layers: list[nn.Module] = []
        for depth in range(num_physical_depths):
            for sublayer_idx in range(num_sublayers_per_depth):
                global_idx = depth * num_sublayers_per_depth + sublayer_idx
                layers.append(
                    sublayer_factory(
                        global_idx=global_idx,
                        depth=depth,
                        sublayer_idx=sublayer_idx,
                        block_type=block_types_per_sublayer[sublayer_idx],
                        has_fusion=(sublayer_idx == 0),
                        has_final_norm=(sublayer_idx == num_sublayers_per_depth - 1),
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
        hidden_states: torch.Tensor,
        embed_fn: Callable[[torch.LongTensor], torch.Tensor],
        position_ids: torch.LongTensor | None = None,
        **block_kwargs,
    ) -> list[torch.Tensor]:
        """Iterate over MTP depths and return per-depth hidden states.

        Args:
            input_ids: Token ids ``[B, S]`` (or ``[T]`` in THD). Rolled
                cumulatively left by 1 per depth.
            hidden_states: Output of the main model's final norm (``h_0``);
                shape matches the model's residual stream.
            embed_fn: Callable applied to rolled ``input_ids`` to produce the
                future-token embedding (typically the model's input embedding
                layer).
            position_ids: Position ids matching ``input_ids``. When supplied,
                rolled cumulatively per depth in lockstep with ``input_ids``
                (so slot ``t`` carries the original position of the rolled
                token) and forwarded to each sublayer via ``block_kwargs``.
                Required for RoPE-using sublayers; ignored by sublayers that
                don't consume it.
            **block_kwargs: Forwarded to each sublayer's ``__call__`` (e.g.
                ``attention_mask``).

        Returns:
            List of length ``num_depths`` containing the hidden state
            produced at each depth.
        """
        num_iterations = self.num_depths
        num_sublayers_per_depth = self.pattern_length
        use_repeated = self.mtp_config.use_repeated_layer
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        cur_position_ids = position_ids
        for depth in range(num_iterations):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            if cur_position_ids is not None:
                cur_position_ids = roll_tensor(cur_position_ids, shifts=-1, dim=-1)

            decoder_input = embed_fn(cur_input_ids)
            physical_depth = 0 if use_repeated else depth
            for sublayer_idx in range(num_sublayers_per_depth):
                sublayer = self.layers[physical_depth * num_sublayers_per_depth + sublayer_idx]
                kwargs = dict(block_kwargs)
                if cur_position_ids is not None:
                    kwargs["position_ids"] = cur_position_ids
                if sublayer_idx == 0:
                    kwargs["embed_input"] = decoder_input
                hidden_states = sublayer(hidden_states, **kwargs)
            per_depth_h.append(hidden_states)
        return per_depth_h
