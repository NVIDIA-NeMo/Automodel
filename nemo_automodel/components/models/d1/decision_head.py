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

"""D1 readouts with explicit option order and frozen vocabulary semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class D1TokenReadoutConfig:
    """Select answer-token rows without untying the released vocabulary head.

    Args:
        hidden_size: Width of the normalized final backbone states.
        token_ids: Unique answer-alias vocabulary IDs in adapter row order.
    """

    hidden_size: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or not self.token_ids:
            raise ValueError("hidden_size and token_ids must be non-empty and positive")
        if len(set(self.token_ids)) != len(self.token_ids) or min(self.token_ids) < 0:
            raise ValueError("token_ids must be unique non-negative vocabulary IDs")

    def build(self, vocabulary_weight: torch.Tensor) -> D1TokenReadout:
        """Build an adapter from the frozen released vocabulary weight.

        Args:
            vocabulary_weight: Tensor of shape [vocabulary, hidden_size].

        Returns:
            Zero-initialized row adapter, with detached base rows.
        """
        return D1TokenReadout(self, vocabulary_weight)


class D1TokenReadout(nn.Module):
    """Train additive answer-row updates while preserving input/output tying.

    This is an adaptation of the released readout, not an independent classifier.
    The zero-initialized adapter preserves the max-over-alias option contract.
    """

    def __init__(self, config: D1TokenReadoutConfig, vocabulary_weight: torch.Tensor) -> None:
        """Copy frozen vocabulary rows and initialize independent additive updates.

        Args:
            config: Answer-alias vocabulary row selection and hidden width.
            vocabulary_weight: Tensor of shape [vocab, hidden]. Selected base
                rows are detached and cloned; neither buffer nor trainable
                updates alias the original vocabulary weight.
        """
        super().__init__()
        self.config = config
        if vocabulary_weight.ndim != 2 or vocabulary_weight.shape[1] != config.hidden_size:
            raise ValueError("vocabulary_weight must have shape [vocabulary, hidden_size]")
        if max(config.token_ids) >= vocabulary_weight.shape[0]:
            raise ValueError("answer token ID exceeds vocabulary size")
        ids = torch.tensor(config.token_ids, device=vocabulary_weight.device, dtype=torch.long)
        self.register_buffer("token_ids", ids)
        self.register_buffer("base_weight", vocabulary_weight.detach().index_select(0, ids).clone())
        self.delta_weight = nn.Parameter(torch.zeros(self.base_weight.shape, device=ids.device, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, alias_groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        """Score declared options by their maximum answer-alias logit.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading
                dimensions, at each causal chain's answer position.
            alias_groups: Adapter row indices for each option, in target order.

        Returns:
            Float32 Tensor of shape [..., options], before option softmax.
        """
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must end in hidden_size")
        if not alias_groups or any(not group for group in alias_groups):
            raise ValueError("every option needs at least one answer alias")
        if any(min(group) < 0 or max(group) >= len(self.config.token_ids) for group in alias_groups):
            raise ValueError("alias_groups must index adapter rows")
        logits = F.linear(hidden_states.to(self.base_weight.dtype), self.base_weight).float()
        logits = logits + F.linear(hidden_states.float(), self.delta_weight.float())
        return torch.stack([logits[..., list(group)].amax(dim=-1) for group in alias_groups], dim=-1)


@dataclass(frozen=True)
class D1OmniHeadConfig:
    """Released Omni marker head configuration.

    Args:
        hidden_size: Encoder width, a positive multiple of the 64-wide head dimension.
        layers: Number of bidirectional transformer layers in the readout.
    """

    hidden_size: int
    layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % 64 or self.layers <= 0:
            raise ValueError("hidden_size must be a positive multiple of 64; layers must be positive")

    def build(self) -> D1OmniDecisionHead:
        """Build an uncalibrated, checkpoint-compatible Omni head."""
        return D1OmniDecisionHead(self)


class D1OmniDecisionHead(nn.Module):
    """Differentiable released Omni head; noul markers retain false/true order.

    State-dict names match the published checkpoint's head subtree. Text-only
    temperature calibration belongs outside this module so media paths can skip it.
    """

    def __init__(self, config: D1OmniHeadConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.type_emb = nn.Embedding(3, width)
        template = nn.TransformerEncoderLayer(
            width, width // 64, 4 * width, dropout=0.0, activation="relu", batch_first=True, norm_first=True
        )
        self.head = nn.TransformerEncoder(template, config.layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_tokens: torch.Tensor,
        marker_positions: torch.Tensor,
        valid_markers: torch.Tensor,
        question_types: torch.Tensor,
    ) -> torch.Tensor:
        """Score each option marker in internal checkpoint order.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden], excluding media prefixes.
            valid_tokens: Boolean Tensor of shape [batch, sequence], true for non-padding tokens.
            marker_positions: Long Tensor of shape [batch, options], indexing text positions.
            valid_markers: Boolean Tensor of shape [batch, options], true for declared options.
            question_types: Long Tensor of shape [batch] with choice=0, score=1, noul=2.

        Returns:
            Float32 Tensor of shape [batch, options], with padding options set to -10000.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [batch, text_length, hidden_size]")
        if valid_tokens.shape != hidden_states.shape[:2] or valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must be boolean with shape [batch, text_length]")
        if marker_positions.ndim != 2 or marker_positions.shape != valid_markers.shape:
            raise ValueError("marker_positions and valid_markers must have shape [batch, options]")
        if marker_positions.shape[0] != hidden_states.shape[0] or question_types.shape != hidden_states.shape[:1]:
            raise ValueError("markers and question_types must match the hidden-state batch")
        if valid_markers.dtype != torch.bool or marker_positions.dtype != torch.long:
            raise ValueError("valid_markers must be boolean and marker_positions must be long")
        values = hidden_states + self.type_emb(question_types).unsqueeze(1)
        for layer in self.head.layers:
            values = layer(values, src_key_padding_mask=~valid_tokens)
        indices = marker_positions.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        selected = values.gather(1, indices)
        return self.scorer(selected).squeeze(-1).float().masked_fill(~valid_markers, -10000.0)
