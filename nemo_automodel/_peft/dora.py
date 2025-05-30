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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
from torch import nn


# -----------------------------------------------------------------------------#
# Helper: very small LoRA-style linear adapter (A ∘ B) with a weight-magnitude
# -----------------------------------------------------------------------------#
class LinearDoRAAdapter(nn.Module):
    """
    Low-rank adapter B  @  A  with an extra per-output weight_magnitude vector.
    Shapes:
        A: (rank, in_features)      – "lora_A"
        B: (out_features, rank)     – "lora_B"
        weight_magnitude: (out_features,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A and B initialisations follow common LoRA defaults
        self.linear_in = nn.Linear(in_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.linear_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_out.weight)

        # Per-output magnitude
        self.weight_magnitude = nn.Parameter(
            torch.ones(out_features, dtype=torch.get_default_dtype()),
            requires_grad=True,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    # --------------------------------------------------------------------- #
    # Forward   –  returns only the low-rank update (B A) x
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None and self.training:
            x = self.dropout(x)
        return self.linear_out(self.linear_in(x)) * self.scaling


# -----------------------------------------------------------------------------#
# Wrapper that “patches” an nn.Linear with DoRA logic from the paper
# -----------------------------------------------------------------------------#
class DoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear (`base_linear`) with a DoRA adapter (`adapter`).
    Implements Eq. (5) (and the dropout correction term) from the paper.
    """

    def __init__(self, base_linear: nn.Linear, adapter: LinearDoRAAdapter):
        super().__init__()
        self.base = base_linear
        self.adapter = adapter
        self._register_buffer("_cached_base_norm", self._get_weight_norm(), persistent=False)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def _get_weight_norm(self) -> torch.Tensor:
        """ || W₀ + B·A ||₍row₋wise₎"""
        with torch.no_grad():
            rank_update = self.adapter.scaling * (
                self.adapter.linear_out.weight @ self.adapter.linear_in.weight
            )
            merged = self.base.weight.data + rank_update
            return torch.linalg.norm(merged, dim=1).to(merged.dtype)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = self.adapter(x)

        # Row-wise magnitudes
        current_norm = self._get_weight_norm()          # || W₀ + B·A ||
        magnitude_scale = (self.adapter.weight_magnitude / current_norm).view(1, -1)

        if self.adapter.dropout is None or not self.training:
            dropout_correction = 0.0
        else:
            # (m/||·|| – 1) · W₀ · (drop(x) – x)
            dropped_x = self.adapter.dropout(x)
            correction_term = magnitude_scale - 1.0
            dropout_correction = correction_term * self.base(dropped_x - x)

        return magnitude_scale * (base_out + adapter_out) + dropout_correction


# -----------------------------------------------------------------------------#
# DoRA “transformer” utility — find Linears and wrap them
# -----------------------------------------------------------------------------#
@dataclass
class DoRA:
    """
    Apply DoRA to all named sub-modules whose name contains one of `target_modules`.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2"]
    )
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"  # kept for API parity

    # ------------------------------------------------------------------ #
    # Main entry: model = DoRA(...).apply(model)
    # ------------------------------------------------------------------ #
    def apply(self, model: nn.Module) -> nn.Module:
        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(tok in full_name for tok in self.target_modules):
                continue

            parent, child_name = self._parent_and_name(model, full_name)
            wrapped = DoRALinear(
                module,
                LinearDoRAAdapter(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                ),
            )
            setattr(parent, child_name, wrapped)
            print(f"[DoRA] wrapped → {full_name}")

        return model

    # ------------------------------------------------------------------ #
    # Helper to get parent module given dotted path
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parent_and_name(root: nn.Module, dotted_name: str):
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
