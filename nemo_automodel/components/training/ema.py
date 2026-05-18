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

"""Exponential moving average of a BAGEL model's parameters.

BAGEL's training recipe (and diffusion-family models in general) uses an EMA
copy of the model as the final saved checkpoint — averaging the noisy SGD
trajectory typically yields a model with measurably better generation quality
than the raw training endpoint. Upstream BAGEL uses ``decay=0.9999`` and
performs the update after every optimizer step (see
``train/fsdp_utils.py::fsdp_ema_update`` in upstream).

Update rule (per param, in-place):

    ema = decay * ema + (1 - decay) * train

This module provides the math; FSDP2-aware wiring (sharded-tensor walking,
checkpoint save/load through DCP) is layered on top by the recipe.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EMAManager:
    """Tracks an exponential moving average of ``model``'s trainable parameters.

    The shadow params are stored in a dict keyed by the same parameter names
    as ``model.named_parameters()``. Walking the model on each update means we
    don't pin to a specific param-list ordering; sharded / re-wrapped models
    work as long as the names are stable.

    Tensors are stored on the same device and dtype as the source params.
    """

    use_distributed_checkpointing = True

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self._shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = p.detach().clone()
            shadow.requires_grad = False
            self._shadow[name] = shadow

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one EMA update step using ``model``'s current parameters."""
        for name, p in model.named_parameters():
            if name not in self._shadow:
                continue
            shadow = self._shadow[name]
            # ema = decay * ema + (1 - decay) * p
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the shadow tensors keyed by param name. Caller-owned copies."""
        return {name: t.detach().clone() for name, t in self._shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load shadow tensors from ``state``. Shapes/dtypes must match."""
        if strict:
            extra = set(state) - set(self._shadow)
            missing = set(self._shadow) - set(state)
            if extra or missing:
                raise KeyError(f"EMAManager.load_state_dict: extra={sorted(extra)[:5]} missing={sorted(missing)[:5]}")
        for name in self._shadow:
            if name in state:
                self._shadow[name].copy_(state[name])

    def __len__(self) -> int:
        return len(self._shadow)

    def __contains__(self, name: str) -> bool:
        return name in self._shadow
