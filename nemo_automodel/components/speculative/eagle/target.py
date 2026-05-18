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

"""Target-model wrapper for minimal EAGLE-3 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail.

    This matches the reference EAGLE-3 target preparation used by SpecForge:
    sequence-aligned tensors are shifted with ``padding(..., left=False)``.
    See SpecForge ``eagle3_target_model.py`` around the target preparation
    logic referenced by the user.
    """
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


@dataclass
class Eagle3TargetBatch:
    """Target-model outputs needed by the draft trainer."""

    aux_hidden_states: torch.Tensor
    logits: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class HFEagle3TargetModel:
    """Thin wrapper that captures three auxiliary hidden states from a causal LM."""

    def __init__(self, model: nn.Module, aux_layer_ids: Sequence[int] | None = None):
        self.model = model.eval()
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids is not None else self._default_aux_layer_ids()

    def _default_aux_layer_ids(self) -> list[int]:
        # Three reference points: an early layer, a middle one, and a late
        # one. ``max(1, ...)`` keeps the indices in-bounds for shallow
        # toy models, but the raw recipe collides on small ``num_layers``
        # (e.g. 5 layers -> ``[1, 1, 1]``); deduplicate preserving order
        # so ``generate_batch`` cannot end up registering multiple hooks
        # on the same module and tripping the post-forward length check.
        num_layers = self.model.config.num_hidden_layers
        candidates = [1, max(1, num_layers // 2 - 1), max(1, num_layers - 4)]
        seen: set[int] = set()
        unique: list[int] = []
        for lid in candidates:
            if lid not in seen:
                unique.append(lid)
                seen.add(lid)
        return unique

    def _get_transformer_layers(self) -> Iterable[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise ValueError("Unsupported model structure for EAGLE-3 aux-layer capture")

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetBatch:
        """Run the target model and capture aux hidden states plus logits."""
        layers = self._get_transformer_layers()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_id: int):
            def _hook(_module, _inputs, outputs):
                captured[layer_id] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        for layer_id in self.aux_layer_ids:
            if layer_id < 0 or layer_id >= len(layers):
                raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {len(layers)} layers")
            handles.append(layers[layer_id].register_forward_hook(_make_hook(layer_id)))

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(self.aux_layer_ids):
            raise RuntimeError(
                f"Expected {len(self.aux_layer_ids)} captured aux layers but got {len(captured)}: {sorted(captured)}"
            )

        aux_hidden_states = torch.cat([captured[layer_id] for layer_id in self.aux_layer_ids], dim=-1)
        shifted_logits = _shift_left_with_zero(outputs.logits)
        shifted_input_ids = _shift_left_with_zero(input_ids)
        shifted_loss_mask = _shift_left_with_zero(loss_mask)
        return Eagle3TargetBatch(
            aux_hidden_states=aux_hidden_states,
            logits=shifted_logits,
            input_ids=shifted_input_ids,
            attention_mask=attention_mask,
            loss_mask=shifted_loss_mask,
        )
