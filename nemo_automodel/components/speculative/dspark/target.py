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

"""Target-model wrapper for DSpark training (online hidden-state capture).

DSpark feeds the draft two things from the frozen target: the concatenation of a
configured set of decoder-layer hidden states (the draft ``fc`` context), and the
final post-norm hidden state (the input the target's ``lm_head`` consumes, used
by the TV / confidence losses). Both are captured in a single forward pass via
forward hooks, mirroring the DFlash target wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class DSparkTargetBatch:
    """Target-model features needed by the DSpark trainer."""

    target_hidden_states: torch.Tensor  # [B, S, len(target_layer_ids) * H]
    target_last_hidden_states: torch.Tensor  # [B, S, H]
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


class HFDSparkTargetModel:
    """Capture intermediate + final hidden states from a frozen HF causal LM.

    A forward hook on decoder layer ``i`` captures ``hidden_states[i + 1]`` (the
    HuggingFace ``output_hidden_states`` offset-1 convention); a hook on the final
    norm captures the post-norm last hidden state.
    """

    def __init__(self, model: nn.Module, target_layer_ids: Sequence[int]):
        self.model = model.eval()
        self.target_layer_ids = self._validate_layer_ids(target_layer_ids)

    def _validate_layer_ids(self, target_layer_ids: Sequence[int]) -> list[int]:
        num_layers = self.model.config.num_hidden_layers
        target_layer_ids = list(target_layer_ids)
        if len(target_layer_ids) == 0:
            raise ValueError("DSpark requires at least one target_layer_id.")
        for layer_id in target_layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"target layer id {layer_id} is out of bounds for model with {num_layers} layers")
        return target_layer_ids

    def _inner_model(self) -> nn.Module:
        """Return the base transformer module (the one owning ``layers`` and ``norm``)."""
        return self.model.model if hasattr(self.model, "model") else self.model

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Return decoder layers as an ordered, integer-indexable list."""
        inner = self._inner_model()
        if hasattr(inner, "layers"):
            container = inner.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            container = self.model.transformer.h
        else:
            raise ValueError("Unsupported model structure for DSpark hidden-state capture")
        if isinstance(container, nn.ModuleDict):
            return [container[str(i)] for i in range(len(container))]
        return list(container)

    def _get_final_norm(self) -> nn.Module:
        """Return the final norm module whose output feeds ``lm_head``."""
        inner = self._inner_model()
        norm = getattr(inner, "norm", None)
        if norm is None:
            raise ValueError("Could not locate the target's final norm for last-hidden capture")
        return norm

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Return the target model output embeddings (lm_head)."""
        return self.model.get_output_embeddings()

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DSparkTargetBatch:
        """Run the target model once and capture the DSpark context + last hidden state."""
        layers = self._get_transformer_layers()
        captured: dict[int, torch.Tensor] = {}
        last_hidden: dict[str, torch.Tensor] = {}
        handles = []

        def _make_layer_hook(layer_id: int):
            def _hook(_module, _inputs, outputs):
                captured[layer_id] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        def _norm_hook(_module, _inputs, outputs):
            last_hidden["h"] = outputs[0] if isinstance(outputs, tuple) else outputs

        for layer_id in self.target_layer_ids:
            handles.append(layers[layer_id].register_forward_hook(_make_layer_hook(layer_id)))
        handles.append(self._get_final_norm().register_forward_hook(_norm_hook))

        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(self.target_layer_ids) or "h" not in last_hidden:
            raise RuntimeError(
                f"Incomplete DSpark capture: layers {sorted(captured)} of {self.target_layer_ids}, "
                f"last_hidden={'h' in last_hidden}"
            )

        target_hidden_states = torch.cat([captured[layer_id] for layer_id in self.target_layer_ids], dim=-1)
        return DSparkTargetBatch(
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=last_hidden["h"],
            input_ids=input_ids,
            loss_mask=loss_mask,
        )


__all__ = ["HFDSparkTargetModel", "DSparkTargetBatch"]
