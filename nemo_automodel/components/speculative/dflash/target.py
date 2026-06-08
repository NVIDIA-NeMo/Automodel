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

"""Target-model wrapper for DFlash training.

Unlike EAGLE-3 (which captures exactly three aux layers and left-shifts the
supervision), DFlash captures an arbitrary set of decoder layers, concatenates
them along the feature dim, and feeds the result to the draft as *context*. No
shifting is applied -- the DFlash block attention mask handles anchor alignment.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class DFlashTargetBatch:
    """Target-model context features needed by the DFlash trainer."""

    hidden_states: torch.Tensor  # [B, S, len(target_layer_ids) * H]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class HFDFlashTargetModel:
    """Capture a set of decoder-layer hidden states from a frozen HF causal LM.

    A forward hook on decoder layer ``i`` captures that layer's output, which in
    HuggingFace's ``output_hidden_states`` convention is ``hidden_states[i + 1]``
    -- matching SpecForge's ``extract_context_feature`` (offset 1).
    """

    def __init__(self, model: nn.Module, target_layer_ids: Sequence[int]):
        self.model = model.eval()
        self.target_layer_ids = self._validate_layer_ids(target_layer_ids)

    def _validate_layer_ids(self, target_layer_ids: Sequence[int]) -> list[int]:
        num_layers = self.model.config.num_hidden_layers
        target_layer_ids = list(target_layer_ids)
        if len(target_layer_ids) == 0:
            raise ValueError("DFlash requires at least one target_layer_id.")
        for layer_id in target_layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"target layer id {layer_id} is out of bounds for model with {num_layers} layers")
        return target_layer_ids

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Return decoder layers as an ordered, integer-indexable list."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            container = self.model.model.layers
        elif hasattr(self.model, "layers"):
            container = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            container = self.model.transformer.h
        else:
            raise ValueError("Unsupported model structure for DFlash hidden-state capture")
        if isinstance(container, nn.ModuleDict):
            return [container[str(i)] for i in range(len(container))]
        return list(container)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetBatch:
        """Run the target model and capture the selected layers' hidden states as context."""
        layers = self._get_transformer_layers()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_id: int):
            def _hook(_module, _inputs, outputs):
                captured[layer_id] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        for layer_id in self.target_layer_ids:
            handles.append(layers[layer_id].register_forward_hook(_make_hook(layer_id)))

        forward_params = inspect.signature(self.model.forward).parameters
        extra_kwargs = {
            name: False for name in ("output_hidden_states", "output_attentions", "use_cache") if name in forward_params
        }
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, **extra_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(self.target_layer_ids):
            raise RuntimeError(
                f"Expected {len(self.target_layer_ids)} captured layers but got {len(captured)}: {sorted(captured)}"
            )

        hidden_states = torch.cat([captured[layer_id] for layer_id in self.target_layer_ids], dim=-1)
        return DFlashTargetBatch(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
