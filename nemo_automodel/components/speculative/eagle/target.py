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

import inspect
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend


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
    """Target-model supervision for one draft-training batch.

    Carries exactly one supervision encoding (validated in ``__post_init__``),
    both consumed directly by ``Eagle3TrainerModule.forward``:

    - ``logits`` -- the target's full-vocab logits; the draft-vocab projection
      happens trainer-side. Used by the co-located backend, where the tensor
      never leaves the GPU.
    - ``target_probs`` + ``position_mask`` -- the already-projected draft-vocab
      distribution, so a backend that computes it itself (e.g. a remote server)
      only transfers draft-vocab-sized tensors.
    """

    aux_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    logits: torch.Tensor | None = None
    target_probs: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        has_logits = self.logits is not None
        has_precomputed = self.target_probs is not None and self.position_mask is not None
        if has_logits == has_precomputed:
            raise ValueError(
                "Eagle3TargetBatch requires exactly one supervision source: either "
                "`logits` (full-vocab, projected trainer-side) or both `target_probs` "
                "and `position_mask` (precomputed over the draft vocab)."
            )

    def to_trainer_inputs(self) -> dict[str, torch.Tensor]:
        """Return kwargs for ``Eagle3TrainerModule.forward``, dispatching on
        whichever supervision encoding this batch carries."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "aux_hidden_states": self.aux_hidden_states,
        }
        if self.logits is not None:
            inputs["target_logits"] = self.logits
        else:
            inputs["target_probs"] = self.target_probs
            inputs["position_mask"] = self.position_mask
        return inputs


class HFEagle3TargetModel(Eagle3TargetBackend):
    """Co-located backend that captures three auxiliary hidden states from a causal LM."""

    def __init__(self, model: nn.Module, aux_layer_ids: Sequence[int] | None = None):
        self.model = model.eval()
        candidate_ids = list(aux_layer_ids) if aux_layer_ids is not None else self._default_aux_layer_ids()
        self.aux_layer_ids = self._validate_aux_layer_ids(candidate_ids)

    def _default_aux_layer_ids(self) -> list[int]:
        # EAGLE-3 default 3-layer recipe (low / mid / high).
        #
        # The downstream draft model's ``fc`` projection is sized for
        # exactly ``num_aux_hidden_states`` layers (default 3) of
        # concatenated target hidden states. Silently deduplicating
        # collisions on shallow targets would yield fewer than 3
        # captured tensors and crash later inside the draft ``fc`` with
        # a confusing shape-mismatch error -- raise here instead so the
        # caller picks 3 distinct in-bounds ids that match the draft
        # config.
        num_layers = self.model.config.num_hidden_layers
        candidates = [1, num_layers // 2 - 1, num_layers - 4]
        if any(c < 0 or c >= num_layers for c in candidates) or len(set(candidates)) != 3:
            raise ValueError(
                f"Target model has num_hidden_layers={num_layers}, which is too shallow "
                f"for the default EAGLE-3 aux recipe {candidates}. Pass aux_layer_ids "
                f"explicitly (must be 3 distinct in-bounds layer indices, matching the "
                f"draft model's num_aux_hidden_states)."
            )
        return candidates

    def _validate_aux_layer_ids(self, aux_layer_ids: Sequence[int]) -> list[int]:
        """Validate aux-layer selection before any forward hooks are registered."""
        num_layers = self.model.config.num_hidden_layers
        aux_layer_ids = list(aux_layer_ids)
        if len(aux_layer_ids) != 3:
            raise ValueError(
                f"EAGLE-3 expects exactly 3 aux_layer_ids, but got {len(aux_layer_ids)}: "
                f"{aux_layer_ids}. This must match the draft model's num_aux_hidden_states."
            )
        if len(set(aux_layer_ids)) != len(aux_layer_ids):
            raise ValueError(
                f"EAGLE-3 aux_layer_ids must be distinct, but got {aux_layer_ids}. "
                "Duplicate ids would collapse the captured aux hidden states."
            )
        for layer_id in aux_layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {num_layers} layers")
        return aux_layer_ids

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Return decoder layers as an ordered list indexable by integer.

        Supports both the HuggingFace layouts (where ``layers`` is a
        ``ModuleList``) and AutoModel's custom-impl layouts (where
        ``layers`` is a ``ModuleDict`` keyed by ``str(i)``). Returning a
        plain list normalizes the access pattern for downstream
        ``register_forward_hook`` calls.
        """
        # Common HF causal-LM layouts:
        #   model.model.layers              (Llama, Qwen, Mistral, Gemma, Phi, ...)
        #   model.layers                    (some VLM text backbones exposed directly)
        #   model.transformer.h             (GPT2 / Falcon-style)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            container = self.model.model.layers
        elif hasattr(self.model, "layers"):
            container = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            container = self.model.transformer.h
        else:
            raise ValueError("Unsupported model structure for EAGLE-3 aux-layer capture")
        if isinstance(container, nn.ModuleDict):
            # AutoModel custom impls use ModuleDict keyed by ``str(i)``.
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

        # AutoModel's custom causal LMs only declare ``input_ids``,
        # ``attention_mask``, ``position_ids``, ``padding_mask`` and a
        # ``**attn_kwargs`` catch-all; the HF flags below mean nothing to
        # them and are dropped to keep the call site honest.
        forward_params = inspect.signature(self.model.forward).parameters
        extra_kwargs = {
            name: False for name in ("output_hidden_states", "output_attentions", "use_cache") if name in forward_params
        }

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **extra_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(self.aux_layer_ids):
            raise RuntimeError(
                f"Expected {len(self.aux_layer_ids)} captured aux layers but got {len(captured)}: {sorted(captured)}"
            )

        aux_hidden_states = torch.cat([captured[layer_id] for layer_id in self.aux_layer_ids], dim=-1)
        # HF causal LM outputs wrap logits in a dataclass; AutoModel's
        # custom causal LM returns the logits tensor directly.
        target_logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shifted_logits = _shift_left_with_zero(target_logits)
        shifted_input_ids = _shift_left_with_zero(input_ids)
        shifted_loss_mask = _shift_left_with_zero(loss_mask)
        return Eagle3TargetBatch(
            aux_hidden_states=aux_hidden_states,
            logits=shifted_logits,
            input_ids=shifted_input_ids,
            attention_mask=attention_mask,
            loss_mask=shifted_loss_mask,
        )
