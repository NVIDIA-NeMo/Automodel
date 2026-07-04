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

from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs


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

    def __init__(self, model: nn.Module, target_layer_ids: Sequence[int], cp_mesh=None):
        self.model = model.eval()
        self._num_layers = len(self._get_transformer_layers())
        # ``-1`` (the embedding output) is accepted, matching
        # ``common.extract_context_feature`` and the draft ``fc`` sizing in the
        # config builders.
        self.target_layer_ids = validate_target_layer_ids(list(target_layer_ids), self._num_layers)
        # Context parallelism shards this frozen forward along the sequence dim;
        # generate_batch gathers the captured hidden states back to the full
        # sequence so the draft's anchor/block masks stay CP-unaware. cp_mesh is the
        # "cp" submesh (or None).
        self.cp_mesh = cp_mesh
        self._cp_size = cp_mesh.size() if cp_mesh is not None else 1
        if self._cp_size > 1:
            from nemo_automodel.components.distributed.cp_utils import (
                attach_context_parallel_hooks,
                attach_cp_kv_gather_hooks,
            )

            # Strip the 4D mask (self_attn then calls SDPA on the local shard), and
            # all-gather K/V so each rank attends its local Q against the full sequence
            # -- torch's ring dispatch does not fire for a plain HF forward.
            attach_context_parallel_hooks(self.model)
            attach_cp_kv_gather_hooks(self.model, cp_mesh)

    def _inner_model(self) -> nn.Module:
        """Return the base transformer module that owns ``layers`` and ``norm``.

        Handles the common nestings: a plain causal LM (``model.model``), a
        decoder-only base (``model``), and multimodal targets whose text stack is
        under ``language_model`` (e.g. Gemma4: ``model.model.language_model``).
        """
        seen = set()
        queue = [self.model, getattr(self.model, "model", None)]
        while queue:
            module = queue.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, "layers"):
                return module
            queue.append(getattr(module, "language_model", None))
            queue.append(getattr(module, "model", None))
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

    def _collapse_hc_streams(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collapse a 4D Hyper-Connection stream ``[B, S, hc_mult, H]`` to ``[B, S, H]``.

        DeepSeek V4 decoder layers emit ``hc_mult`` parallel residual copies; only the
        final-norm output is already collapsed. For an intermediate target-feature
        layer we reduce the streams with their mean: a simple, in-distribution
        reduction that the draft's learnable ``fc`` then reprojects. We deliberately
        avoid the model's final ``hc_head`` here, since it is trained for the
        last-layer stream distribution, not the intermediate ones. Non-HC targets
        emit 3D states and pass through unchanged.
        """
        if tensor.ndim != 4:
            return tensor
        return tensor.mean(dim=2)

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
        **mm_kwargs: torch.Tensor,
    ) -> DSparkTargetBatch:
        """Run the target model once and capture the DSpark context + last hidden state.

        Features follow ``common.extract_context_feature`` exactly: ``-1`` is the
        embedding output, the final layer is the post-norm hidden state (HF
        ``output_hidden_states[num_layers]``), and any other id is that decoder
        layer's output. The final-norm output is also returned as the last hidden
        state for the TV / confidence losses.

        ``mm_kwargs`` carries any multimodal inputs present in the training batch
        (``pixel_values``, ``image_grid_thw``, ...); :func:`filter_forward_kwargs`
        drops whatever the target's own forward signature does not accept, so a
        text-only target (Qwen3, Gemma4, or MiniMax M3 without multimodal training
        data) is never passed inputs it cannot handle -- this is a no-behavior-change
        extension for every existing (``mm_kwargs``-empty) caller. A VLM target
        (e.g. MiniMax M3) splices the vision features into its embedding sequence
        internally when they are provided.
        """
        layers = self._get_transformer_layers()
        last = self._num_layers - 1
        captured: dict[object, torch.Tensor] = {}
        handles = []

        def _make_hook(key):
            def _hook(_module, _inputs, outputs):
                captured[key] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        if -1 in self.target_layer_ids:
            handles.append(self.model.get_input_embeddings().register_forward_hook(_make_hook("embed")))
        for layer_id in self.target_layer_ids:
            if 0 <= layer_id < last:
                handles.append(layers[layer_id].register_forward_hook(_make_hook(layer_id)))
        # The final norm gives both the post-norm last hidden state and the
        # last-layer feature (post-norm), matching the HF offset-1 convention.
        handles.append(self._get_final_norm().register_forward_hook(_make_hook("norm")))

        if self._cp_size > 1:
            # Shard the sequence, run the target as ring attention, then gather every
            # captured hidden state back to the full sequence.
            from nemo_automodel.components.distributed.cp_utils import run_target_cp_forward_and_gather

            keys: list = []

            def _collect(_outputs):
                if "norm" not in captured:
                    raise RuntimeError("DSpark target capture did not record the final-norm output.")
                keys.extend(captured.keys())
                return [captured[k] for k in keys]

            try:
                _outputs, gathered = run_target_cp_forward_and_gather(
                    self.cp_mesh,
                    self.model,
                    input_ids,
                    dict(use_cache=False, **mm_kwargs),
                    _collect,
                    filter_kwargs=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            captured = dict(zip(keys, gathered))
        else:
            forward_kwargs = filter_forward_kwargs(
                self.model,
                dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **mm_kwargs),
            )
            try:
                self.model(**forward_kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            if "norm" not in captured:
                raise RuntimeError("DSpark target capture did not record the final-norm output.")

        def _feature(layer_id: int) -> torch.Tensor:
            if layer_id == -1:
                feat = captured["embed"]
            elif layer_id == last:
                feat = captured["norm"]
            else:
                feat = captured[layer_id]
            return self._collapse_hc_streams(feat)

        target_hidden_states = torch.cat([_feature(layer_id) for layer_id in self.target_layer_ids], dim=-1)
        return DSparkTargetBatch(
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=captured["norm"],
            input_ids=input_ids,
            loss_mask=loss_mask,
        )


__all__ = ["HFDSparkTargetModel", "DSparkTargetBatch"]
