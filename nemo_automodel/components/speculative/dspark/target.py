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
from typing import Optional, Sequence

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.llm.packed_sequence import build_block_causal_additive_mask
from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs


@dataclass
class DSparkTargetBatch:
    """Target-model features needed by the DSpark trainer.

    ``position_ids`` / ``seq_lens`` / ``doc_remaining`` are ``None`` off the
    packing path and carry the (unshifted) packing metadata to the trainer on it.
    """

    target_hidden_states: torch.Tensor  # [B, S, len(target_layer_ids) * H]
    target_last_hidden_states: torch.Tensor  # [B, S, H]
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    doc_remaining: Optional[torch.Tensor] = None


class HFDSparkTargetModel:
    """Capture intermediate + final hidden states from a frozen HF causal LM.

    A forward hook on decoder layer ``i`` captures ``hidden_states[i + 1]`` (the
    HuggingFace ``output_hidden_states`` offset-1 convention); a hook on the final
    norm captures the post-norm last hidden state.
    """

    def __init__(self, model: nn.Module, target_layer_ids: Sequence[int]):
        self.model = model.eval()
        self._num_layers = len(self._get_transformer_layers())
        # ``-1`` (the embedding output) is accepted, matching
        # ``common.extract_context_feature`` and the draft ``fc`` sizing in the
        # config builders.
        self.target_layer_ids = validate_target_layer_ids(list(target_layer_ids), self._num_layers)

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
        position_ids: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        doc_remaining: Optional[torch.Tensor] = None,
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

        target_attention_mask = attention_mask
        extra_kwargs: dict[str, torch.Tensor] = {}
        if seq_lens is not None:
            # Packing: run the frozen target block-causal (SDPA/eager consume the
            # [B, 1, S, S] additive mask; FlashAttention infers boundaries from the
            # per-document position_ids at batch size 1) so the captured context
            # hidden states do not leak across document boundaries.
            if position_ids is None:
                raise ValueError("DSpark sequence packing requires per-document position_ids, but none were provided.")
            extra_kwargs["position_ids"] = position_ids
            attn_impl = getattr(self.model.config, "_attn_implementation", None) or ""
            if "flash" in attn_impl:
                if input_ids.shape[0] != 1:
                    raise ValueError(
                        "DSpark sequence packing with a FlashAttention target only supports "
                        f"micro_batch_size=1 (got {input_ids.shape[0]}); set micro_batch_size=1 or load "
                        "the target with attn_implementation='sdpa'."
                    )
                target_attention_mask = None
            else:
                param_dtype = next(self.model.parameters()).dtype
                target_attention_mask = build_block_causal_additive_mask(
                    seq_lens, seq_length=input_ids.shape[1], dtype=param_dtype, device=input_ids.device
                )

        forward_kwargs = filter_forward_kwargs(
            self.model,
            dict(
                input_ids=input_ids,
                attention_mask=target_attention_mask,
                use_cache=False,
                **extra_kwargs,
                **mm_kwargs,
            ),
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
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )


__all__ = ["HFDSparkTargetModel", "DSparkTargetBatch"]
