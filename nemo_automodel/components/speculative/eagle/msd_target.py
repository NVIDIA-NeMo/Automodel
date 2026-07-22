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

"""Frozen VLM target wrapper for multimodal speculative-decoding training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.target_v12 import _shift_left_with_zero


@dataclass
class MSDTargetBatch:
    """Frozen VLM supervision tensors consumed by :class:`MSDTrainerModule`."""

    inputs_embeds: torch.Tensor
    input_hidden_states: torch.Tensor
    target_hidden_states: torch.Tensor
    target_logits: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    image_mask: torch.Tensor


class _InputEmbeddingCapture:
    """Capture the input embeddings supplied to a VLM language backbone."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def __call__(self, _module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Save the fused VLM embedding tensor passed to the language backbone.

        Args:
            _module: Language backbone receiving the VLM's fused input embeddings.
            _args: Positional arguments accepted by the language backbone.
            kwargs: Keyword arguments accepted by the language backbone. Its
                ``inputs_embeds`` value is a Tensor of shape [batch, sequence,
                hidden] containing text and projected image embeddings.
        """
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor):
            raise RuntimeError(
                "The VLM language backbone did not receive an `inputs_embeds` tensor. "
                "MSD requires a VLM that fuses projected image embeddings before its language backbone."
            )
        self.value = inputs_embeds


class HFMSDTargetModel:
    """Expose image-aware feature supervision from a frozen Hugging Face VLM."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError(
                "MSD requires a VLM config with `image_token_id` so projected image-token positions can be decoupled."
            )
        if not hasattr(model, "model"):
            raise ValueError("MSD requires a Hugging Face VLM exposing its language backbone as `model`.")
        if not hasattr(model, "lm_head"):
            raise ValueError("MSD requires a Hugging Face VLM exposing its language head as `lm_head`.")
        self.image_token_id = int(image_token_id)

    def get_lm_head(self) -> nn.Module:
        """Return the frozen VLM language-model head used by the draft loss."""
        return self.model.lm_head

    @torch.no_grad()
    def generate_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
    ) -> MSDTargetBatch:
        """Run a VLM and construct shifted multimodal draft supervision.

        Args:
            input_ids: Long tensor of shape [batch, sequence], with image-token
                placeholders at positions that receive projected vision features.
            attention_mask: Tensor of shape [batch, sequence], where one denotes
                a real token and zero denotes padding.
            loss_mask: Bool tensor of shape [batch, sequence], already aligned to
                next-token logits. VLM collators produce this from ``labels != -100``.
            model_inputs: Mapping of VLM forward inputs. It includes ``input_ids``
                and ``attention_mask`` plus any vision tensors such as
                ``pixel_values`` and ``image_grid_thw``.

        Returns:
            MSDTargetBatch with shifted image-aware embeddings, target hidden
            states, target logits, and masks. Tensor layouts retain [batch,
            sequence, hidden] or [batch, sequence, vocab] as appropriate.
        """
        capture = _InputEmbeddingCapture()
        handle = self.model.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            outputs = self.model(output_hidden_states=True, return_dict=True, use_cache=False, **model_inputs)
        finally:
            handle.remove()

        if capture.value is None:
            raise RuntimeError("MSD could not capture fused VLM input embeddings from the language backbone.")
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        image_mask = input_ids.eq(self.image_token_id)
        return MSDTargetBatch(
            inputs_embeds=_shift_left_with_zero(capture.value),
            input_hidden_states=hidden_states,
            target_hidden_states=_shift_left_with_zero(hidden_states),
            target_logits=_shift_left_with_zero(logits),
            attention_mask=attention_mask,
            loss_mask=loss_mask.bool(),
            image_mask=_shift_left_with_zero(image_mask),
        )
