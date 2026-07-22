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

"""Target-model wrapper for ViSpec draft training on a vision-language target.

Where the EAGLE-1/2 wrapper hands the draft token ids, ViSpec hands it the
target's **embedding-layer output**: at image positions there is no token
embedding to look up, only the vision tower's projected features, and those
features are exactly what the draft's image adaptor compresses. The wrapper
therefore returns the target's layer-0 hidden states alongside the usual
last-hidden-state / logits supervision, plus the image-token mask that tells the
draft which positions to compress.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left by one and zero-fill the tail.

    Args:
        tensor: Tensor of shape [batch, sequence, ...]; any trailing dimensions.

    Returns:
        Tensor of the same shape, with index ``i`` holding the input's index
        ``i + 1`` and the last position zeroed.
    """
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a possibly-sharded tensor as a plain local tensor.

    Args:
        tensor: Tensor (or DTensor) of any shape.

    Returns:
        Tensor of the same global shape; a no-op for an already-plain tensor.
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


@dataclass
class VispecTargetBatch:
    """Target-model outputs needed by :class:`VispecTrainerModule`.

    Attributes:
        inputs_embeds: Tensor of shape [batch, sequence, hidden] -- the target's
            embedding-layer output with vision features already spliced in,
            shifted left by one position.
        input_hidden_states: Tensor of shape [batch, sequence, hidden] -- the
            target's last hidden state, not shifted (the draft's input feature).
        target_logits: Tensor of shape [batch, sequence, vocab], shifted left.
        attention_mask: Tensor of shape [batch, sequence]; 1 for real tokens.
        loss_mask: Tensor of shape [batch, sequence], shifted left.
        image_mask: Bool tensor of shape [batch, sequence], shifted left so it
            aligns with ``inputs_embeds``.
    """

    inputs_embeds: torch.Tensor
    input_hidden_states: torch.Tensor
    target_logits: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    image_mask: torch.Tensor


class HFVispecTargetModel:
    """Expose embedding-layer, last-hidden-state, and logit supervision from a VLM target.

    Args:
        model: The frozen vision-language target model.
        image_token_id: Token id the target uses as an image placeholder; every
            position holding it carries a vision feature instead of a token
            embedding.
    """

    def __init__(self, model: nn.Module, *, image_token_id: int):
        self.model = model.eval()
        self.image_token_id = int(image_token_id)

    def get_input_embeddings(self) -> nn.Module:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    def get_lm_head(self) -> nn.Module:
        """Return the target model lm_head."""
        return self.model.lm_head

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **multimodal_inputs: torch.Tensor,
    ) -> VispecTargetBatch:
        """Run the frozen target once and assemble the draft's supervision.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence]; 1 for real tokens.
            loss_mask: Tensor of shape [batch, sequence]; 1 at supervised positions.
            **multimodal_inputs: The processor's vision tensors for this batch
                (e.g. ``pixel_values`` of shape [patches, patch_dim] and
                ``image_grid_thw`` of shape [images, 3] for Qwen2.5-VL). Keys the
                target's forward does not declare are dropped.

        Returns:
            VispecTargetBatch, with every tensor on the target's device.
        """
        base_model = self.model.model
        forward_params = inspect.signature(base_model.forward).parameters
        accepted = {name: value for name, value in multimodal_inputs.items() if name in forward_params}
        # A VLM base model declares its vision tensors explicitly but funnels the
        # HF-generic flags through a ``**kwargs`` catch-all, so a plain
        # ``name in forward_params`` test never matches them and the flags are
        # silently dropped -- leaving ``use_cache`` on its config default, which
        # allocates a full-sequence KV cache on every capture forward for nothing.
        has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in forward_params.values())
        extra_kwargs = {
            name: False for name in ("output_attentions", "use_cache") if name in forward_params or has_var_keyword
        }

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **accepted,
            **extra_kwargs,
        )
        hidden_states = outputs.hidden_states
        # HF emits ``num_layers + 1`` states: index 0 is the embedding output
        # (post vision-merge for a VLM), index -1 the post-final-norm state.
        inputs_embeds = hidden_states[0]
        last_hidden_states = hidden_states[-1]
        logits = _to_full_tensor(self.model.lm_head(last_hidden_states))

        return VispecTargetBatch(
            inputs_embeds=_shift_left_with_zero(inputs_embeds),
            input_hidden_states=last_hidden_states,
            target_logits=_shift_left_with_zero(logits),
            attention_mask=attention_mask,
            loss_mask=_shift_left_with_zero(loss_mask),
            image_mask=_shift_left_with_zero((input_ids == self.image_token_id).to(input_ids.dtype)).bool(),
        )
