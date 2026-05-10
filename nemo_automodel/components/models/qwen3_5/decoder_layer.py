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

"""Custom Qwen3.5 decoder layer that threads packed-sequence metadata to ``linear_attn``.

HF's ``Qwen3_5DecoderLayer.forward`` calls ``self.linear_attn`` with only
``hidden_states``, ``cache_params``, ``cache_position`` and ``attention_mask``.
For NEAT-packed inputs the linear-attn kernel additionally needs:

* ``cu_seqlens`` -- per-document cumulative lengths (FLA's segment-reset signal
  for ``chunk_gated_delta_rule``).
* ``indices`` -- non-padding token indices in the flattened sequence (used to
  unpad ``[B, T, ...]`` to ``[1, total_valid, ...]`` before the kernel and
  re-pad after; required for B>1 packed batches).

* ``position_ids`` -- needed by the CP path to undo PyTorch's load-balanced
  shuffle.

This subclass derives the packing kwargs from the indexed ``attention_mask`` and
forwards them, plus ``position_ids``, into ``linear_attn``. ``patch_hf_model``
swaps every ``Qwen3_5DecoderLayer`` instance to this class at model build time,
so this is the *only* file that needs to know about the kwarg drop in HF's
decoder layer.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from nemo_automodel.components.models.common.packing import get_unpad_data, is_indexed_packed_mask


class Qwen3_5DecoderLayerWithPacking(Qwen3_5DecoderLayer):
    """Drop-in subclass of HF ``Qwen3_5DecoderLayer`` with packing-aware dispatch.

    All weights and ``__init__`` are inherited unchanged. Only ``forward`` is
    overridden so the ``linear_attn`` call site receives ``cu_seqlens``,
    ``indices`` and ``position_ids`` in addition to ``attention_mask``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            cu_seqlens: torch.Tensor | None = None
            indices: torch.Tensor | None = None
            linear_attn_mask = attention_mask
            packed_seq_ids = kwargs.get("_packed_seq_ids")
            if is_indexed_packed_mask(attention_mask):
                packing_mask = attention_mask
            elif is_indexed_packed_mask(packed_seq_ids):
                packing_mask = packed_seq_ids
            else:
                packing_mask = None

            if packing_mask is not None:
                indices_t, cu_seqlens_t, _ = get_unpad_data(packing_mask)
                indices = indices_t
                cu_seqlens = cu_seqlens_t.to(torch.long)
                linear_attn_mask = packing_mask

            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=linear_attn_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                indices=indices,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
