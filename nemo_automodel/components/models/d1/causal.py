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

"""Released D1-3B rotary precision on the standard LFM2 causal stack."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2RotaryEmbedding,
    Lfm2ShortConv,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class D1CausalShortConv(Lfm2ShortConv):
    """Retain HF checkpoint/cache ownership with the release's chain convolution."""

    def __init__(self, source: Lfm2ShortConv) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.conv_kernel_size = source.conv_kernel_size
        self.bias = source.bias
        self.layer_type = source.layer_type
        self.conv = source.conv
        self.in_proj = source.in_proj
        self.out_proj = source.out_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        """Run the released depthwise kernel for independent uncached chains.

        Cached decoding and packed sequence indices retain HF's implementation.
        Training uses uncached chains and preserves the publisher's convolution
        accumulation rather than the optional fused causal-conv kernel.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            past_key_values: Optional HF cache. Cached decoding delegates to HF
                and updates its per-layer convolution state in place.
            attention_mask: Optional Tensor of shape [batch, sequence], with
                one/true for real tokens and zero/false for padding.
            seq_idx: Optional integer Tensor of shape [batch, sequence] giving
                packed sequence IDs. When present the HF path owns segment
                boundary handling and its optional causal-convolution kernel.

        Returns:
            Tensor of shape [batch, sequence, hidden] in the input dtype.
        """
        if past_key_values is not None or seq_idx is not None:
            return super().forward(hidden_states, past_key_values, attention_mask, seq_idx)
        states = apply_mask_to_padding_states(hidden_states, attention_mask)
        input_gate, output_gate, values = self.in_proj(states).transpose(1, 2).chunk(3, dim=1)
        mixed = F.conv1d(
            F.pad(input_gate * values, (self.conv_kernel_size - 1, 0)),
            self.conv.weight,
            self.conv.bias,
            groups=states.shape[-1],
        )
        return self.out_proj((output_gate * mixed).transpose(1, 2))


class D1CausalRotaryEmbedding(Lfm2RotaryEmbedding):
    """Keep frequencies and trigonometry in float32, as the released D1 decoder."""

    def _apply(self, fn, recurse: bool = True) -> D1CausalRotaryEmbedding:
        super()._apply(fn, recurse=recurse)
        # The publisher builds frequencies on the execution device. CPU power
        # rounding can change a later bfloat16 rotation at the released width.
        initializer = self.compute_default_rope_parameters
        if self.rope_type != "default":
            initializer = ROPE_INIT_FUNCTIONS[self.rope_type]
        with torch.device(self.inv_freq.device):
            frequencies, scaling = initializer(self.config, device=self.inv_freq.device)
        self.inv_freq = frequencies.to(self.inv_freq.device)
        self.original_inv_freq = self.inv_freq.clone()
        self.attention_scaling = scaling
        return self

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the release's float32 rotary factors.

        Args:
            x: Tensor of shape [batch, sequence, hidden], supplying the device.
            position_ids: Long Tensor of shape [batch, sequence].

        Returns:
            Pair of float32 cosine and sine Tensors, each of shape
            [batch, sequence, head_dim].
        """
        return super().forward(x.float(), position_ids)


class D1CausalAttention(Lfm2Attention):
    """Reuse an HF attention block's weights with the released float32 rotation."""

    def __init__(self, source: Lfm2Attention) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.is_causal = True
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.out_proj = source.out_proj
        self.q_layernorm = source.q_layernorm
        self.k_layernorm = source.k_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run standard GQA after the checkpoint's float32 half-split rotation.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_embeddings: Pair of float32 cosine/sine Tensors, each of
                shape [batch, sequence, head_dim].
            attention_mask: Optional additive Tensor of shape
                [batch, 1, sequence, key_sequence], with zero for allowed keys
                and a negative sentinel for causal/padding exclusions.
            past_key_values: Optional HF autoregressive cache, updated in place.
            **kwargs: Additional HF attention dispatcher arguments.

        Returns:
            Pair of output Tensor of shape [batch, sequence, hidden] and
            optional attention-weight Tensor of shape
            [batch, heads, sequence, key_sequence]. SDPA returns no weights.
        """
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_layernorm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_layernorm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cosine, sine = position_embeddings
        rotated_query, rotated_key = apply_rotary_pos_emb(query.float(), key.float(), cosine.float(), sine.float())
        query, key = rotated_query.to(query.dtype), rotated_key.to(key.dtype)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        implementation = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attended, weights = implementation(
            self, query, key, value, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs
        )
        attended = attended.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.out_proj(attended), weights
