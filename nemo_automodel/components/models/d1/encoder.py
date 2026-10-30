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

"""Omni's bidirectional LFM2 composition with media-prefix isolation."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Lfm2Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention, Lfm2MLP, Lfm2RMSNorm, Lfm2RotaryEmbedding


class D1OmniShortConv(nn.Module):
    """Centred gated convolution; media positions cannot read subsequent text."""

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        if config.conv_L_cache != 3 or config.conv_bias:
            raise ValueError("the released Omni encoder requires three-tap bias-free convolutions")
        width = config.hidden_size
        self.conv = nn.Conv1d(width, width, 3, groups=width, bias=False)
        self.in_proj = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, valid_tokens: torch.Tensor, media_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Mix neighbouring positions without crossing the media/text boundary.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            valid_tokens: Boolean Tensor of shape [batch, sequence] marking real positions.
            media_lengths: Long Tensor of shape [batch] giving each row's media prefix length.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        gates = self.in_proj(hidden_states * valid_tokens.unsqueeze(-1)).transpose(1, 2)
        input_gate, output_gate, values = gates.chunk(3, dim=1)
        neighbourhood = F.pad(input_gate * values, (1, 1)).unfold(-1, 3, 1)
        allow_right = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        allow_right = (allow_right != media_lengths.unsqueeze(1) - 1).to(hidden_states.dtype)
        weights = self.conv.weight[:, 0]
        left = neighbourhood[..., 0] * weights[:, 0].unsqueeze(-1)
        centre = neighbourhood[..., 1] * weights[:, 1].unsqueeze(-1)
        right = neighbourhood[..., 2] * allow_right.unsqueeze(1) * weights[:, 2].unsqueeze(-1)
        mixed = (left + centre) + right
        return self.out_proj((output_gate * mixed).transpose(1, 2))


class D1OmniEncoderLayer(nn.Module):
    """Compose standard LFM2 attention/MLP with Omni's centred convolution."""

    def __init__(self, config: Lfm2Config, layer_index: int) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_index] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_index)
            self.self_attn.is_causal = False
        else:
            self.conv = D1OmniShortConv(config)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        valid_tokens: torch.Tensor,
        media_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run a residual mixer followed by the gated feed-forward block.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_embeddings: Pair of cosine and sine Tensors, each of shape [batch, sequence, head_dim].
            attention_mask: Additive Tensor of shape [batch, 1, sequence, sequence].
            valid_tokens: Boolean Tensor of shape [batch, sequence].
            media_lengths: Long Tensor of shape [batch] of prefix lengths.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        normalized = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            mixed, _ = self.self_attn(normalized, position_embeddings, attention_mask)
        else:
            mixed = self.conv(normalized, valid_tokens, media_lengths)
        residual = hidden_states + mixed
        return residual + self.feed_forward(self.ffn_norm(residual))


class D1OmniEncoder(nn.Module):
    """Checkpoint-compatible bidirectional encoder for text and media embeddings.

    Perception modules supply prefixes. Text positions see the prefix and text;
    media positions see only media. There is no autoregressive cache.
    """

    def __init__(self, text_config: Lfm2Config) -> None:
        super().__init__()
        self.config = deepcopy(text_config)
        config = self.config
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per encoder layer")
        if any(kind not in {"conv", "full_attention"} for kind in config.layer_types):
            raise ValueError("Omni layers must be conv or full_attention")
        config._attn_implementation = "sdpa"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(D1OmniEncoderLayer(config, index) for index in range(config.num_hidden_layers))
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        with torch.device("cpu"):
            self.rotary_emb = Lfm2RotaryEmbedding(config)

    def _apply(self, fn, recurse: bool = True) -> D1OmniEncoder:
        # The released encoder rebuilds its frequencies in float32 on every call.
        # Restore exact frequencies after dtype/device moves rather than widening
        # already-rounded bfloat16 buffers from nn.Module.to().
        super()._apply(fn, recurse=recurse)
        initializer = self.rotary_emb.compute_default_rope_parameters
        if self.rotary_emb.rope_type != "default":
            initializer = ROPE_INIT_FUNCTIONS[self.rotary_emb.rope_type]
        with torch.device("cpu"):
            frequencies, scaling = initializer(self.config, device=torch.device("cpu"))
        frequencies = frequencies.to(self.rotary_emb.inv_freq.device)
        self.rotary_emb.inv_freq = frequencies
        self.rotary_emb.original_inv_freq = frequencies.clone()
        self.rotary_emb.attention_scaling = scaling
        return self

    def forward(
        self, inputs_embeds: torch.Tensor, valid_tokens: torch.Tensor, media_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode combined media and text embeddings.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            valid_tokens: Boolean Tensor of shape [batch, sequence], including media positions.
            media_lengths: Long Tensor of shape [batch], zero for text-only rows.

        Returns:
            Normalized Tensor of shape [batch, sequence, hidden].
        """
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("inputs_embeds must have shape [batch, length, hidden_size]")
        if valid_tokens.shape != inputs_embeds.shape[:2] or valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must be boolean with shape [batch, length]")
        if media_lengths.shape != inputs_embeds.shape[:1] or media_lengths.dtype != torch.long:
            raise ValueError("media_lengths must be long with shape [batch]")
        batch, length, _ = inputs_embeds.shape
        positions = torch.arange(length, device=inputs_embeds.device).expand(batch, -1)
        media_queries = positions < media_lengths.unsqueeze(1)
        text_keys = positions >= media_lengths.unsqueeze(1)
        forbidden = ~valid_tokens[:, None, None, :]
        forbidden = forbidden | (media_queries[:, None, :, None] & text_keys[:, None, None, :])
        mask = torch.zeros((batch, 1, length, length), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        negative = max(-1e9, torch.finfo(inputs_embeds.dtype).min)
        mask = mask.masked_fill(forbidden, negative)
        rotary = self.rotary_emb(inputs_embeds, positions)
        values = inputs_embeds
        for layer in self.layers:
            values = layer(values, rotary, mask, valid_tokens, media_lengths)
        return self.embedding_norm(values)
