# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Dense Kimi K3 MLA backbone for DSpark speculative-decoding training."""

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.kimi_k3.model import KimiK3MLP, KimiRMSNorm
from nemo_automodel.components.speculative.dspark.common import AcceptRatePredictor
from nemo_automodel.components.speculative.dspark.draft_deepseek_v4 import DeepseekV4DSparkModel
from nemo_automodel.components.speculative.dspark.markov_head import build_markov_head


class KimiK3DSparkAttention(nn.Module):
    """Dense non-causal K3 MLA over target context and a parallel noise block.

    K3 MLA keeps one K/V per query head (``num_key_value_heads == num_attention_heads``),
    like GLM-5.2's MLA, so there is no GQA group repeat: ``kv_b_proj`` already emits one
    ``qk_nope_head_dim + v_head_dim`` slice per query head.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        assert self.num_key_value_heads == self.num_heads, (
            "Kimi K3 DSpark MLA expects one K/V head per query head, got "
            f"num_key_value_heads={self.num_key_value_heads} vs num_attention_heads={self.num_heads}."
        )
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)

        if not config.mla_use_nope:
            raise ValueError("Kimi K3 DSpark requires mla_use_nope=True.")
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=torch.float32)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=torch.float32)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.use_output_gate = bool(config.mla_use_output_gate)
        if self.use_output_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attend from the noise block to the frozen target context plus itself.

        Args:
            hidden_states: Tensor of shape ``[batch, query_length, hidden]``; the
                noise block that supplies the queries.
            target_hidden_states: Tensor of shape ``[batch, context_length, hidden]``;
                the frozen target context, concatenated before ``hidden_states`` to
                form the keys and values.
            attention_mask: Optional additive mask of shape
                ``[batch, 1, query_length, context_length + query_length]``.

        Returns:
            Tensor of shape ``[batch, query_length, hidden]``.
        """
        del kwargs
        batch_size, query_length = hidden_states.shape[:-1]
        key_value_states = torch.cat([target_hidden_states, hidden_states], dim=1)
        key_value_length = key_value_states.shape[1]

        if self.q_lora_rank is not None:
            query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, query_length, self.num_heads, self.q_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(key_value_states)
        key_states, key_rotary_states = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        key_states = self.kv_b_proj(self.kv_a_layernorm(key_states))
        key_states = key_states.view(
            batch_size,
            key_value_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        key_states, value_states = torch.split(
            key_states,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        key_rotary_states = key_rotary_states.view(batch_size, 1, key_value_length, self.qk_rope_head_dim)
        key_rotary_states = key_rotary_states.expand(*key_states.shape[:-1], -1)
        key_states = torch.cat([key_states, key_rotary_states], dim=-1)

        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask[..., :key_value_length]
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, -1)
        if self.use_output_gate:
            attention_output = attention_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attention_output)


class KimiK3DSparkDecoderLayer(nn.Module):
    """Pre-norm K3 MLA and dense SiTU feed-forward block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = KimiK3DSparkAttention(config, layer_idx)
        self.mlp = KimiK3MLP(config, dtype=torch.float32)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=torch.float32,
        )

    def forward(
        self,
        target_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run one draft layer over the noise block against the target context.

        Args:
            target_hidden_states: Tensor of shape ``[batch, context_length, hidden]``;
                the frozen target context supplying the extra keys and values.
            hidden_states: Tensor of shape ``[batch, query_length, hidden]``; the
                noise block, also the residual stream.
            attention_mask: Optional additive mask of shape
                ``[batch, 1, query_length, context_length + query_length]``.

        Returns:
            Tensor of shape ``[batch, query_length, hidden]``.
        """
        del kwargs
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden_states,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class KimiK3DSparkModel(DeepseekV4DSparkModel):
    """DSpark draft with a dense K3 MLA backbone and no KDA or MoE layers."""

    _no_split_modules = ["KimiK3DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.target_layer_ids = config.target_layer_ids
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [KimiK3DSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.num_anchors = int(config.num_anchors)
        self.markov_head = build_markov_head(config)
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = bool(self.enable_confidence_head and config.confidence_head_with_markov)
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size + (config.markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Project the target context once, then run the draft layers over the noise block.

        Args:
            position_ids: Tensor of shape ``[batch, sequence]``; unused (NoPE MLA).
            attention_mask: Optional additive mask of shape
                ``[batch, 1, query_length, context_length + query_length]``.
            noise_embedding: Tensor of shape ``[batch, query_length, hidden]``; the
                noise block that becomes the queries.
            target_hidden_states: Tensor of shape
                ``[batch, context_length, len(target_layer_ids) * hidden]``; the
                concatenated target hidden states, projected by ``fc`` and normed
                to ``[batch, context_length, hidden]``.

        Returns:
            Tensor of shape ``[batch, query_length, hidden]``.
        """
        del position_ids, kwargs
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)


__all__ = ["KimiK3DSparkAttention", "KimiK3DSparkDecoderLayer", "KimiK3DSparkModel"]
