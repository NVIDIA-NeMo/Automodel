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
# See the License for the specific governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.moe.rope_utils import apply_rotary_emb, yarn_get_mscale
from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_attn_module_and_func,
    initialize_linear_module,
)


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, backend: BackendConfig, use_sliding_attention: bool = False):
        super().__init__()

        self.sliding_window = config.sliding_window if use_sliding_attention else None
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_attention_heads * self.head_dim, self.hidden_size, bias=True
        )
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads))

        self.softmax_scale = self.head_dim**-0.5

        rope_scaling = config.rope_scaling

        if rope_scaling:
            factor = rope_scaling["factor"]
            mscale = rope_scaling.get("mscale", 1.0)
            original_seq_len = rope_scaling["original_max_position_embeddings"]
            if config.max_position_embeddings > original_seq_len:
                mscale = yarn_get_mscale(factor, mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

        assert backend.attn == "flex", "Only Flex Attention is supported for GPT-OSS"
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=config.num_attention_heads,
            num_qk_channels=config.head_dim,
            num_v_channels=config.head_dim,
            softmax_scale=self.softmax_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.q_proj(x).view(hidden_shape)
        k = self.k_proj(x).view(hidden_shape)
        v = self.v_proj(x).view(hidden_shape)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        output = self.attn_module(
            q,
            k,
            v,
            scale=self.softmax_scale,
            sink_weights=(self.sinks.to_local() if isinstance(self.sinks, DTensor) else self.sinks),
            sliding_window=(self.sliding_window if self.sliding_window is not None else 0),
            enable_gqa=True,
        )
        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(bsz, seqlen, -1).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.o_proj(output)  # (bsz, seqlen, dim)
        return output

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        with buffer_device:
            linear_list = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj,
            ]

            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
