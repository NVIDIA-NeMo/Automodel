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
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.moe.rope_utils import apply_rotary_emb, yarn_get_mscale
from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_attn_module_and_func,
    initialize_linear_module,
    initialize_rms_norm_module,
)


def _create_kwargs_for_te(attention_mask: torch.Tensor | None = None):
    if attention_mask is None:
        return {}
    else:
        return {
            "attn_mask_type": "padding",
            "window_size": (-1, -1),
            "attention_mask": attention_mask.unsqueeze(1).unsqueeze(2),
        }


class MLA(nn.Module):
    def __init__(self, config: DeepseekV3Config, backend: BackendConfig):
        super().__init__()

        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        self.backend = backend
        attn_impl = backend.attn
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm

        hidden_size = config.hidden_size

        if self.q_lora_rank is None:
            self.q_proj = initialize_linear_module(
                linear_impl=linear_impl,
                in_features=hidden_size,
                out_features=self.n_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = initialize_linear_module(
                linear_impl=linear_impl, in_features=hidden_size, out_features=self.q_lora_rank, bias=False
            )
            self.q_a_layernorm = initialize_rms_norm_module(rms_norm_impl=rms_norm_impl, dim=self.q_lora_rank)
            self.q_b_proj = initialize_linear_module(
                linear_impl=linear_impl,
                in_features=self.q_lora_rank,
                out_features=self.n_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = initialize_rms_norm_module(rms_norm_impl=rms_norm_impl, dim=self.kv_lora_rank)
        self.kv_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.kv_lora_rank,
            out_features=self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.n_heads * self.v_head_dim,
            out_features=hidden_size,
            bias=False,
        )
        self.softmax_scale = self.qk_head_dim**-0.5

        rope_scaling = config.rope_scaling

        if rope_scaling:
            factor = rope_scaling["factor"]
            mscale = rope_scaling["mscale"]
            original_seq_len = rope_scaling["original_max_position_embeddings"]
            if config.max_position_embeddings > original_seq_len:
                mscale = yarn_get_mscale(factor, mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=attn_impl,
            num_attention_heads=self.n_heads,
            num_qk_channels=self.qk_head_dim,
            num_v_channels=self.v_head_dim,
            softmax_scale=self.softmax_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        bsz, local_seq_len, _ = x.size()

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.view(bsz, local_seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, -1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2).expand([bsz, -1, self.n_heads, self.qk_rope_head_dim])
        k = torch.cat([k_nope, k_pe], dim=-1)

        attn_kwargs = _create_kwargs_for_te(attention_mask) if self.backend.attn == "te" else {}
        x = self.attn_func(q, k, v, **attn_kwargs)

        x = self.o_proj(x.flatten(2))
        return x

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [
            self.kv_a_proj_with_mqa,
            self.kv_b_proj,
            self.o_proj,
        ]
        if self.q_lora_rank is None:
            linear_list.append(self.q_proj)
        else:
            linear_list.extend([self.q_a_proj, self.q_b_proj])

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        norms = [self.kv_a_layernorm]
        if self.q_lora_rank is not None:
            norms.append(self.q_a_layernorm)
        for norm in norms:
            norm.reset_parameters()
