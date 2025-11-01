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

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.shared.import_utils import is_te_min_version

if TYPE_CHECKING:
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb
from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_linear_module,
)


class GptOssAttention(nn.Module):
    def __init__(self, config: "GptOssConfig", backend: BackendConfig, use_sliding_attention: bool = False):
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

        self.softmax_scale = self.head_dim**-0.5

        assert backend.attn in ("flex", "te"), "Only Flex and TE Attention are supported for GPT-OSS"
        if backend.attn == "te" and not is_te_min_version("2.8.0"):
            raise ValueError(
                "Transformer Engine DotProductAttention for GPT-OSS is only supported for TE version 2.8.0 or higher"
            )

        self.backend = backend
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=config.num_attention_heads,
            num_qk_channels=config.head_dim,
            num_v_channels=config.head_dim,
            softmax_scale=self.softmax_scale,
            num_gqa_groups=self.num_key_value_heads,
            softmax_type="learnable",
        )
        # TE initializes sinks inside the attn_module
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads)) if backend.attn == "flex" else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_attention_heads, self.head_dim)
            k = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
            v = v.view(num_tokens, self.num_key_value_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_attention_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        # freqs_cis is concatenated [cos, sin] along last dim with shape (B, T, head_dim)
        cos, sin = freqs_cis.split(self.head_dim // 2, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.backend.attn == "flex":
            updated_attn_kwargs = {
                "scale": self.softmax_scale,
                "sink_weights": (self.sinks.to_local() if isinstance(self.sinks, DTensor) else self.sinks),
                "sliding_window": (self.sliding_window if self.sliding_window is not None else 0),
                "enable_gqa": True,
            }
        else:
            updated_attn_kwargs = attn_kwargs
            if self.sliding_window is not None:
                updated_attn_kwargs["window_size"] = (self.sliding_window, 0)

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **updated_attn_kwargs
        )
        output = self.attn_func(q, k, v, **_attn_kwargs)
        output = postprocess_output_for_attn(output, self.backend.attn)

        # Reshape and project output
        flatten_dim = 2 if qkv_format == "bshd" else 1
        output = self.o_proj(output.flatten(flatten_dim))
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

            if self.backend.attn == "flex":
                nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)
            else:
                nn.init.trunc_normal_(self.attn_module.softmax_offset, mean=0.0, std=init_std)
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
