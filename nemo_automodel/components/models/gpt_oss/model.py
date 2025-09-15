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

from typing import Any

import torch
import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.gpt_oss.layers import GptOssAttention
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.moe.rope_utils import freqs_cis_from_position_ids, precompute_freqs_cis
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: GptOssConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = GptOssAttention(
            config, backend, use_sliding_attention=config.layer_types[layer_idx] == "sliding_attention"
        )
        self.mlp = MoE(moe_config, backend)
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        # TODO: Support arbitrary attention masks
        attn_out = self.self_attn(self.input_layernorm(x), freqs_cis=freqs_cis)
        x = x + attn_out

        mlp_in = self.post_attention_layernorm(x)
        mlp_out = self._mlp(mlp_in, padding_mask)
        x = x + mlp_out
        return x

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config
        # GPT-OSS is MoE everywhere; set shared experts to 0 to disable shared path in our MoE wrapper.
        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.intermediate_size,
            n_routed_experts=config.num_local_experts,
            n_shared_experts=0,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=getattr(config, "n_group", 1),
            n_limited_groups=getattr(config, "topk_group", 1),
            train_gate=True,
            gate_bias_update_factor=0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=config.router_aux_loss_coef,
            norm_topk_prob=getattr(config, "norm_topk_prob", False),
            expert_bias=True,
            router_bias=True,
            expert_activation="quick_geglu",
            activation_alpha=1.702,
            activation_limit=getattr(config, "swiglu_limit", 7.0),
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        # RoPE precompute using shared utils (like DeepSeek-V3)
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.head_dim,
                self.max_seq_len,
                getattr(config, "rope_theta", 10000.0),
                getattr(config, "rope_scaling", None),
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        with torch.no_grad():
            freqs_cis = freqs_cis_from_position_ids(position_ids, self.freqs_cis)

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        for layer in self.layers.values():
            h = layer(
                h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )

        h = self.norm(h) if self.norm else h
        return h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            # refresh rope cache to target device
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim,
                self.max_seq_len,
                self.config.rope_theta,
                self.config.rope_scaling,
            ).to(buffer_device)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class GptOssForCausalLM(nn.Module):
    @classmethod
    def from_config(
        cls,
        pretrained_model_name_or_path: str | GptOssConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        trust_remote_code: bool = False,
    ):
        if isinstance(pretrained_model_name_or_path, str):
            config = GptOssConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        else:
            config = pretrained_model_name_or_path
        return cls(config, moe_config, backend)

    def __init__(
        self,
        config: GptOssConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = GptOssModel(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(hidden) if self.lm_head else hidden
        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        self.to(dtype)
        with buffer_device:
            self.model.freqs_cis = precompute_freqs_cis(
                self.config.head_dim,
                self.model.max_seq_len,
                self.config.rope_theta,
                self.config.rope_scaling,
            )
            self.model.freqs_cis = self.model.freqs_cis.to(buffer_device)
