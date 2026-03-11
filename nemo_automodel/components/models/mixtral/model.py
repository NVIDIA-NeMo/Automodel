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

"""Custom Mixtral MoE model with the NeMo Automodel MoE extension.

Mixtral is a sparse Mixture-of-Experts model where every decoder layer
uses a top-k router to select from ``num_local_experts`` experts.

This implementation re-uses the shared MoE building blocks
(``MoE``, ``MLP``, ``Gate``, ``MoEFSDPSyncMixin``, expert-parallel
sharding) so that Mixtral benefits from the same EP / FSDP / CP
infrastructure as Qwen3-MoE, DeepSeek-V3, etc.
"""

from typing import Any

import torch
import torch.nn as nn
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.mixtral.state_dict_adapter import MixtralStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class MixtralAttention(nn.Module):
    """Mixtral GQA attention (RoPE, no per-head QK norm)."""

    def __init__(self, config: MixtralConfig, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)

        attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, attention_bias
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, attention_bias
        )

        softmax_scale = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=softmax_scale,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk

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
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_emb_qk(
            q,
            k,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.backend.rope_fusion,
            cu_seqlens=attn_kwargs.get("cu_seqlens", None),
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        out = self.o_proj(out.flatten(flatten_dim))
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)


class Block(nn.Module):
    """Single Mixtral decoder block (attention + MoE feed-forward)."""

    def __init__(self, layer_idx: int, config: MixtralConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = MixtralAttention(config, backend)
        self.mlp = MoE(moe_config, backend)
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(x), padding_mask)
        x = x + mlp_out
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class MixtralModel(nn.Module):
    """Mixtral transformer backbone (embeddings + decoder blocks + final norm)."""

    def __init__(self, config: MixtralConfig, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config

        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.intermediate_size,
            n_routed_experts=config.num_local_experts,
            n_shared_experts=0,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.0),
            norm_topk_prob=True,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=True,
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        base, rope_scaling, _ = get_rope_config(config)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
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

        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
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
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class MixtralForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Mixtral sparse-MoE causal LM with NeMo EP / FSDP / CP support."""

    @classmethod
    def from_config(
        cls,
        config: MixtralConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = MixtralConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: MixtralConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = MixtralModel(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MixtralStateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(hidden) if self.lm_head else hidden
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
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

        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = MixtralForCausalLM
