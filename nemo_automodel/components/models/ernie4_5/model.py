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

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.models.ernie4_5.configuration_ernie4_5 import Ernie4_5Config
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.ernie4_5.rope_utils import Ernie4_5RotaryEmbedding, apply_rotary_pos_emb
from nemo_automodel.components.models.ernie4_5.state_dict_adapter import (
    Ernie4_5_MoeStateDictAdapter,
    Ernie4_5StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _config_dtype(config: Any) -> torch.dtype:
    return get_dtype(getattr(config, "torch_dtype", getattr(config, "dtype", None)), torch.bfloat16)


class Ernie4_5Attention(nn.Module):
    """ERNIE 4.5 GQA attention with interleaved RoPE."""

    def __init__(self, config: Ernie4_5Config | Ernie4_5_MoeConfig, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
            dtype=_config_dtype(config),
        )
        self.k_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.use_bias,
            dtype=_config_dtype(config),
        )
        self.v_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.use_bias,
            dtype=_config_dtype(config),
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.use_bias,
            dtype=_config_dtype(config),
        )
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q,
            k,
            v,
            attention_mask,
            self.backend.attn,
            **attn_kwargs,
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        return self.o_proj(out.flatten(flatten_dim))


class Ernie4_5Block(nn.Module):
    """Dense ERNIE 4.5 decoder block."""

    def __init__(self, config: Ernie4_5Config | Ernie4_5_MoeConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = Ernie4_5Attention(config, backend)
        self.mlp = MLP(
            config.hidden_size,
            config.intermediate_size,
            backend.linear,
            dtype=_config_dtype(config),
            activation="swiglu",
            bias=config.use_bias,
        )
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        del padding_mask
        attn_out = self.self_attn(
            self.input_layernorm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Ernie4_5MoeBlock(nn.Module):
    """ERNIE 4.5 MoE decoder block."""

    def __init__(self, layer_idx: int, config: Ernie4_5_MoeConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = Ernie4_5Attention(config, backend)
        is_moe_layer = (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        )
        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
        else:
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=_config_dtype(config),
                activation="swiglu",
                bias=config.use_bias,
            )
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            self.input_layernorm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out
        mlp_in = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoE):
            x = x + self.mlp(mlp_in, padding_mask)
        else:
            x = x + self.mlp(mlp_in)
        return x


class Ernie4_5Model(nn.Module):
    """Dense ERNIE 4.5 transformer body."""

    def __init__(self, config: Ernie4_5Config, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.backend = backend
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=_config_dtype(config),
        )
        self.layers = nn.ModuleDict(
            {str(layer_idx): Ernie4_5Block(config, backend) for layer_idx in range(config.num_hidden_layers)}
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )
        self.rotary_emb = Ernie4_5RotaryEmbedding(config)

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
            if input_ids.ndim == 1:
                position_ids = torch.arange(0, input_ids.shape[0], device=input_ids.device)
            else:
                position_ids = (
                    torch.arange(0, input_ids.shape[1], device=input_ids.device)
                    .unsqueeze(0)
                    .expand(input_ids.shape[0], -1)
                )
        qkv_format = attn_kwargs.get("qkv_format", "bshd")
        position_embeddings = self.rotary_emb(input_ids, position_ids, qkv_format=qkv_format)

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids
        for layer in self.layers.values():
            h = layer(
                h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
        return self.norm(h)


class Ernie4_5_MoeModel(nn.Module):
    """ERNIE 4.5 MoE transformer body."""

    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.moe_num_experts,
            n_shared_experts=config.moe_num_shared_experts,
            n_activated_experts=config.moe_k,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax_with_bias",
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.0),
            norm_topk_prob=True,
            expert_bias=config.use_bias,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            shared_expert_inter_dim=config.moe_intermediate_size,
            shared_expert_activation="swiglu",
            force_e_score_correction_bias=True,
            dtype=_config_dtype(config),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=_config_dtype(config),
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Ernie4_5MoeBlock(layer_idx, config, self.moe_config, backend)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=_config_dtype(config),
        )
        self.rotary_emb = Ernie4_5RotaryEmbedding(config)

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
            if input_ids.ndim == 1:
                position_ids = torch.arange(0, input_ids.shape[0], device=input_ids.device)
            else:
                position_ids = (
                    torch.arange(0, input_ids.shape[1], device=input_ids.device)
                    .unsqueeze(0)
                    .expand(input_ids.shape[0], -1)
                )
        qkv_format = attn_kwargs.get("qkv_format", "bshd")
        position_embeddings = self.rotary_emb(input_ids, position_ids, qkv_format=qkv_format)

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids
        for layer in self.layers.values():
            h = layer(
                h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
        return self.norm(h)


class Ernie4_5ForCausalLM(HFCheckpointingMixin, nn.Module):
    """Dense ERNIE 4.5 causal language model."""

    supports_gradient_checkpointing = True
    _skip_init_weights_on_load = True
    _nemo_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @classmethod
    def from_config(
        cls,
        config: Ernie4_5Config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = Ernie4_5Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Ernie4_5Config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = Ernie4_5Model(config, self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=_config_dtype(config),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Ernie4_5StateDictAdapter(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
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
        if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            hidden = hidden[:, slice_indices, :]
        logits = self.lm_head(hidden)
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
        return logits


class Ernie4_5_MoeForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """ERNIE 4.5 MoE causal language model with AutoModel EP support."""

    supports_gradient_checkpointing = True
    _skip_init_weights_on_load = True
    _nemo_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @classmethod
    def from_config(
        cls,
        config: Ernie4_5_MoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = Ernie4_5_MoeConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Ernie4_5_MoeModel(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=config.use_bias,
            dtype=_config_dtype(config),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Ernie4_5_MoeStateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=_config_dtype(config),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
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
        if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            hidden = hidden[:, slice_indices, :]
        logits = self.lm_head(hidden)
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
        return logits


ModelClass = Ernie4_5_MoeForCausalLM
