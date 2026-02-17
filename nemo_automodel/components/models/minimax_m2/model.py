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

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.minimax_m2.layers import MiniMaxM2Attention
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE, MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    """MiniMax-M2 transformer block with attention + MoE."""

    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = MiniMaxM2Attention(config, backend)
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


class MiniMaxM2Model(nn.Module):
    def __init__(self, config: Any, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config
        # Keep compatibility with generic MoE utilities that read config.num_experts.
        self.config.num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", None))

        score_func = getattr(config, "scoring_func", "sigmoid")
        score_func = "softmax" if str(score_func).lower() == "softmax" else "sigmoid"

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
            score_func=score_func,
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.0),
            norm_topk_prob=True,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=(score_func == "softmax"),
            force_e_score_correction_bias=True,
            dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)

        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Resolve partial_rotary_factor the same way HF does: from rope_parameters dict,
        # defaulting to 1.0. Note: config.rotary_dim is NOT used by HF's RoPE init.
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        else:
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=getattr(config, "rope_theta", 10000),
            dtype=torch.float32,
            initial_context_length=4096,
            scaling_factor=1.0,
            ntk_alpha=1.0,
            ntk_beta=32.0,
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
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
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class MiniMaxM2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    @classmethod
    def from_config(
        cls,
        config: Any,
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
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f"Could not load config from {pretrained_model_name_or_path}. "
                "Make sure the model path contains a valid config.json."
            ) from e
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = MiniMaxM2Model(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM2StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
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
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
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
            self.model.rotary_emb.device = buffer_device


ModelClass = MiniMaxM2ForCausalLM
