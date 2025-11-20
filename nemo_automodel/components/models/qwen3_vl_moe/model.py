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
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel as HFQwen3VLMoeModel,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextRotaryEmbedding,
)

from nemo_automodel.components.models.qwen3_moe.model import Block
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Qwen3VLMoeStateDictAdapter


class Fp32SafeQwen3VLMoeTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    """Ensure inv_freq stays in float32 while still following device moves."""

    def _apply(self, fn: Any, recurse: bool = True):
        # Keep an fp32 copy before super mutates the registered buffer
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        # Restore dtype while honoring the new device placement
        self.register_buffer(
            "inv_freq",
            inv_freq_fp32.to(device=self.inv_freq.device),
            persistent=False,
        )
        return result

def save_tensor(tensor: torch.Tensor, path: str) -> None:
    import pickle
    if torch.distributed.get_rank() == 0:
        with open("auto_" + path, "wb") as f:
            pickle.dump(tensor, f)


class Qwen3VLMoeModel(HFQwen3VLMoeModel):
    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm


class Qwen3VLMoeTextModelBackend(nn.Module):
    """Qwen3-VL text decoder rebuilt on top of the Qwen3-MoE block implementation."""

    def __init__(self, config: Qwen3VLMoeTextConfig, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=getattr(config, "moe_intermediate_size", config.intermediate_size),
            n_routed_experts=getattr(config, "num_experts", 0),
            n_shared_experts=0,
            n_activated_experts=getattr(config, "num_experts_per_tok", 1),
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.0),
            norm_topk_prob=getattr(config, "norm_topk_prob", False),
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=True,
        )

        embed_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=embed_dtype)
        self.layers = nn.ModuleList(
            [Block(layer_id, config, self.moe_config, backend) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fp32SafeQwen3VLMoeTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        **attn_kwargs: Any,
    ) -> Qwen3VLMoeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache is not supported for the Qwen3-VL backend implementation.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask.bool().logical_not()

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        head_dim = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)

        save_tensor(text_position_ids, "text_position_ids.pkl")
        save_tensor(freqs_cis, "freqs_cis.pkl")
        save_tensor(hidden_states, "hidden_states_before_layers.pkl")

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                x=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            rope_deltas=None,
        )

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if visual_pos_masks is None:
            return hidden_states

        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device

        for layer in self.layers:
            layer.init_weights(buffer_device=buffer_device)


class Qwen3VLMoeForConditionalGeneration(HFQwen3VLMoeForConditionalGeneration, MoEFSDPSyncMixin):
    """Qwen3-VL conditional generation model using the Qwen3-MoE backend components."""

    @classmethod
    def from_config(
        cls,
        config: Qwen3VLMoeConfig,
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
        config = Qwen3VLMoeConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        backend = backend or BackendConfig()
        super().__init__(config)

        self.backend = backend
        self.model.__class__ = Qwen3VLMoeModel

        text_config = config.text_config if hasattr(config, "text_config") else config
        self.model.language_model = Qwen3VLMoeTextModelBackend(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.model.moe_config = self.model.language_model.moe_config

        self.vocab_size = text_config.vocab_size
        self.pad_token_id = text_config.pad_token_id if text_config.pad_token_id is not None else -1

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3VLMoeStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=get_dtype(text_config.torch_dtype, torch.bfloat16),
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask
        elif "padding_mask" in kwargs and kwargs["padding_mask"] is None:
            kwargs.pop("padding_mask")

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config

        with buffer_device:
            self.model.language_model.init_weights(buffer_device=buffer_device)
            final_out_std = text_config.hidden_size**-0.5
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
            self.model.language_model.rotary_emb.device = buffer_device


ModelClass = Qwen3VLMoeForConditionalGeneration

