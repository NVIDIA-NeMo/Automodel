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

from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.deepseek_v3.layers import MLA
from nemo_automodel.components.models.deepseek_v3.model import Block
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb_qk,
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)
from nemo_automodel.components.models.mistral4.state_dict_adapter import (
    Mistral4MultimodalStateDictAdapter,
    Mistral4StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _get_llama_4_attn_scale(position_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    """Position-dependent attention scaling for long-context extrapolation (Llama 4 / Mistral 4)."""
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids.float() / max_position_embeddings))
    return scaling.unsqueeze(-1)


class Mistral4MLA(MLA):
    """MLA with Llama 4 attention scaling for Mistral 4.

    Compared to DeepSeek V3 MLA, adds position-dependent scaling to q_pe after RoPE
    (llama_4_scaling_beta). RoPE itself uses the same complex-number approach as DSV3.
    """

    def __init__(self, config, backend: BackendConfig):
        super().__init__(config, backend)
        rope_parameters = config.rope_parameters if hasattr(config, "rope_parameters") else config.rope_scaling
        self.llama_4_scaling_beta = rope_parameters.get("llama_4_scaling_beta") if rope_parameters else None
        self.llama_4_orig_max_pos = rope_parameters.get("original_max_position_embeddings") if rope_parameters else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ):
        from nemo_automodel.components.attention.utils import (
            postprocess_output_for_attn,
            preprocess_args_and_kwargs_for_attn,
        )

        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, local_seq_len, _ = x.size()

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        if qkv_format == "thd":
            q = q.view(num_tokens, self.n_heads, self.qk_head_dim)
        else:
            q = q.view(bsz, local_seq_len, self.n_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)

        head_unsqueeze_dim = 2 if qkv_format == "bshd" else 1
        k_pe = k_pe.unsqueeze(head_unsqueeze_dim)

        # Apply RoPE (same complex-number approach as DSV3)
        cu_seqlens = attn_kwargs.get("cu_seqlens", None)
        q_pe, k_pe = apply_rotary_emb_qk(
            q_pe,
            k_pe,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.rope_fusion,
            cu_seqlens=cu_seqlens,
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )

        # Llama 4 attention scaling on q_pe (no-op for positions < orig_max_pos)
        if self.llama_4_scaling_beta is not None:
            position_ids = attn_kwargs.get("position_ids", None)
            if position_ids is not None:
                attn_scale = _get_llama_4_attn_scale(
                    position_ids, self.llama_4_scaling_beta, self.llama_4_orig_max_pos
                ).to(q_pe.dtype)
                q_pe = q_pe * attn_scale.unsqueeze(-1)

        k_pe = k_pe.squeeze(head_unsqueeze_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.kv_b_proj(kv)
        if qkv_format == "thd":
            kv = kv.view(num_tokens, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(1).expand([num_tokens, self.n_heads, self.qk_rope_head_dim])
        else:
            kv = kv.view(bsz, local_seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(2).expand([bsz, local_seq_len, self.n_heads, self.qk_rope_head_dim])

        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )

        x = self.attn_func(q, k, v, **_attn_kwargs)
        x = postprocess_output_for_attn(x, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        x = self.o_proj(x.flatten(flatten_dim))
        return x


class Mistral4Block(Block):
    """Block using Mistral4MLA instead of MLA."""

    def __init__(self, layer_idx, config, moe_config, backend):
        super().__init__(layer_idx, config, moe_config, backend)
        # Replace the MLA with Mistral4MLA
        self.self_attn = Mistral4MLA(config, backend)


def _build_moe_config(config, moe_overrides: dict | None = None) -> MoEConfig:
    """Build MoEConfig from a Mistral4 text config."""
    moe_defaults = dict(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="softmax_with_bias",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0,
        norm_topk_prob=config.norm_topk_prob,
        dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
    )
    if moe_overrides:
        moe_defaults.update(moe_overrides)
    return MoEConfig(**moe_defaults)


class Mistral4Model(nn.Module):
    def __init__(
        self,
        config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        self.moe_config = moe_config or _build_moe_config(config, moe_overrides=moe_overrides)

        # Resolve model dtype once; thread it explicitly to every sub-module
        # so fp32 master weights work even when construction is not wrapped in
        # local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Mistral4Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        self.max_seq_len = config.max_position_embeddings
        rope_theta, rope_scaling, _ = get_rope_config(config)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.qk_rope_head_dim,
                self.max_seq_len,
                rope_theta,
                rope_scaling,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
            )

        with torch.no_grad():
            freqs_cis = freqs_cis_from_position_ids(
                position_ids,
                self.freqs_cis,
                qkv_format=attn_kwargs.get("qkv_format", "bshd"),
                for_fused_rope=self.backend.rope_fusion,
                cp_size=attn_kwargs.get("cp_size", 1),
            )

        h = inputs_embeds

        # Pass position_ids through attn_kwargs for Mistral4MLA's llama_4_scaling_beta
        attn_kwargs["position_ids"] = position_ids

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

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for _, block in self.layers.named_children():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            rope_theta, rope_scaling, _ = get_rope_config(self.config)
            self.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.max_seq_len,
                rope_theta,
                rope_scaling,
            )
            self.freqs_cis = self.freqs_cis.to(buffer_device)
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class Mistral4ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    @classmethod
    def from_config(
        cls,
        config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        # Extract text_config if this is a multimodal wrapper config
        text_config = getattr(config, "text_config", config)
        return cls(text_config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        # Extract text_config if this is a multimodal wrapper config
        config = getattr(config, "text_config", config)
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Mistral4Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Mistral4StateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
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

        logits = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(logits) if self.lm_head else logits
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
        return logits

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for _, block in self.model.layers.named_children():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

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
            rope_theta, rope_scaling, _ = get_rope_config(self.config)
            self.model.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.model.max_seq_len,
                rope_theta,
                rope_scaling,
            )


# =============================================================================
# Multimodal: Mistral3ForConditionalGeneration
# Pixtral vision tower + Mistral3MultiModalProjector + Mistral4 text backbone
# =============================================================================

try:
    from transformers.models.mistral3.modeling_mistral3 import (  # noqa: F401
        Mistral3ForConditionalGeneration as HFMistral3ForConditionalGeneration,
    )
    from transformers.models.mistral3.modeling_mistral3 import (  # noqa: F401
        Mistral3Model as HFMistral3Model,
    )

    _HF_MISTRAL3_AVAILABLE = True
except ImportError:
    _HF_MISTRAL3_AVAILABLE = False


if _HF_MISTRAL3_AVAILABLE:
    from transformers.modeling_outputs import BaseModelOutputWithPast

    class Mistral4TextModelBackend(nn.Module):
        """Backend-aware Mistral4 text model for use inside the multimodal wrapper.

        Wraps Mistral4Model in self.model (like KimiK25VLLanguageModelBackend wraps
        DeepseekV3Model). This ensures embed_tokens/layers/norm are accessed via
        @property aliases rather than as direct nn.Module children, which avoids
        FSDP double-root-init when the parallelizer wraps both embed_tokens and
        this module.
        """

        def __init__(
            self,
            config,
            backend: BackendConfig,
            *,
            moe_config: MoEConfig | None = None,
            moe_overrides: dict | None = None,
        ):
            super().__init__()
            if moe_config is not None and moe_overrides is not None:
                raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
            self.model = Mistral4Model(
                config,
                backend,
                moe_config=moe_config,
                moe_overrides=moe_overrides,
            )
            self.moe_config = self.model.moe_config
            # lm_head lives inside language_model (like KimiVLLanguageModelBackend)
            # so the parallelizer wraps it as part of _model, matching the Kimi pattern.
            self.lm_head = initialize_linear_module(
                backend.linear,
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
            )

        @property
        def embed_tokens(self):
            return self.model.embed_tokens

        @property
        def layers(self):
            return self.model.layers

        @property
        def norm(self):
            return self.model.norm

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def init_weights(self, buffer_device: torch.device | None = None):
            self.model.init_weights(buffer_device)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            *,
            inputs_embeds: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            padding_mask: torch.Tensor | None = None,
            # HF kwargs accepted but unused by our backend
            past_key_values=None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> BaseModelOutputWithPast:
            h = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                padding_mask=padding_mask,
                **kwargs,
            )
            return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=None)

    class Mistral3Model(nn.Module):
        """VLM wrapper composing vision tower + projector + Mistral4 text backend.

        Follows KimiK25VLModel pattern: plain nn.Module (not HF PreTrainedModel)
        to avoid FSDP conflicts from PreTrainedModel's module registration hooks.
        Vision processing logic is replicated from HF Mistral3Model.
        """

        def __init__(self, config, vision_tower, multi_modal_projector, language_model):
            super().__init__()
            self.config = config
            self.vision_tower = vision_tower
            self.multi_modal_projector = multi_modal_projector
            self.language_model = language_model

        @property
        def layers(self):
            return self.language_model.layers

        @property
        def embed_tokens(self):
            return self.language_model.embed_tokens

        @property
        def norm(self):
            return self.language_model.norm

        def get_input_embeddings(self):
            return self.language_model.get_input_embeddings()

        def _get_image_features(self, pixel_values, image_sizes, vision_feature_layer=-1):
            """Encode images through vision tower + projector (from HF Mistral3Model)."""
            image_outputs = self.vision_tower(
                pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
            )
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
            downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
            split_sizes = (
                (torch.as_tensor(image_sizes, device=image_features.device) // downsample_ratio).prod(dim=-1).tolist()
            )
            return torch.split(image_features.squeeze(0), split_sizes)

        def forward(
            self,
            input_ids=None,
            pixel_values=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            image_sizes=None,
            padding_mask=None,
            **kwargs,
        ):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                embed_tokens = self.language_model.get_input_embeddings()
                if embed_tokens is not None:
                    inputs_embeds = embed_tokens(input_ids)
                elif (
                    input_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    inputs_embeds = input_ids
                    input_ids = None
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

            if pixel_values is not None and self.vision_tower is not None:
                image_features = self._get_image_features(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    vision_feature_layer=getattr(self.config, "vision_feature_layer", -1),
                )
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

                # Merge image features into text embeddings at image token positions
                image_token_index = getattr(self.config, "image_token_index", 10)
                special_image_mask = (
                    (input_ids == image_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            hidden_states = self.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                padding_mask=padding_mask,
                **kwargs,
            )
            return hidden_states

    from transformers import AutoModel
    from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

    class Mistral3ForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
        """Full multimodal Mistral 4: Pixtral vision + projector + Mistral4 MLA/MoE text backbone.

        Follows KimiK25VLForConditionalGeneration pattern: inherits from nn.Module
        (not HF PreTrainedModel) to avoid FSDP conflicts.
        """

        @classmethod
        def supports_config(cls, config) -> bool:
            """Only handle configs whose text backbone is Mistral4 (MoE + MLA)."""
            text_config = getattr(config, "text_config", None)
            return text_config is not None and getattr(text_config, "model_type", None) == "mistral4"

        @classmethod
        def from_config(
            cls,
            config,
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
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return cls.from_config(config, *model_args, **kwargs)

        def __init__(
            self,
            config,
            moe_config: MoEConfig | None = None,
            backend: BackendConfig | None = None,
            **kwargs,
        ):
            super().__init__()
            backend = backend or BackendConfig()
            num_hidden_layers = kwargs.pop("num_hidden_layers", None)
            if num_hidden_layers is not None:
                config.text_config.num_hidden_layers = num_hidden_layers

            # _init_model() only overrides the top-level hf_config.torch_dtype;
            # for VL configs the nested text_config / vision_config keep their
            # original dtype (typically bf16 from the checkpoint's
            # config.json). Propagate the user-requested dtype to every nested
            # sub-config that exposes a torch_dtype attribute, before building
            # the vision tower / text backend.
            top_dtype = getattr(config, "torch_dtype", None)
            if top_dtype is not None:
                for sub_cfg in vars(config).values():
                    if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                        sub_cfg.torch_dtype = top_dtype

            self.config = config
            self.backend = backend
            text_config = config.text_config

            # Build components: vision tower from HF, projector from HF, text from our backend
            vision_tower = AutoModel.from_config(config.vision_config)
            multi_modal_projector = Mistral3MultiModalProjector(config)
            moe_overrides = kwargs.pop("moe_overrides", None)
            language_model = Mistral4TextModelBackend(
                text_config,
                backend=backend,
                moe_config=moe_config,
                moe_overrides=moe_overrides,
            )

            self.model = Mistral3Model(
                config=config,
                vision_tower=vision_tower,
                multi_modal_projector=multi_modal_projector,
                language_model=language_model,
            )
            self.moe_config = self.model.language_model.moe_config
            self.model.moe_config = self.moe_config

            self.vocab_size = text_config.vocab_size
            self.pad_token_id = getattr(text_config, "pad_token_id", -1) or -1
            self.image_token_index = getattr(config, "image_token_index", 10)

            if backend.enable_hf_state_dict_adapter:
                self.state_dict_adapter = Mistral4MultimodalStateDictAdapter(
                    config,
                    self.moe_config,
                    backend,
                    dtype=get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16),
                )

        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.language_model.set_input_embeddings(value)

        @property
        def lm_head(self):
            return self.model.language_model.lm_head

        def get_output_embeddings(self):
            return self.model.language_model.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.model.language_model.lm_head = new_embeddings

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            *,
            position_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            padding_mask: torch.Tensor | None = None,
            pixel_values: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            # PP VLM support: retrieve pixel_values from stored chunks
            if (
                pixel_values is None
                and hasattr(self, "_vlm_pixel_values_chunks")
                and self._vlm_pixel_values_chunks is not None
            ):
                has_media_tokens = (
                    input_ids is not None
                    and self.image_token_index is not None
                    and (input_ids == self.image_token_index).any()
                )
                if has_media_tokens:
                    chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                    if chunk_idx < len(self._vlm_pixel_values_chunks):
                        pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                        image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                        if image_grid_hws is not None:
                            image_sizes = image_grid_hws
                        self._vlm_chunk_idx = chunk_idx + 1

            if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
                input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                    input_ids, position_ids, padding_mask, kwargs
                )
                attention_mask = None

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                image_sizes=image_sizes,
                padding_mask=padding_mask,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs
            try:
                lm = self.lm_head
            except (AttributeError, TypeError):
                lm = None
            logits = lm(hidden_states) if lm is not None else hidden_states

            if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
                logits = logits.unsqueeze(0)

            return logits

        def update_moe_gate_bias(self) -> None:
            with torch.no_grad():
                for _, block in self.model.language_model.layers.named_children():
                    if isinstance(block.mlp, MoE):
                        block.mlp.gate.update_bias()

        @torch.no_grad()
        def initialize_weights(
            self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
        ) -> None:
            buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
            text_config = self.config.text_config

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


ModelClass = Mistral4ForCausalLM
