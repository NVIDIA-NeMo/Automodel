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

"""Qwen3.5-MoE (VL) NeMo Automodel support."""

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.qwen3_5_moe is not available."})


try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeForConditionalGeneration,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeModelOutputWithPast,
        Qwen3_5MoeTextRotaryEmbedding,
        Qwen3_5MoeVisionRotaryEmbedding,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeModel as HFQwen3_5MoeModel,
    )

    _QWEN3_5_MOE_HF_AVAILABLE = True
except ModuleNotFoundError:
    _QWEN3_5_MOE_HF_AVAILABLE = False
    Qwen3_5MoeConfig = _make_missing("Qwen3_5MoeConfig")
    Qwen3_5MoeTextConfig = _make_missing("Qwen3_5MoeTextConfig")
    HFQwen3_5MoeForConditionalGeneration = _make_missing("Qwen3_5MoeForConditionalGeneration")
    Qwen3_5MoeGatedDeltaNet = _make_missing("Qwen3_5MoeGatedDeltaNet")
    Qwen3_5MoeModelOutputWithPast = _make_missing("Qwen3_5MoeModelOutputWithPast")
    Qwen3_5MoeTextRotaryEmbedding = _make_missing("Qwen3_5MoeTextRotaryEmbedding")
    Qwen3_5MoeVisionRotaryEmbedding = _make_missing("Qwen3_5MoeVisionRotaryEmbedding")
    HFQwen3_5MoeModel = _make_missing("Qwen3_5MoeModel")

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from nemo_automodel.components.models.qwen3_next.model import Block
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Qwen3_5MoeStateDictAdapter


class Qwen3_5MoeBlock(Block):
    """Block that uses the Qwen3.5-MoE native GatedDeltaNet (separate in_proj_qkv,
    in_proj_z, in_proj_b, in_proj_a)"""

    def __init__(self, layer_idx, config, moe_config, backend):
        super().__init__(layer_idx, config, moe_config, backend)
        # Replace the Qwen3Next fused GatedDeltaNet with the Qwen3.5-MoE native one
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        if self.layer_type == "full_attention":
            self.self_attn.init_weights(buffer_device)
        elif self.layer_type == "linear_attention":
            self.linear_attn.dt_bias.data.fill_(1.0)
            self.linear_attn.A_log.data.uniform_(0, 16).log_()
            linear_list = [
                self.linear_attn.in_proj_qkv,
                self.linear_attn.in_proj_z,
                self.linear_attn.in_proj_b,
                self.linear_attn.in_proj_a,
                self.linear_attn.out_proj,
            ]
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            self.linear_attn.norm.reset_parameters()
        self.mlp.init_weights(buffer_device)


class Fp32SafeQwen3_5MoeTextRotaryEmbedding(Qwen3_5MoeTextRotaryEmbedding):
    """Ensure inv_freq stays in float32 across ``.to(dtype)`` calls."""

    def _apply(self, fn: Any, recurse: bool = True):
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer(
            "inv_freq",
            inv_freq_fp32.to(device=self.inv_freq.device),
            persistent=False,
        )
        return result


class Fp32SafeQwen3_5MoeVisionRotaryEmbedding(Qwen3_5MoeVisionRotaryEmbedding):
    """Ensure the vision rotary inv_freq buffer remains float32."""

    def _apply(self, fn: Any, recurse: bool = True):
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer(
            "inv_freq",
            inv_freq_fp32.to(device=self.inv_freq.device),
            persistent=False,
        )
        return result


# ---------------------------------------------------------------------------
# VL composite model (wraps HF Qwen3_5MoeModel to expose backend language_model)
# ---------------------------------------------------------------------------
class Qwen3_5MoeModel(HFQwen3_5MoeModel):
    """Thin wrapper that exposes ``language_model`` internals as properties
    expected by the NeMo training loop (e.g. ``model.layers``)."""

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        embed_tokens = self.get_input_embeddings()
        if inputs_embeds is None:
            if embed_tokens is not None:
                inputs_embeds = embed_tokens(input_ids)
            elif (
                input_ids is not None
                and isinstance(input_ids, torch.Tensor)
                and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                # Pipeline-parallel: input_ids may already be embeddings
                inputs_embeds = input_ids
                input_ids = None
            else:
                raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        # If we have pixel values and a vision encoder, go through the full HF
        # VL forward (vision encoding + multimodal scatter + text).
        if pixel_values is not None and self.visual is not None:
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        # Text-only path: call the NeMo backend language model directly.
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        return outputs


# ---------------------------------------------------------------------------
# Text decoder backend (replaces HF Qwen3_5MoeTextModel with NeMo blocks)
# ---------------------------------------------------------------------------
class Qwen3_5MoeTextModelBackend(nn.Module):
    """Qwen3.5-MoE text decoder rebuilt on top of the Qwen3-Next Block."""

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        # --------------- MoE config ---------------
        # Qwen3.5-MoE has MoE on every layer, with a shared expert + sigmoid gate.
        # No ``decoder_sparse_step`` — defaults to 1 so every layer is MoE.
        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.hidden_size,  # unused — no dense MLP layers
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=1,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.001),
            norm_topk_prob=True,  # Qwen3.5-MoE always normalises topk weights
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=True,
            shared_expert_gate=True,
            shared_expert_inter_dim=config.shared_expert_intermediate_size,
        )

        # --------------- Layers ---------------
        embed_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=embed_dtype)

        # Use Qwen3_5MoeBlock — same as Qwen3Next Block but with native GatedDeltaNet
        self.layers = nn.ModuleDict(
            {
                str(layer_id): Qwen3_5MoeBlock(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )

        # Use Qwen3NextRMSNorm (1+weight formula)
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # M-RoPE (interleaved) — use HF implementation, kept in fp32
        self.rotary_emb = Fp32SafeQwen3_5MoeTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        **attn_kwargs: Any,
    ) -> Qwen3_5MoeModelOutputWithPast:
        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache is not supported for the Qwen3.5-MoE backend implementation.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # --- M-RoPE position handling (3-D: temporal / height / width) ---
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Qwen3.5-MoE uses [4, bs, seq] position_ids where dim-0 is [text, T, H, W].
        # We strip the text positions (dim 0) and keep [T, H, W] for M-RoPE.
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = inputs_embeds

        # Compute M-RoPE (cos, sin) via HF rotary emb, then convert to freqs_cis
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        head_dim = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)

        # --- Decoder layers (Qwen3Next Block, unmodified) ---
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                x=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            rope_deltas=None,
        )

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

        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)


# ---------------------------------------------------------------------------
# Top-level conditional generation model
# ---------------------------------------------------------------------------
class Qwen3_5MoeForConditionalGeneration(HFCheckpointingMixin, HFQwen3_5MoeForConditionalGeneration, MoEFSDPSyncMixin):
    """Qwen3.5-MoE VL conditional generation model using NeMo backend components.

    Inherits the HF model to reuse:
      * Vision encoder (``Qwen3_5MoeVisionModel``)
      * VL forward logic (image/video scatter, M-RoPE position computation)
      * ``prepare_inputs_for_generation`` / ``_expand_inputs_for_generation``

    Replaces:
      * ``model.language_model`` with ``Qwen3_5MoeTextModelBackend``
      * ``lm_head`` with NeMo backend linear
    """

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5MoeConfig,
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
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        config = Qwen3_5MoeConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        backend = backend or BackendConfig()
        # Initialize HF parent (creates self.model, self.lm_head, vision encoder, etc.)
        super().__init__(config)

        self.backend = backend

        # Swap HF model wrapper with our NeMo-aware version
        self.model.__class__ = Qwen3_5MoeModel

        # Replace HF text decoder with our NeMo backend
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.model.language_model = Qwen3_5MoeTextModelBackend(text_config, backend=self.backend, moe_config=moe_config)

        # Replace lm_head with NeMo backend linear
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )

        # Expose moe_config for FSDP sync mixin
        self.model.moe_config = self.model.language_model.moe_config

        self.vocab_size = text_config.vocab_size
        pad_token_id = getattr(text_config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        # State dict adapter for checkpoint conversion
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5MoeStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=get_dtype(text_config.torch_dtype, torch.bfloat16),
            )

        # Wrap vision rotary embedding with fp32-safe version
        vision_model = getattr(self.model, "visual")
        rotary = vision_model.rotary_pos_emb
        dim = rotary.inv_freq.shape[0] * 2
        fp32_safe_rotary = Fp32SafeQwen3_5MoeVisionRotaryEmbedding(dim)
        fp32_safe_rotary.register_buffer(
            "inv_freq",
            rotary.inv_freq.detach().clone().to(torch.float32, copy=True),
            persistent=False,
        )
        fp32_safe_rotary.to(rotary.inv_freq.device)
        vision_model.rotary_pos_emb = fp32_safe_rotary

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # PP VLM support: retrieve pixel_values from stored chunks if not passed
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if (
            pixel_values is None
            and hasattr(self, "_vlm_pixel_values_chunks")
            and self._vlm_pixel_values_chunks is not None
        ):
            image_token_id = self.config.image_token_id
            vision_start_token_id = self.config.vision_start_token_id
            has_media_tokens = input_ids is not None and (
                (input_ids == image_token_id).any() or (input_ids == vision_start_token_id).any()
            )

            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                    if image_grid_hws is not None and image_grid_hws.numel() > 0:
                        if image_grid_hws.shape[-1] == 2:
                            ones = torch.ones(
                                image_grid_hws.shape[0], 1, dtype=image_grid_hws.dtype, device=image_grid_hws.device
                            )
                            image_grid_thw = torch.cat([ones, image_grid_hws], dim=-1)
                        else:
                            image_grid_thw = image_grid_hws
                    kwargs["pixel_values"] = pixel_values
                    kwargs["image_grid_thw"] = image_grid_thw
                    self._vlm_chunk_idx = chunk_idx + 1

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states

        return logits

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config

        with buffer_device:
            language_model = self.model.language_model
            try:
                language_model.init_weights(buffer_device=buffer_device)
            except TypeError:
                language_model.init_weights()
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


if _QWEN3_5_MOE_HF_AVAILABLE:
    ModelClass = Qwen3_5MoeForConditionalGeneration
