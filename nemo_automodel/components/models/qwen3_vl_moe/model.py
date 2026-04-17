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
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel as HFQwen3VLMoeModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeVisionRotaryEmbedding,
)

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_moe.model import Block
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Qwen3VLMoeStateDictAdapter


class Fp32SafeQwen3VLMoeTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    """Ensure inv_freq stays in float32"""

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


class Fp32SafeQwen3VLMoeVisionRotaryEmbedding(Qwen3VLMoeVisionRotaryEmbedding):
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


class Qwen3VLMoeBlock(Block):
    """Qwen3-VL block adapter that accepts HF-style position embeddings."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if freqs_cis is None and position_embeddings is not None:
            cos, sin = position_embeddings
            head_dim = cos.shape[-1] // 2
            freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)
        if freqs_cis is None:
            raise ValueError("Qwen3VLMoeBlock requires freqs_cis or position_embeddings.")
        return super().forward(
            x=x,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )


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
                inputs_embeds = input_ids
                input_ids = None
            else:
                raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        if (pixel_values is not None or pixel_values_videos is not None) and self.visual is not None:
            # Process vision features inline instead of delegating to
            # super().forward(input_ids=None).  This ensures get_placeholder_mask
            # receives the original input_ids so it can locate placeholder tokens
            # via fast integer comparison rather than the fragile bfloat16
            # embedding comparison path.
            image_mask = None
            video_mask = None

            has_images = pixel_values is not None
            has_videos = pixel_values_videos is not None

            if has_images and has_videos:
                # Merged visual forward to avoid two separate FSDP-synchronized calls
                pixel_values = pixel_values.type(self.visual.dtype)
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                merged_pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0)
                merged_grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)

                merged_output = self.visual(merged_pixel_values, grid_thw=merged_grid_thw, return_dict=True)
                merged_embeds = merged_output.pooler_output
                merged_deepstack = merged_output.deepstack_features

                spatial_merge_size_sq = self.visual.spatial_merge_size**2
                total_image_tokens = (image_grid_thw.prod(-1) // spatial_merge_size_sq).sum().item()

                image_embeds = merged_embeds[:total_image_tokens].to(inputs_embeds.device, inputs_embeds.dtype)
                video_embeds = merged_embeds[total_image_tokens:].to(inputs_embeds.device, inputs_embeds.dtype)

                deepstack_image_embeds = [ds[:total_image_tokens] for ds in merged_deepstack]
                deepstack_video_embeds = [ds[total_image_tokens:] for ds in merged_deepstack]

                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            elif has_images:
                image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
                image_embeds = image_outputs.pooler_output
                deepstack_image_embeds = image_outputs.deepstack_features
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            else:  # has_videos only
                video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
                video_embeds = video_outputs.pooler_output
                deepstack_video_embeds = video_outputs.deepstack_features
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Build deepstack visual masks (same logic as HF model)
            visual_pos_masks = None
            deepstack_visual_embeds = None
            if image_mask is not None and video_mask is not None:
                image_mask = image_mask[..., 0]
                video_mask = video_mask[..., 0]
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                image_mask = image_mask[..., 0]
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                video_mask = video_mask[..., 0]
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds

            # Compute mRoPE position_ids if not pre-computed
            if position_ids is None:
                mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
                attention_mask_tensor = (
                    attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention")
                )
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas

            outputs = self.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

            return Qwen3VLMoeModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=None,
                rope_deltas=getattr(self, "rope_deltas", None),
            )

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


class Qwen3VLMoeTextModelBackend(nn.Module):
    """Qwen3-VL text decoder rebuilt on top of the Qwen3-MoE block implementation."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
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

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        moe_defaults = dict(
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
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        embed_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=embed_dtype)
        self.layers = nn.ModuleDict(
            {
                str(layer_id): Qwen3VLMoeBlock(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
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
            position_ids = position_ids[1:]

        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = inputs_embeds

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        head_dim = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)

        for layer_id_str, decoder_layer in self.layers.items():
            layer_idx = int(layer_id_str)
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

        if self.norm is not None:
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

        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)


class Qwen3VLMoeForConditionalGeneration(HFCheckpointingMixin, HFQwen3VLMoeForConditionalGeneration, MoEFSDPSyncMixin):
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
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model.language_model = Qwen3VLMoeTextModelBackend(
            text_config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.model.moe_config = self.model.language_model.moe_config

        self.vocab_size = text_config.vocab_size
        pad_token_id = getattr(text_config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3VLMoeStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=get_dtype(text_config.torch_dtype, torch.bfloat16),
            )

        vision_model = getattr(self.model, "visual")
        rotary = vision_model.rotary_pos_emb
        dim = rotary.inv_freq.shape[0] * 2
        fp32_safe_rotary = Fp32SafeQwen3VLMoeVisionRotaryEmbedding(dim)
        fp32_safe_rotary.register_buffer(
            "inv_freq",
            rotary.inv_freq.detach().clone().to(torch.float32, copy=True),
            persistent=False,
        )
        fp32_safe_rotary.to(rotary.inv_freq.device)
        vision_model.rotary_pos_emb = fp32_safe_rotary

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        # PP VLM support: retrieve pixel_values from stored chunks if not passed directly
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if (
            pixel_values is None
            and hasattr(self, "_vlm_pixel_values_chunks")
            and self._vlm_pixel_values_chunks is not None
        ):
            # Check if we have media tokens in input_ids
            # 151655 = <|image_pad|>, 151656 = <|video_pad|>
            has_media_tokens = input_ids is not None and ((input_ids == 151655).any() or (input_ids == 151656).any())

            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                    # Convert image_grid_hws [N, 2] to image_grid_thw [N, 3] by prepending T=1
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

        # With pipeline parallelism, attention_mask (from batch kwargs) can have a
        # different sequence length than inputs_embeds (hidden states from prev stage).
        # Drop mismatched masks to avoid size errors in MoE token routing.
        if inputs_embeds is not None and attention_mask is not None:
            if attention_mask.shape[-1] != inputs_embeds.shape[1]:
                attention_mask = None
                padding_mask = None

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

        cast_model_to_dtype(self, dtype)

        with buffer_device:
            self.model.language_model.rotary_emb.device = buffer_device


ModelClass = Qwen3VLMoeForConditionalGeneration
