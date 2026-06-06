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

"""Qwen3.5 dense causal LM with Megatron-style MTP support."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    create_causal_mask,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5ForConditionalGeneration,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule, roll_tensor
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Qwen3_5DenseStateDictAdapter


@dataclass
class Qwen3_5CausalLMOutputWithPast(CausalLMOutputWithPast):
    """Qwen3.5 causal-LM output extended with MTP auxiliary hidden states."""

    rope_deltas: torch.Tensor | None = None
    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


def _resolve_mtp_num_layers(config: Any, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    value = getattr(config, "num_nextn_predict_layers", None)
    if value is None:
        value = getattr(config, "mtp_num_hidden_layers", 0)
    return int(value or 0)


def build_mtp_config_from_hf(
    config: Any,
    *,
    loss_scaling_factor: float = 0.1,
    num_nextn_predict_layers: int | None = None,
) -> MTPConfig:
    """Build Qwen3.5 MTP runtime config from HF-style config fields."""
    num_layers = _resolve_mtp_num_layers(config, num_nextn_predict_layers)
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern="*" if num_layers > 0 else "",
        loss_scaling_factor=loss_scaling_factor,
    )


def _make_full_attention_config(config: Qwen3_5TextConfig, layer_idx: int) -> Qwen3_5TextConfig:
    mtp_config = copy.copy(config)
    layer_types = list(getattr(config, "layer_types", []) or [])
    if len(layer_types) <= layer_idx:
        fill = layer_types[-1] if layer_types else "full_attention"
        layer_types.extend([fill] * (layer_idx + 1 - len(layer_types)))
    layer_types[layer_idx] = "full_attention"
    mtp_config.layer_types = layer_types
    mtp_config.num_hidden_layers = max(int(getattr(config, "num_hidden_layers", 0) or 0), layer_idx + 1)
    return mtp_config


def _split_qwen3_5_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    past_key_values: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        base = torch.arange(seq_len, device=device) + past_seen_tokens
        position_ids = base.view(1, 1, -1).expand(4, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:], position_ids[0]
    return position_ids, None


def _rolled_embed_inputs(inputs_embeds: torch.Tensor, num_depths: int) -> tuple[torch.Tensor, ...]:
    embed_inputs = []
    cur = inputs_embeds
    for _ in range(num_depths):
        cur = roll_tensor(cur, shifts=-1, dim=-2)
        embed_inputs.append(cur)
    return tuple(embed_inputs)


class Qwen3_5DenseMTPSublayer(Qwen3_5DecoderLayer):
    """One full-attention Qwen3.5 dense MTP sublayer."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        *,
        has_fusion: bool = False,
        has_final_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(_make_full_attention_config(config, layer_idx), layer_idx)
        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm
        if has_fusion:
            self.enorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        if has_final_norm:
            self.final_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        rotary_emb: nn.Module,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.has_fusion:
            if embed_input is None:
                raise ValueError("first Qwen3.5 MTP sublayer requires embed_input")
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))

        if position_ids is None:
            seq_len = hidden_states.shape[-2]
            base = torch.arange(seq_len, device=hidden_states.device).view(1, -1)
            position_ids = base.expand(hidden_states.shape[0], -1)

        position_embeddings = rotary_emb(hidden_states, position_ids)
        hidden_states = super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids[0] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else None,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        init_std = float(getattr(self.self_attn.config, "initializer_range", 0.02))
        target_device = buffer_device or torch.device("cpu")
        with target_device:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, Qwen3_5RMSNorm):
                    nn.init.zeros_(module.weight)


def build_qwen3_5_dense_mtp(
    config: Qwen3_5TextConfig,
    mtp_config: MTPConfig,
    dtype: torch.dtype,
) -> MTPModule:
    """Construct dense Qwen3.5 MTP blocks.

    Qwen3.5 MTP follows Megatron Bridge: each depth is one full-attention
    Qwen3.5 decoder block, regardless of the backbone's GatedDeltaNet layers.
    """
    base_layer_idx = int(config.num_hidden_layers)

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        del depth, sublayer_idx, block_type
        return Qwen3_5DenseMTPSublayer(
            config,
            layer_idx=base_layer_idx + global_idx,
            has_fusion=has_fusion,
            has_final_norm=has_final_norm,
            dtype=dtype,
        )

    return MTPModule(
        mtp_config=mtp_config,
        block_types_per_sublayer=["full_attention"],
        sublayer_factory=factory,
    )


class Qwen3_5ForCausalLM(HFCheckpointingMixin, nn.Module):
    """Qwen3.5 dense causal LM with optional Megatron-style MTP head."""

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5TextConfig,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        config = Qwen3_5TextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.backend = backend or BackendConfig()

        self.model = Qwen3_5TextModel(config)
        dtype = next(self.model.parameters()).dtype
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

        self.mtp_config = build_mtp_config_from_hf(
            config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = build_qwen3_5_dense_mtp(config, self.mtp_config, dtype=dtype) if self.mtp_config.enabled else None

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5DenseStateDictAdapter()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Qwen3_5CausalLMOutputWithPast:
        del labels
        effective_use_cache = False if use_cache is None else use_cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=effective_use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            source_embeds = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
            rotary_position_ids, text_position_ids = _split_qwen3_5_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                past_key_values=past_key_values,
            )
            causal_mask = create_causal_mask(
                config=self.model.config,
                inputs_embeds=source_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
            if input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=self.model.embed_tokens,
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )

        return Qwen3_5CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None) or (hidden_states,),
            attentions=getattr(outputs, "attentions", None),
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        with buffer_device:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.padding_idx is not None:
                        module.weight[module.padding_idx].zero_()
                elif isinstance(module, Qwen3_5RMSNorm):
                    nn.init.zeros_(module.weight)
        cast_model_to_dtype(self, dtype)


class Qwen3_5ForConditionalGeneration(HFCheckpointingMixin, HFQwen3_5ForConditionalGeneration):
    """Qwen3.5/Qwen3.6 dense VLM with optional Megatron-style MTP head.

    The base VLM stays on the upstream HF implementation so image/video feature
    insertion, M-RoPE position handling, and generation helpers remain intact.
    MTP is added as an auxiliary train-time module over the final language
    hidden states, matching the dense text-only MTP architecture.
    """

    # forward() pulls per-microbatch pixel_values from _vlm_pixel_values_chunks;
    # patch_hf_model_for_pp must not replace it under PP.
    _pp_keep_self_forward: bool = True

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5Config,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        config = Qwen3_5Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5Config,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(config)
        self.backend = backend or BackendConfig()

        text_config = config.text_config
        param_dtype = next(self.model.language_model.parameters()).dtype
        dtype = get_dtype(getattr(text_config, "torch_dtype", None), param_dtype)
        self.mtp_config = build_mtp_config_from_hf(
            text_config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = build_qwen3_5_dense_mtp(text_config, self.mtp_config, dtype=dtype) if self.mtp_config.enabled else None
        if self.mtp is not None:
            cast_model_to_dtype(self.mtp, dtype)

    def _pop_staged_vlm_media(
        self,
        input_ids: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        pixel_values = kwargs.get("pixel_values", None)
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        has_media_tokens = input_ids is not None and (
            (input_ids == image_token_id).any()
            or (input_ids == video_token_id).any()
            or (input_ids == vision_start_token_id).any()
        )
        if not has_media_tokens:
            return pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw

        chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
        consumed_vlm_chunk = False

        if pixel_values is None:
            image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            if image_chunks is not None and chunk_idx < len(image_chunks):
                pixel_values = image_chunks[chunk_idx]
                image_grid_chunks = getattr(self, "_vlm_image_grid_hws_chunks", None)
                if image_grid_chunks is not None and chunk_idx < len(image_grid_chunks):
                    image_grid_hws = image_grid_chunks[chunk_idx]
                    if image_grid_hws is not None and image_grid_hws.numel() > 0:
                        if image_grid_hws.shape[-1] == 2:
                            ones = torch.ones(
                                image_grid_hws.shape[0], 1, dtype=image_grid_hws.dtype, device=image_grid_hws.device
                            )
                            image_grid_thw = torch.cat([ones, image_grid_hws], dim=-1)
                        else:
                            image_grid_thw = image_grid_hws
                consumed_vlm_chunk = True

        if pixel_values_videos is None:
            video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
            if video_chunks is not None and chunk_idx < len(video_chunks):
                video_chunk = video_chunks[chunk_idx]
                if video_chunk.numel() > 0:
                    pixel_values_videos = video_chunk
                    video_grid_chunks = getattr(self, "_vlm_video_grid_thw_chunks", None)
                    if video_grid_chunks is not None and chunk_idx < len(video_grid_chunks):
                        video_grid_thw = video_grid_chunks[chunk_idx]
                consumed_vlm_chunk = True

        if consumed_vlm_chunk:
            self._vlm_chunk_idx = chunk_idx + 1

        return pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Qwen3_5CausalLMOutputWithPast:
        effective_use_cache = False if use_cache is None and self.training else use_cache
        kwargs = dict(kwargs)
        if effective_use_cache is not None:
            kwargs["use_cache"] = effective_use_cache
        kwargs["pixel_values"] = pixel_values
        kwargs["pixel_values_videos"] = pixel_values_videos
        kwargs["image_grid_thw"] = image_grid_thw
        kwargs["video_grid_thw"] = video_grid_thw
        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw = self._pop_staged_vlm_media(input_ids, kwargs)

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask

        model_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        }
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **model_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            language_model = self.model.language_model
            source_embeds = inputs_embeds if inputs_embeds is not None else language_model.embed_tokens(input_ids)
            rotary_position_ids, text_position_ids = _split_qwen3_5_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                past_key_values=past_key_values,
            )
            causal_mask = create_causal_mask(
                config=language_model.config,
                inputs_embeds=source_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
            mtp_kwargs = {
                key: value
                for key, value in model_kwargs.items()
                if key
                not in {
                    "cache_position",
                    "cu_seqlens",
                    "cu_seqlens_padded",
                    "max_seqlen",
                    "mm_token_type_ids",
                    "padding_mask",
                    "qkv_format",
                }
            }
            if input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=language_model.embed_tokens,
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None) or (hidden_states,),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        mtp = getattr(self, "mtp", None)
        if mtp is not None:
            buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
            with buffer_device:
                for sublayer in mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)
        cast_model_to_dtype(self, dtype)


ModelClass = Qwen3_5ForCausalLM
