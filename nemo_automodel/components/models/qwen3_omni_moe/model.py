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
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration as HFQwen3OmniMoeThinkerForConditionalGeneration,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

from nemo_automodel.components.models.qwen3_moe.model import Block
from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Qwen3OmniMoeThinkerTextModel(
    nn.Module
):  # corresponding to qwen3_moe/model.py Qwen3MoeModel, diff: use MRopeRotaryEmbedding instead of RotaryEmbedding
    """Qwen3OmniMoe Thinker Text Model with MRoPE and sparse MoE layers."""

    def __init__(self, config: Qwen3OmniMoeTextConfig, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.backend = backend
        self.config = config

        # Map HF Qwen3OmniMoe config -> our MoE wrapper
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Block(layer_id, config, self.moe_config, backend) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        # args for deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            seq_length = inputs_embeds.shape[1]
            position_ids = (
                torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
            )
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        padding_mask = attention_mask.bool().logical_not()
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device),
            past_key_values=None,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        half = position_embeddings[0].shape[-1] // 2
        freqs_cis = torch.cat((position_embeddings[0][..., :half], position_embeddings[1][..., :half]), dim=-1)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                x=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_pos_masks = visual_pos_masks[..., 0]
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            # Ensure rotary embedding uses correct device
            self.rotary_emb.device = buffer_device

        for layer in self.layers:
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class Qwen3OmniMoeThinkerForConditionalGeneration(HFQwen3OmniMoeThinkerForConditionalGeneration, MoEFSDPSyncMixin):
    """Qwen3OmniMoe Thinker for Conditional Generation with multimodal support."""

    @classmethod
    def from_config(
        cls,
        config: Qwen3OmniMoeThinkerConfig,
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
        config = Qwen3OmniMoeThinkerConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        base_config = config.thinker_config if hasattr(config, "thinker_config") else config
        backend = backend or BackendConfig()

        super().__init__(base_config)

        self.backend = backend

        text_config = base_config.text_config if hasattr(base_config, "text_config") else base_config
        self.model = Qwen3OmniMoeThinkerTextModel(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )

        self.vocab_size = text_config.vocab_size
        self.pad_token_id = base_config.pad_token_id if base_config.pad_token_id is not None else -1
        self.spatial_merge_size = (
            base_config.vision_config.spatial_merge_size if hasattr(base_config, "vision_config") else 2
        )
        self.rope_deltas = None
        self.num_experts = text_config.num_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0)

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3OmniMoeStateDictAdapter(
                text_config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(text_config.torch_dtype, torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        use_audio_in_video: bool | None = None,
        video_second_per_grid: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor | dict:
        """Forward pass with multimodal fusion.

        Args:
            input_ids: Input token IDs
            input_features: Audio input features
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values
            image_grid_thw: Image grid (temporal, height, width)
            video_grid_thw: Video grid (temporal, height, width)
            attention_mask: Attention mask
            feature_attention_mask: Feature attention mask for audio
            audio_feature_lengths: Audio feature lengths
            position_ids: Position IDs (3D for MRoPE)
            padding_mask: Padding mask
            inputs_embeds: Optional pre-computed input embeddings
            labels: Labels for loss computation
            output_router_logits: Whether to output router logits
            use_audio_in_video: Whether audio is in video
            video_second_per_grid: Seconds per grid for videos
            **attn_kwargs: Additional attention arguments

        Returns:
            Logits tensor or dict with loss/aux_loss if labels provided
        """
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        # 1. Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None

        # 2. Merge multimodal features
        # Audio
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        # Images
        if pixel_values is not None:
            image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask
            visual_embeds_multiscale = image_embeds_multiscale

        # Videos
        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale
                visual_pos_masks = video_mask
            else:
                # Merge image and video multiscale features
                visual_pos_masks = video_mask | image_mask
                visual_embeds_multiscale_joint = ()
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(visual_embeds_multiscale, video_embeds_multiscale):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
                visual_embeds_multiscale = visual_embeds_multiscale_joint

        # 3. Compute MRoPE position IDs
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas

        # 4. Forward through text model
        hidden = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            deepstack_visual_embeds=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
            **attn_kwargs,
        )

        logits = self.lm_head(hidden) if self.lm_head else hidden

        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)

        # 5. Optionally compute loss/aux outputs
        if labels is not None or output_router_logits:
            output: dict[str, Any] = {"logits": logits}

            aux_loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

                if output_router_logits:
                    aux_loss = torch.tensor(0.0, device=logits.device)
                    loss = loss + self.router_aux_loss_coef * aux_loss
                    output["aux_loss"] = aux_loss

                output["loss"] = loss
            elif output_router_logits:
                aux_loss = torch.tensor(0.0, device=logits.device)
                output["aux_loss"] = aux_loss

            return output

        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config

        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
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
            # Ensure rotary embedding uses correct device after dtype move
            self.model.rotary_emb.device = buffer_device


ModelClass = Qwen3OmniMoeThinkerForConditionalGeneration
