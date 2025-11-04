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
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    _get_feat_extract_output_lengths
)

from nemo_automodel.components.models.qwen3_moe.model import Block
from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Qwen3OmniMoeThinkerTextModel(nn.Module):  #corresponding to qwen3_moe/model.py Qwen3MoeModel, diff: use MRopeRotaryEmbedding instead of RotaryEmbedding
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

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
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
            position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

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


class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module, MoEFSDPSyncMixin):
    """Qwen3OmniMoe Thinker for Conditional Generation with multimodal support.
    
    This model supports text, audio, image, and video inputs. The audio and vision encoders
    are automatically instantiated from HuggingFace implementations if the config includes
    audio_config and vision_config.
    
    Architecture:
        - Audio encoder: HF Qwen3OmniMoeAudioEncoder (if audio_config present)
        - Vision encoder: HF Qwen3OmniMoeVisionEncoder (if vision_config present)
        - Text model: Qwen3OmniMoeThinkerTextModel (AutoModel with MoE optimization)
        - LM head: Linear layer for token prediction
    """

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
        super().__init__()
        self.config = config.thinker_config if hasattr(config, 'thinker_config') else config
        self.backend = backend or BackendConfig()
        
        # Use the text_config from the thinker config
        text_config = self.config.text_config if hasattr(self.config, 'text_config') else self.config
        
        self.audio_tower = Qwen3OmniMoeAudioEncoder(self.config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder(self.config.vision_config)

        # Core text model - uses automodel backend for MoE optimization
        self.model = Qwen3OmniMoeThinkerTextModel(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        
        # Multimodal config
        self.vocab_size = text_config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size if hasattr(config, 'vision_config') else 2
        self.rope_deltas = None
        self.num_experts = text_config.num_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0)
        
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3OmniMoeStateDictAdapter(
                text_config, self.model.moe_config, self.backend, dtype=get_dtype(text_config.torch_dtype, torch.bfloat16)
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None
    ):
        """Encodes videos into continuous embeddings.
        
        Args:
            pixel_values_videos: Video pixel values
            video_grid_thw: Temporal, height, width grid for each video
            
        Returns:
            Tuple of (video_embeds, video_embeds_multiscale) for DeepStack
        """
        if self.visual is None:
            raise RuntimeError(
                "Vision encoder not available. Model was initialized without vision_config."
            )
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype) 
        return self.visual(pixel_values_videos, grid_thw=video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
        """Encodes images into continuous embeddings.
        
        Args:
            pixel_values: Image pixel values
            image_grid_thw: Temporal, height, width grid for each image
            
        Returns:
            Tuple of (image_embeds, image_embeds_multiscale) for DeepStack
        """
        if self.visual is None:
            raise RuntimeError(
                "Vision encoder not available. Model was initialized without vision_config."
            )
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
    ):
        """Encodes audios into continuous embeddings.
        
        Args:
            input_features: Audio input features
            feature_attention_mask: Attention mask for features
            audio_feature_lengths: Length of each audio feature
            
        Returns:
            Audio embeddings
        """
        if self.audio_tower is None:
            raise RuntimeError(
                "Audio encoder not available. Model was initialized without audio_config."
            )
        
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        return audio_outputs.last_hidden_state

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """Get masks for placeholder tokens (image, video, audio).
        
        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings
            image_features: Image feature embeddings (for validation)
            video_features: Video feature embeddings (for validation)
            
        Returns:
            Tuple of (image_mask, video_mask, audio_mask)
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    # Copied from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision
    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    # Copied from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_chunked_index
    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                                token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
            remove_index (`int`) An index id to subtract from `token_indices` before chunking

        Returns:
            `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    # Copied from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index
    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: torch.LongTensor | None = None,
        second_per_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    # Audio in Video
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(audio_llm_pos_ids)

                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st += int(text_len + bos_len + audio_len + video_len + eos_len)

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas

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
        text_config = self.config.text_config if hasattr(self.config, 'text_config') else self.config
        
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