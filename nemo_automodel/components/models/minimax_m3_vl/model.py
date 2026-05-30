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

"""MiniMax M3 (mixed sparse/dense MoE) text backbone.

Stage 1 implements ``MiniMaxM3TextModel`` and the standalone
``MiniMaxM3SparseForCausalLM`` so the language path can be parity-tested against
the sglang reference before the vision tower / VLM wrapper (Stage 3) embeds the
text model as ``language_model``.
"""

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import Block, MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import (
    MiniMaxM3StateDictAdapter,
    MiniMaxM3VLStateDictAdapter,
)
from nemo_automodel.components.models.minimax_m3_vl.vision_encoder import MiniMaxM3VisionModel
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def build_moe_config(config: Any, dtype: torch.dtype) -> MoEConfig:
    """Build the routed-expert ``MoEConfig`` for the M3 backbone.

    Shared experts are handled in :class:`~...layers.Block` (SwiGLU-OAI), so
    ``n_shared_experts`` is 0 here. Routed experts use the ``swigluoai``
    activation ``gate * sigmoid(alpha * gate) * (up + 1)`` over the concatenated
    grouped gate/up projection produced by ``MoESplitExpertsStateDictMixin``.
    """
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.intermediate_size,
        n_routed_experts=config.num_local_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sigmoid" if str(getattr(config, "scoring_func", "sigmoid")).lower() != "softmax" else "softmax",
        route_scale=float(getattr(config, "routed_scaling_factor", 1.0)),
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swigluoai",
        activation_alpha=float(getattr(config, "swiglu_alpha", 1.702)),
        activation_limit=float(getattr(config, "swiglu_limit", 7.0)),
        softmax_before_topk=False,
        force_e_score_correction_bias=bool(getattr(config, "use_routing_bias", True)),
        dtype=dtype,
    )


class MiniMaxM3TextModel(nn.Module):
    """Embedding + decoder stack + final norm for the M3 text backbone."""

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        self.config.num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", None))

        dtype = get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16)
        self.moe_config = moe_config or build_moe_config(config, dtype)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)

        gemma = getattr(config, "use_gemma_norm", False)
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
            rotary_dim = getattr(config, "rotary_dim", self.head_dim)
            config.rope_parameters = {
                "rope_theta": getattr(config, "rope_theta", 5000000.0),
                "rope_type": "default",
                "partial_rotary_factor": rotary_dim / self.head_dim,
            }

        base, rope_scaling, partial_rotary_factor = get_rope_config(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(0, h.shape[1], device=h.device).unsqueeze(0).expand(h.shape[0], -1)

        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )

        return self.norm(h)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight)
            self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)


class MiniMaxM3SparseForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Standalone M3 text backbone for causal LM (Stage 1 parity target)."""

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLTextConfig.from_pretrained(pretrained_model_name_or_path)
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
        self.model = MiniMaxM3TextModel(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3StateDictAdapter(
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
        if attn_kwargs.get("qkv_format") == "thd":
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
        if attn_kwargs.get("qkv_format") == "thd":
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
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
                )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


class MiniMaxM3SparseForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """MiniMax M3 VL: CLIP-style vision tower + projector/merger + M3 text backbone.

    Vision features (``vision_tower(pixel_values, grid_thw)``) are spliced into
    the text embeddings at ``image_token_index`` / ``video_token_index``
    positions, then run through the (sparse/dense MoE) language model + lm_head.
    """

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLConfig.from_pretrained(pretrained_model_name_or_path)
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
        text_config = config.text_config
        self.backend = backend or BackendConfig()
        self.model = MiniMaxM3TextModel(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.vision_tower = MiniMaxM3VisionModel(
            config.vision_config,
            text_config.hidden_size,
            config.projector_hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
        )
        self.image_token_index = config.image_token_index
        self.video_token_index = config.video_token_index
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3VLStateDictAdapter(
                config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(text_config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    @property
    def language_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    @staticmethod
    def _to_grid_list(grid_thw) -> list[list[int]]:
        if isinstance(grid_thw, torch.Tensor):
            return grid_thw.detach().cpu().to(torch.int64).tolist()
        return [list(map(int, g)) for g in grid_thw]

    def _splice_multimodal(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw,
        token_index: int,
    ) -> torch.Tensor:
        features = self.vision_tower(pixel_values, self._to_grid_list(grid_thw))
        mask = input_ids == token_index
        expected = int(mask.sum().item())
        if features.shape[0] != expected:
            raise ValueError(
                f"MiniMax M3 VL: got {features.shape[0]} vision tokens for {expected} placeholder positions "
                f"(token_index={token_index})."
            )
        inputs_embeds[mask] = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw=None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None or pixel_values_videos is not None:
                inputs_embeds = inputs_embeds.clone()
            if pixel_values is not None:
                inputs_embeds = self._splice_multimodal(
                    inputs_embeds, input_ids, pixel_values, image_grid_thw, self.image_token_index
                )
            if pixel_values_videos is not None:
                inputs_embeds = self._splice_multimodal(
                    inputs_embeds, input_ids, pixel_values_videos, video_grid_thw, self.video_token_index
                )

        hidden = self.model(
            None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return self.lm_head(hidden)

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.text_config.hidden_size**-0.5
            nn.init.trunc_normal_(
                self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
            )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = MiniMaxM3SparseForConditionalGeneration
