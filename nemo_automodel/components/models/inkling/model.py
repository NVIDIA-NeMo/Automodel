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

"""NeMo AutoModel wrapper for the Inkling multimodal MoE model.

The wrapper rebuilds the Inkling text decoder as a single FSDP/PP-friendly unit
(:class:`InklingTextModelBackend`) so it can be wrapped, pipelined, and sharded
as one module, while reusing the HuggingFace vision/audio towers, attention with
short convolutions, relative-position logits, norms, and embeddings verbatim.
Only the sparse feed-forward of each decoder layer is swapped for an
expert-parallel :class:`InklingMoE`. This keeps the non-MoE numerics
bit-identical to transformers while giving the routed experts grouped-GEMM
compute and expert-parallel (EP) sharding, and follows the proven
``qwen3_vl_moe`` structure for PP + FSDP2 + EP support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.inkling.configuration_inkling import InklingConfig
from transformers.models.inkling.modeling_inkling import (
    InklingCausalLMOutputWithPast,
    InklingDecoderLayer,
    InklingModelOutputWithPast,
    InklingRMSNorm,
)
from transformers.models.inkling.modeling_inkling import (
    InklingForConditionalGeneration as HFInklingForConditionalGeneration,
)
from transformers.models.inkling.modeling_inkling import InklingModel as HFInklingModel
from transformers.models.inkling.modeling_inkling import InklingMoE as HFInklingMoE

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .layers import InklingMoE, build_inkling_moe_config
from .state_dict_adapter import InklingStateDictAdapter

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class InklingTextModelBackend(nn.Module):
    """Inkling text decoder rebuilt as one FSDP/PP-friendly unit.

    Owns the token embedding, the pre-decoder embedding norm, the decoder layers
    (each sparse layer's HF ``InklingMoE`` swapped for the expert-parallel
    :class:`InklingMoE`), and the final norm. ``forward`` is a stage-aware port of
    HF ``InklingTextModel.forward``: it accepts pre-computed ``inputs_embeds``
    (pipeline stages after the first) and indexes ``config.layer_types`` by each
    layer's own ``layer_idx`` so a pipelined subset of layers still selects the
    correct attention type.
    """

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.moe_config = moe_config or build_inkling_moe_config(config, backend)

        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=model_dtype)
        self.embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [InklingDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Swap each sparse decoder layer's HF MoE for the expert-parallel InklingMoE.
        for layer in self.layers:
            if isinstance(layer.mlp, HFInklingMoE):
                layer.mlp = InklingMoE(config, backend, moe_config=self.moe_config)
        self.norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> InklingModelOutputWithPast:
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))

        # Build the causal / sliding-window / recurrent masks exactly like HF unless
        # the caller already prepared the mapping (e.g. `generate`).
        if not isinstance(mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        # Under PP the splitter rewrites the layer ModuleList into a ModuleDict subset
        # (keyed by original index), so iterate values to get the modules either way.
        layers = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for layer in layers:
            # Index config.layer_types by the layer's own idx: under PP the subset
            # still retains each layer's original global index.
            layer_idx = layer.self_attn.layer_idx
            attention_type = "full_attention" if self.config.layer_types[layer_idx] == "hybrid" else "sliding_attention"
            hidden_states = layer(
                hidden_states,
                attention_mask=mask_mapping[attention_type],
                conv_mask=mask_mapping["linear_attention"],
                past_key_values=None,
                **kwargs,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return InklingModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)

    def get_input_embeddings(self) -> nn.Module | None:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.embed_norm is not None:
                self.embed_norm.reset_parameters()
            if self.norm is not None:
                self.norm.reset_parameters()
        for layer in self.layers:
            if isinstance(layer.mlp, InklingMoE):
                layer.mlp.init_weights(buffer_device=buffer_device)


class InklingModel(HFInklingModel):
    """PP-aware Inkling base model delegating the text decoder to the backend."""

    @property
    def layers(self) -> nn.ModuleList:
        return self.language_model.layers

    @property
    def embed_tokens(self) -> nn.Module | None:
        return self.language_model.embed_tokens

    @property
    def norm(self) -> nn.Module | None:
        return self.language_model.norm

    def get_input_embeddings(self) -> nn.Module | None:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> InklingModelOutputWithPast:
        embed_tokens = self.get_input_embeddings()
        if inputs_embeds is None:
            if embed_tokens is not None:
                # Stage 0: embed + pre-decoder norm (matches HF InklingModel.forward).
                inputs_embeds = self.language_model.embed_norm(embed_tokens(input_ids))
            elif input_ids is not None and isinstance(input_ids, torch.Tensor) and input_ids.dtype in _FLOAT_DTYPES:
                # Later PP stage: hidden states arrive in the input_ids slot.
                inputs_embeds = input_ids
                input_ids = None
            else:
                raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        # Merge text and images (stage 0 only, where the vision tower lives).
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_features, self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and audio (optional).
        if audio_input_ids is not None:
            audio_features = self.get_audio_features(audio_input_ids, audio_input_ids_mask).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, audio_features, self.config.audio_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        return InklingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=None,
            image_hidden_states=image_features,
        )


class InklingForConditionalGeneration(HFCheckpointingMixin, HFInklingForConditionalGeneration, MoEFSDPSyncMixin):
    """Inkling VLM with expert-parallel MoE feed-forwards, PP + FSDP2 + EP ready."""

    # VLM under pipeline parallelism: keep this model's own forward so pixel_values
    # reach the vision tower (otherwise patch_hf_model_for_pp swaps in the generic
    # CausalLM forward and drops them). Mirrors qwen3_vl_moe.
    _pp_keep_self_forward: bool = True

    # Keep parameters in a single (model) dtype so FSDP2 sees a uniform dtype per
    # wrapped group. The short convolutions upcast their compute to fp32 internally
    # and the router scores/bias are computed in fp32 (gate_precision), so uniform
    # bf16 storage does not change those numerically-sensitive paths. Overriding
    # HF's fp32-pinned short-conv list is intentional.
    _keep_in_fp32_modules_strict = []

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        config = InklingConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> None:
        backend = backend or BackendConfig()

        # Propagate the requested top-level dtype to the nested sub-configs so the
        # HF towers and our MoE parameters are constructed in a consistent dtype.
        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_cfg in vars(config).values():
                if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                    sub_cfg.torch_dtype = top_dtype

        super().__init__(config)

        self.backend = backend
        # Router scoring is selection-sensitive; keep it in fp32 unless overridden.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        text_config = config.text_config
        self.moe_config = moe_config or build_inkling_moe_config(text_config, backend)

        # Swap the HF base model for the PP-aware subclass and rebuild the text
        # decoder as one FSDP/PP-friendly backend unit (mirrors qwen3_vl_moe).
        self.model.__class__ = InklingModel
        self.model.language_model = InklingTextModelBackend(text_config, backend, moe_config=self.moe_config)
        # Exposed on the inner model too so the parallelizer can discover it.
        self.model.moe_config = self.moe_config

        model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False, dtype=model_dtype
        )
        self.vocab_size = text_config.vocab_size

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = InklingStateDictAdapter(
                text_config,
                self.moe_config,
                self.backend,
                dtype=model_dtype,
            )

    def get_input_embeddings(self) -> nn.Module | None:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module | None:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ):
        text_config = self.config.text_config

        # PP VLM support: retrieve pixel_values from stored per-microbatch chunks
        # when they were not passed directly for this stage's call.
        pixel_values = kwargs.get("pixel_values", None)
        has_image_tokens = (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.dtype not in _FLOAT_DTYPES
            and (input_ids == self.config.image_token_id).any()
        )
        if pixel_values is None and has_image_tokens:
            image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            if image_chunks is not None and chunk_idx < len(image_chunks):
                kwargs["pixel_values"] = image_chunks[chunk_idx]
                self._vlm_chunk_idx = chunk_idx + 1

        # Under pipeline parallelism, attention_mask (from batch kwargs) can have a
        # different sequence length than inputs_embeds (hidden states from the prev
        # stage). Drop the mismatched mask to avoid size errors in token routing.
        if inputs_embeds is not None and attention_mask is not None:
            if attention_mask.shape[-1] != inputs_embeds.shape[1]:
                attention_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Pipeline-parallel intermediate stage (no lm_head): pass hidden states
        # through unchanged so the next stage receives the raw tensor it expects.
        if self.lm_head is None:
            return hidden_states

        # Logits path must match HF InklingForConditionalGeneration.forward exactly:
        # divide by the muP width multiplier, then slice to the unpadded vocab size.
        hidden_states = hidden_states / text_config.logits_mup_width_multiplier
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        unpadded_vocab_size = text_config.unpadded_vocab_size
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            # ``.contiguous()`` so the last pipeline stage's output matches the meta
            # tensor's stride (the slice alone leaves a padded-vocab storage stride).
            logits = logits[..., :unpadded_vocab_size].contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs)

        return InklingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            image_hidden_states=getattr(outputs, "image_hidden_states", None),
        )

    def update_moe_gate_bias(self) -> None:
        """Inkling uses a trained correction bias, so gate-bias updates are a no-op."""
        return

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep the pre-decoder ``embed_norm`` on the first pipeline stage.

        ``embed_norm`` normalizes token embeddings before the vision merge, so it must
        live on stage 0 next to ``embed_tokens``. The generic split spec only keeps
        ``embed_tokens``/``norm``/``lm_head``, so add ``embed_norm`` explicitly.
        """
        if module_names_per_stage:
            embed_norm_fqn = f"{layers_prefix}embed_norm"
            if embed_norm_fqn not in module_names_per_stage[0]:
                module_names_per_stage[0].append(embed_norm_fqn)
        return module_names_per_stage


ModelClass = InklingForConditionalGeneration
