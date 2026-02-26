# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import logging
import types
from typing import TYPE_CHECKING, Callable, Optional, Union
from unittest.mock import Mock as UnitTestMock

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants for identifying text/language modules in multimodal models
TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder", "backbone")
MULTIMODAL_SUFFIXES = (
    "vision_tower",
    "visual",
    "image_encoder",
    "vision_encoder",
    "audio_tower",
    "audio_encoder",
    "audio_model",
    "mm_projector",
    "multi_modal_projector",
    "multimodal_projector",
    "vision_projector",
    "audio_projector",
)


def _safe_getattr(obj, name: str, default=None):
    """Get an attribute while avoiding synthetic attributes from unittest.mock.Mock."""
    if isinstance(obj, UnitTestMock):
        return obj.__dict__.get(name, default)
    return getattr(obj, name, default)


def get_text_module(model: nn.Module) -> nn.Module:
    """Return the nested text/LLM module if present, else the model itself."""
    if model is None:
        return model
    for attr_name in TEXT_MODULE_ATTRS:
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            if nested is not None:
                return nested
    return model


def create_pipeline_forward_inner(model_class_name: str = "AutoModel") -> Callable:
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        causal_mask_mapping: Optional[dict] = None,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutputWithPast]:
        layers = _safe_getattr(self, "layers", None)
        first_layer = None
        if layers is not None:
            if hasattr(layers, "values"):
                layer_values = list(layers.values())
                first_layer = layer_values[0] if len(layer_values) > 0 else None
            else:
                first_layer = layers[0] if len(layers) > 0 else None
        is_nemotron_h_like = first_layer is not None and hasattr(first_layer, "block_type")

        # Embeddings handling
        if inputs_embeds is None:
            embed_tokens = _safe_getattr(self, "embed_tokens", None)
            embeddings = _safe_getattr(self, "embeddings", None)
            if embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = embed_tokens(input_ids)
            elif embeddings is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = embeddings(input_ids)
            else:
                if (
                    input_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    inputs_embeds = input_ids
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache()

        # Stage-level hints (set by pipeline splitter) let us skip unused mask/position work.
        pp_needs_attention_mask = bool(_safe_getattr(self, "_nemo_pp_needs_attention_mask", True))
        pp_needs_cache_position = bool(_safe_getattr(self, "_nemo_pp_needs_cache_position", True))

        if cache_position is None and (pp_needs_cache_position or use_cache):
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        if is_nemotron_h_like:
            # NemotronH layer blocks require cache_params and per-block masking logic.
            causal_mask = None
            mamba_mask = None
            if pp_needs_attention_mask:
                update_causal_mask = _safe_getattr(self, "_update_causal_mask", None)
                update_mamba_mask = _safe_getattr(self, "_update_mamba_mask", None)
                causal_mask = (
                    update_causal_mask(attention_mask, inputs_embeds, cache_position) if update_causal_mask else None
                )
                mamba_mask = (
                    update_mamba_mask(attention_mask, cache_position) if update_mamba_mask else attention_mask
                )
            layer_iter = layers.values() if hasattr(layers, "values") else layers
            for mixer_block in layer_iter:
                if mixer_block.block_type == "mamba":
                    layer_mask = mamba_mask if pp_needs_attention_mask else None
                elif mixer_block.block_type == "attention":
                    layer_mask = causal_mask if pp_needs_attention_mask else None
                else:
                    layer_mask = None
                # Some NemotronH-like blocks (e.g., local NemotronV3Block) do not accept
                # cache kwargs, while HF NemotronH blocks do. Use signature-aware dispatch.
                signature_owner = getattr(mixer_block, "_checkpoint_wrapped_module", mixer_block)
                supports_cache_params = getattr(signature_owner, "_nemo_pp_supports_cache_params", None)
                supports_cache_position = getattr(signature_owner, "_nemo_pp_supports_cache_position", None)
                if supports_cache_params is None or supports_cache_position is None:
                    try:
                        forward_params = inspect.signature(signature_owner.forward).parameters
                        supports_cache_params = "cache_params" in forward_params
                        supports_cache_position = "cache_position" in forward_params
                    except (TypeError, ValueError):
                        supports_cache_params = True
                        supports_cache_position = True
                    setattr(signature_owner, "_nemo_pp_supports_cache_params", supports_cache_params)
                    setattr(signature_owner, "_nemo_pp_supports_cache_position", supports_cache_position)

                block_kwargs = {"attention_mask": layer_mask}
                if supports_cache_params:
                    block_kwargs["cache_params"] = past_key_values
                if supports_cache_position:
                    block_kwargs["cache_position"] = cache_position if pp_needs_cache_position else None

                hidden_states = mixer_block(hidden_states, **block_kwargs)
        else:
            # Attention mask handling (compilation-friendly):
            # causal_mask_mapping should be precomputed in data pipeline via default_collater
            # If not provided, model will fail - this enforces clean separation
            if causal_mask_mapping is None:
                # If causal_mask_mapping is missing, fall back to on-the-fly computation.
                # This is not recommended for compilation, as it introduces runtime overhead.
                logger.warning(
                    "causal_mask_mapping not provided; computing it here. "
                    "This is slow and not recommended for compilation. "
                    "Precompute causal_mask_mapping in the data pipeline for best performance."
                )
                if not isinstance((causal_mask_mapping := attention_mask), dict):
                    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

                    # Note: input_embeds is only used for shape and dtype, not values
                    # We could use a dummy tensor here, but inputs_embeds is already available
                    mask_kwargs = {
                        "config": self.config,
                        "input_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        "cache_position": cache_position,
                        "past_key_values": None,  # Training-only: no KV cache
                        "position_ids": position_ids,
                    }
                    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
                    if hasattr(self, "has_sliding_layers") and self.has_sliding_layers:
                        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            # Rotary embeddings precomputation (shared across layers)
            position_embeddings = None
            rotary_emb = _safe_getattr(get_text_module(self), "rotary_emb", None)
            if rotary_emb is not None:
                position_embeddings = rotary_emb(hidden_states, position_ids)

            if layers is not None:
                # Works for dict-like or list-like containers
                layer_iter = layers.values() if hasattr(layers, "values") else layers
                for decoder_layer in layer_iter:
                    layer_attention_mask = causal_mask_mapping.get("full_attention")
                    if hasattr(decoder_layer, "attention_type"):
                        layer_attention_mask = causal_mask_mapping.get(
                            getattr(decoder_layer, "attention_type"), causal_mask_mapping.get("full_attention")
                        )

                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

        norm_f = _safe_getattr(self, "norm_f", None)
        norm = _safe_getattr(self, "norm", None)
        if norm_f is not None:
            hidden_states = norm_f(hidden_states)
        elif norm is not None:
            hidden_states = norm(hidden_states)

        if model_class_name == "PipelineStage":
            return hidden_states
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

    return pipeline_forward


def create_pipeline_forward_causal_lm() -> Callable:
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward_causal_lm(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutputWithPast]:
        inner_model = _safe_getattr(self, "model", None)
        backbone = _safe_getattr(self, "backbone", None)
        lm_head = _safe_getattr(self, "lm_head", None)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inner_model is not None:
            outputs = inner_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )
            if isinstance(outputs, BaseModelOutputWithPast):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs
                outputs = None
        elif backbone is not None:
            outputs = backbone(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_params=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                attention_mask=attention_mask,
                **kwargs,
            )
            if isinstance(outputs, BaseModelOutputWithPast):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            elif hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs
        else:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            elif input_ids is not None and input_ids.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                hidden_states = input_ids
            else:
                raise ValueError("Expected hidden states as input for pipeline stage without inner model")
            outputs = None

        if lm_head is not None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = lm_head(hidden_states[:, slice_indices, :])
            return logits
        else:
            return hidden_states

    return pipeline_forward_causal_lm


def patch_hf_model_for_pp(model, patch_inner_model: bool = True, patch_causal_lm_model: bool = True) -> None:
    """Patch a HF model/module to produce pipeline-compatible forward.

    - If model has .model (e.g., LlamaForCausalLM), patch inner and outer.
    - Else, patch the module itself.
    """
    if hasattr(model, "model"):
        if patch_inner_model and getattr(model, "model", None) is not None:
            model.model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model.model)

        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_causal_lm(), model)
    elif hasattr(model, "backbone"):
        if patch_inner_model and getattr(model, "backbone", None) is not None:
            model.backbone.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model.backbone)

        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_causal_lm(), model)
    else:
        if patch_inner_model:
            model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model)


def init_hf_model_buffers(model: torch.nn.Module, device: torch.device) -> None:
    if hasattr(getattr(model, "model", model), "rotary_emb"):
        rotary_owner = getattr(model, "model", model)
        if hasattr(rotary_owner.rotary_emb, "rope_init_fn"):
            inv_freq, _ = rotary_owner.rotary_emb.rope_init_fn(rotary_owner.rotary_emb.config, device)
            rotary_owner.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


def validate_hf_model_for_pipeline_support(model: torch.nn.Module) -> None:
    """Validate if a model is compatible with torch.distributed.pipelining."""
    model_name = getattr(getattr(model, "config", object()), "pretrained_model_name_or_path", "Unknown")
    config = getattr(model, "config", None)

    issues: list[str] = []

    if config is not None:
        if getattr(config, "tie_word_embeddings", False):
            issues.append(
                "tie_word_embeddings=True is not supported for pipelining. Use separate input/output embeddings."
            )
        if getattr(config, "is_encoder_decoder", False):
            issues.append("Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism.")

    if issues:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(issues, 1):
            error_msg += f"{i}. {issue}\n"
        raise ValueError(error_msg)
