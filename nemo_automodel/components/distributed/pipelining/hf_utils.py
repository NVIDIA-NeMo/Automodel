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

"""Generic pipeline-parallel adaptation of un-migrated HuggingFace models.

Nothing here may encode model-specific architecture policy: that belongs to the
owning model package under ``nemo_automodel/components/models/<model>/``. A model
that needs its own pipeline behavior inherits
``nemo_automodel.shared.pipeline.PipelineModelMixin`` and declares
``pipeline_forward_style = PipelineForwardStyle.MODEL``; the pipeline builder then
leaves its forward alone.

``TEXT_MODULE_ATTRS`` and ``MULTIMODAL_SUFFIXES`` are a name-sniffing
compatibility heuristic for models that have not been migrated to that contract.
They guess where a HuggingFace model keeps its text decoder and its vision/audio
towers from attribute names alone, so every new naming convention has to be
appended to them reactively. Migrated models must not rely on them: they express
stage ownership through ``PipelineModelMixin.pipeline_stage_modules`` instead.
"""

import logging
import types
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn

from nemo_automodel.shared.pipeline import PipelineForwardStyle, PipelineModelMixin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants for identifying text/language modules in multimodal models
TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder")
MULTIMODAL_SUFFIXES = (
    "vision_tower",
    "visual",
    "vision_model",
    "image_encoder",
    "vision_encoder",
    "embed_vision",
    "audio_tower",
    "audio_encoder",
    "audio_model",
    "mm_projector",
    "multi_modal_projector",
    "multimodal_projector",
    "vision_projector",
    "vit_large_projector",
    "audio_projector",
)


def get_text_module(model: nn.Module) -> nn.Module:
    """Return the nested text/LLM module if present, else the model itself."""
    if model is None:
        return model
    for attr_name in TEXT_MODULE_ATTRS:
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            # The text module must be a real submodule: the caller partitions it
            # into pipeline stages by module FQN and calls it as a stage. An
            # attribute that merely shares the name but is not an nn.Module
            # cannot own layers, so skip it and keep searching.
            if nested is not None and isinstance(nested, nn.Module):
                return nested
    return model


def _build_or_reuse_pp_causal_mask(module, inputs_embeds, attention_mask, cache_position, position_ids):
    """Build a stage's ``causal_mask_mapping``, caching it per stage when safe.

    Under pipeline parallelism the mask precomputed in the data pipeline only reaches
    the first stage; non-first stages arrive with ``causal_mask_mapping=None`` and used
    to recompute it on every microbatch (slow, and a torch.compile graph-break). When
    no explicit ``attention_mask`` is provided -- the common fixed-length / packed
    training case, and exactly what non-first stages receive -- the causal mask depends
    only on ``(seq_len, dtype, device)`` and is constant across microbatches and steps,
    so it is built once per stage and reused. With an explicit ``attention_mask`` (which
    may encode per-batch padding) it is rebuilt each call. Behavior is identical to the
    previous recompute; only the redundant recomputation is removed.
    """
    # An ``attention_mask`` that is already a mask-mapping dict is used as-is.
    if isinstance(attention_mask, dict):
        return attention_mask

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cacheable = attention_mask is None
    cache_key = (inputs_embeds.shape[1], inputs_embeds.dtype, inputs_embeds.device)
    cache = getattr(module, "_pp_causal_mask_cache", None)
    if cache is not None and not isinstance(cache, MutableMapping):
        cache = None
    if cacheable and cache is not None and cache_key in cache:
        return cache[cache_key]

    # Note: inputs_embeds is only used for shape and dtype, not values.
    mask_kwargs = {
        "config": module.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,  # Training-only: no KV cache
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(module, "has_sliding_layers", False) is True:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    if cacheable:
        if cache is None:
            cache = {}
            module._pp_causal_mask_cache = cache
        cache[cache_key] = causal_mask_mapping
    return causal_mask_mapping


def create_pipeline_forward_inner(model_class_name: str = "AutoModel") -> Callable:
    """Create a pipeline-compatible forward method for HuggingFace inner models."""
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
        # For VLM models the text components (embed_tokens, layers, norm) live on a
        # nested text module (e.g. model.language_model) rather than directly on self.
        # get_text_module returns self when no nesting exists (e.g. LlamaModel).
        text_module = get_text_module(self)

        # Embeddings handling
        if inputs_embeds is None:
            if hasattr(text_module, "embed_tokens") and text_module.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = text_module.embed_tokens(input_ids)
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

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Attention mask handling (compilation-friendly):
        # causal_mask_mapping is precomputed in the data pipeline (default_collater +
        # add_causal_masks_to_batch) and passed to the first stage. The PP schedule
        # cannot forward this dict to non-first stages, which therefore arrive with
        # causal_mask_mapping=None. Build it once per stage and cache it (see
        # _build_or_reuse_pp_causal_mask) instead of recomputing every microbatch.
        if causal_mask_mapping is None:
            causal_mask_mapping = _build_or_reuse_pp_causal_mask(
                self, inputs_embeds, attention_mask, cache_position, position_ids
            )

        hidden_states = inputs_embeds

        # Rotary embeddings precomputation (shared across layers)
        position_embeddings = None
        rotary_emb = get_text_module(self).rotary_emb
        if rotary_emb is not None:
            position_embeddings = rotary_emb(hidden_states, position_ids)

        if hasattr(text_module, "layers") and text_module.layers is not None:
            # Works for dict-like or list-like containers
            layer_iter = text_module.layers.values() if hasattr(text_module.layers, "values") else text_module.layers
            for decoder_layer in layer_iter:
                layer_attention_mask = causal_mask_mapping.get("full_attention")
                if hasattr(decoder_layer, "attention_type"):
                    layer_attention_mask = causal_mask_mapping.get(
                        getattr(decoder_layer, "attention_type"), causal_mask_mapping.get("full_attention")
                    )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if hasattr(text_module, "norm") and text_module.norm is not None:
            hidden_states = text_module.norm(hidden_states)

        if model_class_name == "PipelineStage":
            return hidden_states
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

    return pipeline_forward


def create_pipeline_forward_causal_lm() -> Callable:
    """Create a pipeline-compatible forward method for causal LM wrappers."""
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
        """Pipeline-stage forward for a causal-LM wrapper.

        B=microbatch, S=seq, H=hidden, V=vocab. Non-first stages take input
        hidden states ``[B, S, H]`` via ``inputs_embeds`` (or ``input_ids`` when
        already floating-point).

        Returns hidden states ``[B, S, H]`` when ``self._pp_return_hidden_states``
        is set (lm_head deferred to FusedLinearCrossEntropy); else logits
        ``[B, S', V]`` when this stage owns ``lm_head`` (``S'`` = ``S`` sliced by
        ``logits_to_keep``); else hidden states ``[B, S, H]`` for the next stage.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if hasattr(self, "model") and self.model is not None:
            outputs = self.model(
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
        else:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            elif input_ids is not None and input_ids.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                hidden_states = input_ids
            else:
                raise ValueError("Expected hidden states as input for pipeline stage without inner model")
            outputs = None

        if getattr(self, "_pp_return_hidden_states", False) is True:
            return hidden_states

        if hasattr(self, "lm_head") and self.lm_head is not None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return logits
        else:
            return hidden_states

    return pipeline_forward_causal_lm


def model_keeps_self_forward(model: torch.nn.Module) -> bool:
    """Return whether a model contract owns its pipeline-aware forward.

    Models using the generic HuggingFace pipeline path do not inherit
    ``PipelineModelMixin`` and are patched by the shared implementation.
    """
    return isinstance(model, PipelineModelMixin) and model.pipeline_forward_style is PipelineForwardStyle.MODEL


def patch_hf_model_for_pp(model, patch_inner_model: bool = True, patch_causal_lm_model: bool = True) -> None:
    """Patch a HF model/module to produce pipeline-compatible forward.

    The caller is responsible for skipping this function when the model opts out
    via ``model_keeps_self_forward(model)``. This function itself only branches on
    the module layout:

    - A module with an inner ``model`` (e.g. ``LlamaForCausalLM``): patch the
      inner model with the generic backbone forward and the outer module with the
      generic causal-LM forward.
    - A module without one (a bare backbone): patch the module itself with the
      generic backbone forward.

    Args:
        model: Model part to patch in place.
        patch_inner_model: Whether to patch the backbone forward.
        patch_causal_lm_model: Whether to patch the causal-LM wrapper forward.
    """
    inner_model = getattr(model, "model", None)

    if inner_model is not None:
        if patch_inner_model:
            inner_model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), inner_model)
        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_causal_lm(), model)
            # The generic causal-LM forward honors ``_pp_return_hidden_states``,
            # so this part can defer its vocabulary projection to a fused loss.
            model.pipeline_supports_hidden_state_output = True
    else:
        if patch_inner_model:
            model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model)


def _is_vlm(model: torch.nn.Module) -> bool:
    """Best-effort check for whether ``model`` is a vision-language model.

    Looks at the standard VLM markers used elsewhere in the codebase: a nested
    ``text_config``, a ``vision_tower`` attribute on the outer model, or a
    ``visual`` attribute on the inner model (Qwen-VL convention).
    """
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "text_config", None) is not None:
        return True
    if hasattr(model, "vision_tower"):
        return True
    inner = getattr(model, "model", None)
    return inner is not None and (hasattr(inner, "vision_tower") or hasattr(inner, "visual"))


def validate_hf_model_for_pipeline_support(model: torch.nn.Module) -> None:
    """Validate if a model is compatible with torch.distributed.pipelining."""
    model_name = getattr(getattr(model, "config", object()), "pretrained_model_name_or_path", "Unknown")
    config = getattr(model, "config", None)

    issues: list[str] = []

    if config is not None:
        # For VLMs, check text_config (the outer VLM config tie flag is irrelevant for PP)
        check_config = getattr(config, "text_config", config)
        if getattr(check_config, "tie_word_embeddings", False):
            # Only a real problem if lm_head and embed_tokens share the same weight tensor
            lm_head = getattr(model, "lm_head", None)
            inner = getattr(model, "model", model)
            embed_tokens = getattr(inner, "embed_tokens", None)
            if embed_tokens is None:
                lang = getattr(inner, "language_model", None)
                if lang is not None:
                    embed_tokens = getattr(lang, "embed_tokens", None)
            weights_tied = (
                lm_head is not None
                and embed_tokens is not None
                and hasattr(lm_head, "weight")
                and hasattr(embed_tokens, "weight")
                and lm_head.weight is embed_tokens.weight
            )
            if weights_tied:
                issues.append(
                    "Pipeline parallelism does not support tie_word_embeddings=True, and overriding "
                    "it to tie_word_embeddings=False is not supported either. Train this model with "
                    "another supported parallelism strategy (e.g., FSDP2) instead."
                )
        if getattr(config, "is_encoder_decoder", False):
            issues.append("Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism.")

        # VLM PP routing: vision_tower only runs on stage 0, and the media
        # tensors are staged per microbatch on the stage-0 module rather than
        # carried in the batch. Only the model class knows how to consume them,
        # so it must own its pipeline forward. Otherwise patch_hf_model_for_pp
        # installs the generic causal-LM forward, which silently drops
        # pixel_values and trains the language model on placeholder text
        # embeddings.
        if _is_vlm(model) and not model_keeps_self_forward(model):
            issues.append(
                f"VLM {type(model).__name__} does not own its pipeline forward. Make the class inherit "
                "nemo_automodel.shared.pipeline.PipelineModelMixin, set "
                "pipeline_forward_style = PipelineForwardStyle.MODEL, and handle the pipeline stage "
                "layout in its own forward (embed + vision merge on stage 0, incoming hidden states on "
                "later stages, lm_head on the last stage). Without that, patch_hf_model_for_pp replaces "
                "the model's forward with the generic CausalLM forward and the staged media never "
                "reaches the vision tower."
            )

    if issues:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(issues, 1):
            error_msg += f"{i}. {issue}\n"
        raise ValueError(error_msg)
