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

import logging
import types
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants for identifying text/language modules in multimodal models
TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder")
MULTIMODAL_SUFFIXES = (
    "vision_tower",
    "visual",
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
    "audio_projector",
)


def get_text_module(model: nn.Module) -> nn.Module:
    """Return the nested text/LLM module if present, else the model itself."""
    if model is None:
        return model
    for attr_name in TEXT_MODULE_ATTRS:
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            # Only descend into a real submodule; Mock-only attrs from tests
            # (which hasattr() accepts) are not nn.Module and should be skipped.
            if nested is not None and isinstance(nested, nn.Module):
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
        # causal_mask_mapping should be precomputed in data pipeline via default_collater
        # If not provided, model will fail - this enforces clean separation
        if causal_mask_mapping is None:
            # If causal_mask_mapping is missing, fall back to on-the-fly computation.
            # This is not recommended for compilation, as it introduces runtime overhead.
            # TODO(PP): In pipeline parallelism, causal_mask_mapping is passed as a kwarg
            # but it is a dict (not a tensor), so it cannot be chunked by the PP schedule.
            # Non-first stages receive causal_mask_mapping=None and hit this fallback,
            # recomputing the mask every microbatch. This is a performance issue but not
            # a correctness bug since each stage has the full config to recompute correctly.
            # Long-term fix: pass the mask through stage input/output or compute it once
            # per stage and cache it.
            logger.warning(
                "causal_mask_mapping not provided; computing it here. "
                "This is slow and not recommended for compilation. "
                "Precompute causal_mask_mapping in the data pipeline for best performance."
            )
            if not isinstance((causal_mask_mapping := attention_mask), dict):
                from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

                # Note: inputs_embeds is only used for shape and dtype, not values
                # We could use a dummy tensor here, but inputs_embeds is already available
                mask_kwargs = {
                    "config": self.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": None,  # Training-only: no KV cache
                    "position_ids": position_ids,
                }
                causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
                if hasattr(self, "has_sliding_layers") and self.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

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

        if hasattr(self, "lm_head") and self.lm_head is not None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return logits
        else:
            return hidden_states

    return pipeline_forward_causal_lm


def create_pipeline_forward_gemma4_text() -> Callable:
    """Pipeline-compatible forward for the Gemma4 text decoder backbone.

    Works for both HF Gemma4TextModel (dense path) and Gemma4MoETextModelBackend (MoE path).
    Handles:
    - Optional embed_tokens (None on non-first PP stages; hidden states arrive in input_ids slot)
    - Both full_attention and sliding_attention causal masks (Gemma4 uses mixed layer types)
    - Per-layer-type position embeddings: Gemma4RotaryEmbedding.forward(x, pos_ids, layer_type)
    """

    def pipeline_forward_gemma4_text(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds must be provided")
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                # Non-first PP stage: previous stage output arrives as a float tensor in input_ids
                if input_ids is not None and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    inputs_embeds = input_ids
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")

        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask.bool().logical_not()

        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # Per-layer-type rotary embeddings: Gemma4RotaryEmbedding takes (x, pos_ids, layer_type)
        position_embeddings_map: dict = {}
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            for lt in set(getattr(self.config, "layer_types", ["full_attention"])):
                try:
                    position_embeddings_map[lt] = self.rotary_emb(inputs_embeds, position_ids, lt)
                except TypeError:
                    position_embeddings_map[lt] = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        config_layer_types = getattr(self.config, "layer_types", None)
        if hasattr(self, "layers") and self.layers is not None:
            layer_iter = self.layers.values() if hasattr(self.layers, "values") else self.layers
            for decoder_layer in layer_iter:
                # Prefer config.layer_types[layer_idx] over decoder_layer attribute — the
                # attribute lookup defaults to "full_attention" and mis-assigns position
                # embeddings (wrong head_dim) to sliding-window layers.
                if config_layer_types is not None and hasattr(decoder_layer, "layer_idx"):
                    idx = decoder_layer.layer_idx
                    layer_type = config_layer_types[idx] if idx < len(config_layer_types) else "full_attention"
                else:
                    layer_type = getattr(decoder_layer, "attention_type", "full_attention")
                layer_attention_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention"))
                position_embeddings = position_embeddings_map.get(
                    layer_type, position_embeddings_map.get("full_attention")
                )
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    padding_mask=padding_mask,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if hasattr(self, "norm") and self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    return pipeline_forward_gemma4_text


def create_pipeline_forward_gemma4_vlm() -> Callable:
    """Pipeline-compatible forward for Gemma4ForConditionalGeneration (VLM top-level).

    Stage 0: embeds text tokens, merges image features from vision tower (if pixel_values
    provided or stored in _vlm_pixel_values_chunks), then calls the patched language model.
    Non-first stages: passes hidden states straight to the patched language model.
    Last stage: applies lm_head and final-logit softcapping.
    """

    def pipeline_forward_gemma4_vlm(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        lang_model = self.model.language_model
        embed_tokens = getattr(lang_model, "embed_tokens", None)
        is_first_stage = embed_tokens is not None

        # PP VLM: retrieve pixel_values from chunks stored by the training loop
        if pixel_values is None and is_first_stage and getattr(self, "_vlm_pixel_values_chunks", None) is not None:
            has_media_tokens = (
                input_ids is not None
                and hasattr(self.config, "image_token_id")
                and (input_ids == self.config.image_token_id).any()
            )
            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    image_grid_chunk = (
                        self._vlm_image_grid_hws_chunks[chunk_idx]
                        if getattr(self, "_vlm_image_grid_hws_chunks", None) is not None
                        else None
                    )
                    if image_grid_chunk is not None:
                        image_position_ids = image_grid_chunk
                    self._vlm_chunk_idx = chunk_idx + 1

        if is_first_stage:
            if inputs_embeds is None:
                inputs_embeds = embed_tokens(input_ids)

            vision_tower = getattr(self.model, "vision_tower", None)
            if vision_tower is not None and pixel_values is not None:
                image_features = self.model.get_image_features(
                    pixel_values, image_position_ids=image_position_ids, return_dict=True
                ).pooler_output
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

                if mm_token_type_ids is not None:
                    special_image_mask = mm_token_type_ids == 1
                elif input_ids is not None:
                    special_image_mask = input_ids == self.config.image_token_id
                else:
                    special_image_mask = torch.zeros(
                        inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
                    )
                image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        else:
            # Non-first stage: input_ids carries hidden states from the previous PP stage
            if inputs_embeds is None:
                if input_ids is not None and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    inputs_embeds = input_ids
                else:
                    raise ValueError("Expected float hidden states for non-first PP stage")

        if cache_position is None and inputs_embeds is not None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        hidden_states = lang_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            **kwargs,
        )
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.last_hidden_state

        if hasattr(self, "lm_head") and self.lm_head is not None:
            logits = self.lm_head(hidden_states)
            text_config = getattr(self.config, "text_config", self.config)
            final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
            if final_logit_softcapping is not None:
                logits = logits / final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * final_logit_softcapping
            return logits
        return hidden_states

    return pipeline_forward_gemma4_vlm


def create_pipeline_forward_mistral3_vlm() -> Callable:
    """Pipeline-compatible forward for Mistral3ForConditionalGeneration (VLM top-level).

    Stage 0: embeds text tokens, runs vision_tower + multi_modal_projector for
    image tokens, merges image features into inputs_embeds via
    ``get_placeholder_mask``/``masked_scatter``, then calls the patched language
    model. Non-first stages: passes hidden states straight through the patched
    language model. Last stage: applies lm_head.

    Mirrors the generic CausalLM PP forward but adds the Mistral3 vision path
    so ``pixel_values``/``image_sizes`` reach ``get_image_features`` on stage 0.
    Without this, the generic CausalLM path never touches vision_tower and
    image tokens are embedded as garbage text tokens.
    """

    def pipeline_forward_mistral3_vlm(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        vision_feature_layer=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inner = self.model
        lang_model = inner.language_model
        embed_tokens = getattr(lang_model, "embed_tokens", None)
        is_first_stage = embed_tokens is not None

        # PP VLM: under the training loop, pixel_values and image_sizes are
        # popped from the batch before schedule.step() and pre-chunked onto
        # stage0_model._vlm_pixel_values_chunks (and _vlm_image_grid_hws_chunks
        # for image_sizes). Retrieve the current microbatch's chunk here so
        # vision_tower actually receives its inputs. Mirrors the Gemma4 VLM
        # path at create_pipeline_forward_gemma4_vlm().
        if pixel_values is None and is_first_stage:
            chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            has_media_tokens = (
                input_ids is not None
                and hasattr(self.config, "image_token_id")
                and (input_ids == self.config.image_token_id).any()
            )
            if chunks is not None and has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(chunks):
                    pixel_values = chunks[chunk_idx]
                    grid_chunks = getattr(self, "_vlm_image_grid_hws_chunks", None)
                    if grid_chunks is not None and chunk_idx < len(grid_chunks):
                        image_sizes = grid_chunks[chunk_idx]
                    self._vlm_chunk_idx = chunk_idx + 1

        if is_first_stage:
            if inputs_embeds is None:
                inputs_embeds = embed_tokens(input_ids)

            vision_tower = getattr(inner, "vision_tower", None)
            if vision_tower is not None and pixel_values is not None:
                # HF's outer Mistral3ForConditionalGeneration.forward resolves
                # `vision_feature_layer` from config via @merge_with_config_defaults.
                # Our patched forward bypasses that decorator, so we must pull the
                # config default explicitly — otherwise None flows through to
                # `get_image_features` and the selected hidden state can differ.
                if vision_feature_layer is None:
                    vision_feature_layer = getattr(self.config, "vision_feature_layer", None)
                image_features = inner.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    image_sizes=image_sizes,
                    return_dict=True,
                ).pooler_output
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                special_image_mask = inner.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        else:
            # Non-first stage: input_ids carries hidden states from the previous PP stage.
            if inputs_embeds is None:
                if input_ids is not None and input_ids.dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                ):
                    inputs_embeds = input_ids
                else:
                    raise ValueError("Expected float hidden states for non-first PP stage")

        if cache_position is None and inputs_embeds is not None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        hidden_states = lang_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            **kwargs,
        )
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.last_hidden_state

        if hasattr(self, "lm_head") and self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states

    return pipeline_forward_mistral3_vlm


def _is_mistral3_vlm(model: torch.nn.Module) -> bool:
    """Return True for Mistral3ForConditionalGeneration (Pixtral + Ministral3)."""
    config = getattr(model, "config", None)
    return config is not None and getattr(config, "model_type", None) == "mistral3"


def _is_gemma4_vlm(model: torch.nn.Module) -> bool:
    """Return True only for Gemma4 VLM variants.

    ``model.model.language_model`` alone is not enough to identify Gemma4 —
    Kimi VL, Mistral4, Qwen3 VL MoE, Llava OneVision and others share that
    structure. Gate the Gemma4-specific PP forward on the HF ``model_type``
    so unrelated VLMs fall through to the generic CausalLM path instead of
    receiving Gemma4's sliding/full-attention and softcapping logic.
    """
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", None)
    if model_type == "gemma4":
        return True
    # VLM configs usually nest the text backbone under ``text_config``.
    text_config = getattr(config, "text_config", None)
    return getattr(text_config, "model_type", None) == "gemma4"


def model_keeps_self_forward(model: torch.nn.Module) -> bool:
    """Return True when *model* opts out of pipeline-aware forward patching.

    Used by the pipeline split call site to skip ``patch_hf_model_for_pp``
    entirely for models whose own ``forward`` is already PP-aware (typically
    because it pulls pixel_values out of ``self._vlm_pixel_values_chunks``
    set by the training loop). Currently set on Qwen3-VL-MoE, Qwen3.5-MoE,
    KimiVL, and Kimi-K2.5-VL.
    """
    return bool(getattr(type(model), "_pp_keep_self_forward", False))


def patch_hf_model_for_pp(model, patch_inner_model: bool = True, patch_causal_lm_model: bool = True) -> None:
    """Patch a HF model/module to produce pipeline-compatible forward.

    The caller is responsible for skipping this function when the model
    opts out via ``model_keeps_self_forward(model)``. This function itself
    only branches on the patch *flavor*:

    - Gemma4 VLM (``config.model_type == 'gemma4'`` with a nested text
      backbone at ``model.model.language_model``): patch the text backbone
      and VLM outer with Gemma4-specific VLM-aware forwards.
    - Mistral3 VLM: patch the text backbone with the generic inner forward
      and the outer with the Mistral3-specific VLM forward.
    - Other models with ``model.model`` (e.g., LlamaForCausalLM and other
      LLMs): patch inner and outer with the generic CausalLM forwards.
    - Else: patch the module itself with the generic inner forward.
    """
    inner_model = getattr(model, "model", None)
    text_backbone = getattr(inner_model, "language_model", None) if inner_model is not None else None

    if inner_model is not None and text_backbone is not None and _is_gemma4_vlm(model):
        # Gemma4 VLM: the text backbone needs sliding/full-attention RoPE
        # dispatch and the VLM outer needs final_logit_softcapping.
        if patch_inner_model:
            text_backbone.forward = types.MethodType(create_pipeline_forward_gemma4_text(), text_backbone)
        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_gemma4_vlm(), model)
    elif inner_model is not None and text_backbone is not None and _is_mistral3_vlm(model):
        # Mistral3 VLM (Pixtral + Ministral3 text): route pixel_values/image_sizes
        # through vision_tower on stage 0 and let the text backbone use the
        # generic inner PP forward.
        if patch_inner_model:
            text_backbone.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), text_backbone)
        if patch_causal_lm_model:
            model.forward = types.MethodType(create_pipeline_forward_mistral3_vlm(), model)
    elif inner_model is not None:
        if patch_inner_model:
            inner_model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), inner_model)
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


# VLM model_types whose vision routing is handled inside ``patch_hf_model_for_pp``
# (a dedicated ``pipeline_forward_*_vlm`` function reads ``_vlm_pixel_values_chunks``).
_PP_VLM_MODEL_TYPES_WITH_DEDICATED_FORWARD: tuple[str, ...] = ("gemma4", "mistral3")


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
                    "tie_word_embeddings=True is not supported for pipelining. Use separate input/output embeddings."
                )
        if getattr(config, "is_encoder_decoder", False):
            issues.append("Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism.")

        # VLM PP routing: vision_tower only runs on stage 0, and pixel_values
        # are passed through the training loop's _vlm_pixel_values_chunks
        # mechanism. The model class must either (a) be on the dedicated PP
        # forward list (Gemma4 / Mistral3) or (b) declare
        # _pp_keep_self_forward = True so its own forward is preserved.
        # Otherwise patch_hf_model_for_pp replaces forward with the generic
        # CausalLM path, which silently drops pixel_values and trains the
        # language model on placeholder text embeddings.
        if _is_vlm(model):
            mt_outer = getattr(config, "model_type", None)
            mt_inner = getattr(getattr(config, "text_config", None), "model_type", None)
            has_dedicated = (
                mt_outer in _PP_VLM_MODEL_TYPES_WITH_DEDICATED_FORWARD
                or mt_inner in _PP_VLM_MODEL_TYPES_WITH_DEDICATED_FORWARD
            )
            keeps_own = bool(getattr(type(model), "_pp_keep_self_forward", False))
            if not has_dedicated and not keeps_own:
                issues.append(
                    f"VLM model_type='{mt_outer}' is not on the pipeline-aware list "
                    f"({', '.join(_PP_VLM_MODEL_TYPES_WITH_DEDICATED_FORWARD)}) and the model class "
                    f"{type(model).__name__} does not declare ``_pp_keep_self_forward = True``. "
                    "Without one of these, patch_hf_model_for_pp will replace the model's forward "
                    "with the generic CausalLM forward, and pixel_values stored in "
                    "``_vlm_pixel_values_chunks`` will never reach the vision tower."
                )

    if issues:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(issues, 1):
            error_msg += f"{i}. {issue}\n"
        raise ValueError(error_msg)
