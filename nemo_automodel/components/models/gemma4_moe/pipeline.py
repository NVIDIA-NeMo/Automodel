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

"""Pipeline-stage forward for the dense Gemma4 VL model.

A pipeline stage owns a slice of the model: stage 0 keeps ``embed_tokens`` and
the vision tower, the last stage keeps ``norm`` and ``lm_head``, and every stage
keeps a contiguous span of decoder layers. The stock HuggingFace forwards cannot
run on such a slice, so ``Gemma4ForConditionalGeneration`` routes pipeline stages
here instead (``PipelineForwardStyle.MODEL``).

Gemma4-specific policy lives here and nowhere else: mixed sliding/full attention
masks, per-layer-type rotary embeddings, the shared key/value store used by
kv-sharing layers, and final-logit softcapping.
"""

from collections.abc import MutableMapping

import torch
import torch.nn as nn

from nemo_automodel.shared.pipeline import pp_media_chunk


def gemma4_pipeline_text_forward(
    text_model: nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    shared_kv_states: MutableMapping,
) -> torch.Tensor:
    """Run the decoder layers this pipeline stage owns.

    Args:
        text_model: Gemma4 text backbone pruned to this stage's layers. Only the
            last stage still owns ``norm``.
        inputs_embeds: Hidden states of shape [batch, sequence, hidden] -- token
            embeddings on stage 0, the previous stage's output afterwards.
        attention_mask: Padding mask of shape [batch, sequence], or None.
        position_ids: Positions of shape [batch, sequence], or None to derive
            them from ``cache_position``.
        cache_position: Positions of shape [sequence] within the sequence, or
            None to derive them from ``inputs_embeds``.
        padding_mask: Boolean mask of shape [batch, sequence] that is True on
            padding, or None to derive it from ``attention_mask``.
        shared_kv_states: Store threaded through every decoder layer so a
            kv-shared layer reads the keys/values written by its source layer.
            Must be pytree-opaque (see ``_FSDPSafeSharedKVStates``); a plain dict
            is copied per layer by FSDP2 and loses the writes.

    Returns:
        Hidden states of shape [batch, sequence, hidden].
    """
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    if cache_position is None:
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    if padding_mask is None and attention_mask is not None:
        padding_mask = attention_mask.bool().logical_not()

    mask_kwargs = {
        "config": text_model.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,  # Training-only: no KV cache.
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    # Gemma4 sizes its rotary embeddings per layer type, so build one set per
    # type present in the config rather than a single shared set.
    config_layer_types = getattr(text_model.config, "layer_types", None) or ["full_attention"]
    position_embeddings_map = {
        layer_type: text_model.rotary_emb(inputs_embeds, position_ids, layer_type)
        for layer_type in set(config_layer_types)
    }

    hidden_states = inputs_embeds
    layers = getattr(text_model, "layers", None)
    layer_iter = layers.values() if hasattr(layers, "values") else (layers or ())
    for decoder_layer in layer_iter:
        # Prefer config.layer_types[layer_idx] over the decoder-layer attribute:
        # the attribute lookup defaults to "full_attention" and would hand a
        # sliding-window layer position embeddings built for the wrong head_dim.
        layer_idx = getattr(decoder_layer, "layer_idx", None)
        if layer_idx is not None and layer_idx < len(config_layer_types):
            layer_type = config_layer_types[layer_idx]
        else:
            layer_type = getattr(decoder_layer, "attention_type", "full_attention")

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention")),
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings_map.get(layer_type, position_embeddings_map.get("full_attention")),
            padding_mask=padding_mask,
            shared_kv_states=shared_kv_states,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    if getattr(text_model, "norm", None) is not None:
        hidden_states = text_model.norm(hidden_states)
    return hidden_states


def _has_image_tokens(model: nn.Module, input_ids: torch.Tensor | None) -> bool:
    """Return whether ``input_ids`` of shape [batch, sequence] holds image tokens."""
    image_token_id = getattr(model.config, "image_token_id", None)
    if input_ids is None or image_token_id is None or torch.is_floating_point(input_ids):
        return False
    return bool((input_ids == image_token_id).any())


def gemma4_pipeline_forward(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    pixel_values: torch.Tensor | None,
    image_position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    pp_media_index: torch.Tensor | None,
    shared_kv_states: MutableMapping,
) -> torch.Tensor:
    """Run one Gemma4 VL pipeline stage.

    Args:
        model: The stage's ``Gemma4ForConditionalGeneration``, pruned to the
            modules it owns.
        input_ids: Token ids of shape [batch, sequence] on stage 0; on later
            stages the incoming hidden states of shape [batch, sequence, hidden],
            which the pipeline schedule delivers in this slot.
        attention_mask: Padding mask of shape [batch, sequence], or None.
        position_ids: Positions of shape [batch, sequence], or None.
        inputs_embeds: Embeddings of shape [batch, sequence, hidden], or None.
        pixel_values: Image tensor for the vision tower, or None to take this
            microbatch's chunk from the media staged on stage 0.
        image_position_ids: Image position ids for ``get_image_features``, or
            None to take this microbatch's staged chunk.
        mm_token_type_ids: Token-type ids of shape [batch, sequence] marking
            image positions with 1, or None to locate them from ``input_ids``.
        cache_position: Positions of shape [sequence], or None.
        padding_mask: Boolean mask of shape [batch, sequence] that is True on
            padding, or None.
        pp_media_index: Tensor of shape [microbatch] holding this microbatch's
            index into the media staged on stage 0, or None when the batch
            carries no media.
        shared_kv_states: Per-stage key/value store for kv-sharing layers.

    Returns:
        Logits of shape [batch, sequence, vocab] on the last stage (the one
        owning ``lm_head``), otherwise hidden states of shape
        [batch, sequence, hidden] for the next stage.
    """
    language_model = model.model.language_model
    embed_tokens = getattr(language_model, "embed_tokens", None)
    is_first_stage = embed_tokens is not None

    if is_first_stage:
        # The VLM pipeline collate strips media from the batch and stages one chunk
        # per microbatch on this module, so pull this microbatch's chunk. Only for a
        # microbatch that actually holds image tokens, so a text-only microbatch does
        # not run the vision tower on an empty chunk.
        if pixel_values is None and _has_image_tokens(model, input_ids):
            pixel_values = pp_media_chunk(model, "pixel_values", pp_media_index)
            staged_grid = pp_media_chunk(model, "image_grid_hws", pp_media_index)
            if staged_grid is not None:
                image_position_ids = staged_grid

        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)

        vision_tower = getattr(model.model, "vision_tower", None)
        if vision_tower is not None and pixel_values is not None:
            image_features = model.model.get_image_features(
                pixel_values, image_position_ids=image_position_ids, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            if mm_token_type_ids is not None:
                special_image_mask = mm_token_type_ids == 1
            elif input_ids is not None:
                special_image_mask = input_ids == model.config.image_token_id
            else:
                special_image_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    elif inputs_embeds is None:
        if input_ids is None or not torch.is_floating_point(input_ids):
            raise ValueError(
                "A non-first Gemma4 pipeline stage expects the previous stage's hidden states "
                "as a floating-point tensor in input_ids or inputs_embeds."
            )
        inputs_embeds = input_ids

    hidden_states = gemma4_pipeline_text_forward(
        language_model,
        inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        padding_mask=padding_mask,
        shared_kv_states=shared_kv_states,
    )

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return hidden_states

    logits = lm_head(hidden_states)
    text_config = getattr(model.config, "text_config", model.config)
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return logits
