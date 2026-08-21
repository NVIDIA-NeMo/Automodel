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

"""Pipeline-stage forward for the Mistral3 VLM (Pixtral vision + Ministral3 text).

A pipeline stage owns a slice of the model: stage 0 keeps ``embed_tokens``, the
vision tower and the projector, the last stage keeps ``norm`` and ``lm_head``,
and every stage keeps a contiguous span of decoder layers. The stock HuggingFace
forwards cannot run on such a slice, so
``Mistral3FP8VLMForConditionalGeneration`` routes pipeline stages here instead
(``PipelineForwardStyle.MODEL``).

Without this, the generic causal-LM pipeline forward would never call the vision
tower and the image placeholder tokens would train as ordinary text embeddings.
"""

import torch
import torch.nn as nn

from nemo_automodel.shared.pipeline import pp_media_chunk


def _has_image_tokens(model: nn.Module, input_ids: torch.Tensor | None) -> bool:
    """Return whether ``input_ids`` of shape [batch, sequence] holds image tokens."""
    image_token_id = getattr(model.config, "image_token_id", None)
    if input_ids is None or image_token_id is None or torch.is_floating_point(input_ids):
        return False
    return bool((input_ids == image_token_id).any())


def mistral3_pipeline_text_forward(
    text_model: nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Run the Ministral3 decoder layers this pipeline stage owns.

    Args:
        text_model: Ministral3 backbone pruned to this stage's layers. Only the
            last stage still owns ``norm``.
        inputs_embeds: Hidden states of shape [batch, sequence, hidden] -- token
            embeddings on stage 0, the previous stage's output afterwards.
        attention_mask: Padding mask of shape [batch, sequence], or None.
        position_ids: Positions of shape [batch, sequence], or None to derive
            them from the sequence length of ``inputs_embeds``.

    Returns:
        Hidden states of shape [batch, sequence, hidden].
    """
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Ministral3 applies one mask to every layer: sliding-window when the config
    # declares a window, plain causal otherwise (mirrors Ministral3Model.forward).
    mask_function = (
        create_sliding_window_causal_mask
        if getattr(text_model.config, "sliding_window", None) is not None
        else create_causal_mask
    )
    causal_mask = mask_function(
        config=text_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,  # Training-only: no KV cache.
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids=position_ids)

    layers = getattr(text_model, "layers", None)
    layer_iter = layers.values() if hasattr(layers, "values") else (layers or ())
    for decoder_layer in layer_iter:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    if getattr(text_model, "norm", None) is not None:
        hidden_states = text_model.norm(hidden_states)
    return hidden_states


def mistral3_vlm_pipeline_forward(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    pixel_values: torch.Tensor | None,
    image_sizes: torch.Tensor | None,
    pp_media_index: torch.Tensor | None,
) -> torch.Tensor:
    """Run one Mistral3 VLM pipeline stage.

    Args:
        model: The stage's ``Mistral3FP8VLMForConditionalGeneration``, pruned to
            the modules it owns.
        input_ids: Token ids of shape [batch, sequence] on stage 0; on later
            stages the incoming hidden states of shape [batch, sequence, hidden],
            which the pipeline schedule delivers in this slot.
        attention_mask: Padding mask of shape [batch, sequence], or None.
        position_ids: Positions of shape [batch, sequence], or None.
        inputs_embeds: Embeddings of shape [batch, sequence, hidden], or None.
        pixel_values: Image patches for the vision tower, or None to take this
            microbatch's staged chunk.
        image_sizes: Image sizes of shape [images, 2] driving the patch split, or
            None to take this microbatch's staged chunk.
        pp_media_index: Tensor of shape [microbatch] holding this microbatch's
            index into the media staged on stage 0, or None when the batch
            carries no media.

    Returns:
        Logits of shape [batch, sequence, vocab] on the last stage (the one
        owning ``lm_head``), otherwise hidden states of shape
        [batch, sequence, hidden] for the next stage.
    """
    inner = model.model
    language_model = inner.language_model
    embed_tokens = getattr(language_model, "embed_tokens", None)
    is_first_stage = embed_tokens is not None

    if is_first_stage:
        # The VLM pipeline collate strips media from the batch and stages one chunk
        # per microbatch on this module, so pull this microbatch's chunk. Only for a
        # microbatch that actually holds image tokens, so a text-only microbatch does
        # not run the vision tower on an empty chunk.
        if pixel_values is None and _has_image_tokens(model, input_ids):
            pixel_values = pp_media_chunk(model, "pixel_values", pp_media_index)
            staged_image_sizes = pp_media_chunk(model, "image_grid_hws", pp_media_index)
            if staged_image_sizes is not None:
                image_sizes = staged_image_sizes

        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)

        vision_tower = getattr(inner, "vision_tower", None)
        if vision_tower is not None and pixel_values is not None:
            # HF resolves vision_feature_layer from the config through a decorator on
            # its own forward, which this stage forward bypasses; resolve it here so
            # the selected vision hidden state matches the non-pipeline path.
            image_features = inner.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=model.config.vision_feature_layer,
                image_sizes=image_sizes,
                return_dict=True,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = inner.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    elif inputs_embeds is None:
        if input_ids is None or not torch.is_floating_point(input_ids):
            raise ValueError(
                "A non-first Mistral3 pipeline stage expects the previous stage's hidden states "
                "as a floating-point tensor in input_ids or inputs_embeds."
            )
        inputs_embeds = input_ids

    hidden_states = mistral3_pipeline_text_forward(
        language_model,
        inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return hidden_states
    return lm_head(hidden_states)
