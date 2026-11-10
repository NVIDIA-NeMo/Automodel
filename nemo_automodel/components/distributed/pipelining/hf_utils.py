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

from __future__ import annotations

import logging
import types
from collections.abc import Iterable, MutableMapping
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, TypeVar

import torch
import torch.nn as nn

from nemo_automodel.components.distributed._patching import get_undeclared_attr, set_undeclared_attr
from nemo_automodel.components.distributed.pipelining.media import next_vlm_media_chunk

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.cache_utils import Cache
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.modeling_outputs import BaseModelOutputWithPast

    # Attention-type name ("full_attention", "sliding_attention") -> complete mask, as built by
    # transformers.masking_utils; None lets attention run without an explicit mask.
    _CausalMaskMapping: TypeAlias = dict[str, torch.Tensor | BlockMask | None]
    # (sequence length, dtype, device) of the stage input; the inputs that determine an unpadded mask.
    _CausalMaskCacheKey: TypeAlias = tuple[int, torch.dtype, torch.device]

logger = logging.getLogger(__name__)

# Public Hugging Face FlashAttention argument names belong to this HF adapter.
_FLASH_VARLEN_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")

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


class _CausalMaskKwargs(TypedDict):
    """Arguments shared by Hugging Face's full and sliding-window causal-mask builders."""

    config: PreTrainedConfig
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor | None
    past_key_values: Cache | None
    position_ids: torch.Tensor | None


class _CausalMaskOwner(Protocol):
    """Decoder module whose config selects the mask implementation.

    It also owns the stage mask cache, an attribute this module attaches (``_pp_causal_mask_cache``).
    """

    config: PreTrainedConfig


class _SubmoduleAttributes(Protocol):
    """Module whose undeclared attributes resolve to its registered children, parameters or buffers.

    This is ``nn.Module``'s attribute fallback. The pipeline forwards read decoder parts (embeddings,
    layers, norm, rotary embedding) through it and narrow each result.
    """

    def __getattr__(self, name: str, /) -> torch.Tensor | nn.Module:
        """Return the registered submodule, parameter or buffer called ``name``."""
        ...


class _DecoderStage(_SubmoduleAttributes, Protocol):
    """Module the inner pipeline forward is bound to: a decoder backbone, or a module nesting one.

    Its config selects the causal-mask implementation; decoder parts owned by other pipeline
    stages are pruned to None.
    """

    config: PreTrainedConfig


_ModuleT = TypeVar("_ModuleT", bound=_SubmoduleAttributes)


class _CausalLMStage(Protocol):
    """Causal-LM wrapper; submodules owned by other pipeline stages are pruned to None."""

    config: PreTrainedConfig
    model: nn.Module | None
    lm_head: nn.Module | None


class _Gemma4TextStage(Protocol):
    """Gemma4 text decoder; submodules owned by other pipeline stages are pruned to None."""

    config: PreTrainedConfig
    embed_tokens: nn.Module | None
    rotary_emb: nn.Module | None
    layers: nn.ModuleList | nn.ModuleDict | None
    norm: nn.Module | None


class _PooledImageFeatures(Protocol):
    """Vision-tower output whose pooled features are one tensor per image."""

    pooler_output: tuple[torch.Tensor, ...]


class _Gemma4VLMConfig(Protocol):
    """Composite Gemma4 VLM config."""

    text_config: PreTrainedConfig
    image_token_id: int


class _Gemma4VLMBackbone(Protocol):
    """Gemma4 VLM inner model holding the vision tower and the text decoder."""

    language_model: nn.Module

    def get_image_features(
        self, pixel_values: torch.Tensor, /, *, image_position_ids: torch.Tensor | None, return_dict: bool
    ) -> _PooledImageFeatures:
        """Encode images [images, ...] into one pooled feature tensor per image."""
        ...


class _Gemma4VLMStage(Protocol):
    """Gemma4ForConditionalGeneration-style stage; lm_head is None before the last stage."""

    config: _Gemma4VLMConfig
    model: _Gemma4VLMBackbone
    lm_head: nn.Module | None


class _Mistral3VLMBackbone(Protocol):
    """Mistral3 VLM inner model holding the vision tower, projector and the text decoder."""

    language_model: nn.Module

    def get_image_features(
        self,
        *,
        pixel_values: torch.Tensor,
        vision_feature_layer: int | list[int] | None,
        image_sizes: torch.Tensor | None,
        return_dict: bool,
    ) -> _PooledImageFeatures:
        """Encode images [images, channels, height, width] into one pooled feature tensor per image."""
        ...

    def get_placeholder_mask(
        self, input_ids: torch.Tensor | None, /, *, inputs_embeds: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the bool mask [batch, sequence, hidden] of image-token embedding slots."""
        ...


class _Mistral3VLMStage(Protocol):
    """Mistral3ForConditionalGeneration-style stage; lm_head is None before the last stage."""

    config: PreTrainedConfig
    model: _Mistral3VLMBackbone
    lm_head: nn.Module | None


class _InnerPipelineForward(Protocol):
    """Pipeline forward bound to a decoder backbone (or a module nesting one)."""

    def __call__(
        self,
        module: _DecoderStage,
        /,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        causal_mask_mapping: _CausalMaskMapping | None = None,
        **kwargs: object,
    ) -> torch.Tensor | BaseModelOutputWithPast: ...


class _CausalLMPipelineForward(Protocol):
    """Pipeline forward bound to a causal-LM wrapper."""

    def __call__(
        self,
        module: _CausalLMStage,
        /,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _Gemma4TextPipelineForward(Protocol):
    """Pipeline forward bound to a Gemma4 text decoder."""

    def __call__(
        self,
        module: _Gemma4TextStage,
        /,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _Gemma4VLMPipelineForward(Protocol):
    """Pipeline forward bound to a Gemma4 VLM."""

    def __call__(
        self,
        module: _Gemma4VLMStage,
        /,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _Mistral3VLMPipelineForward(Protocol):
    """Pipeline forward bound to a Mistral3 VLM."""

    def __call__(
        self,
        module: _Mistral3VLMStage,
        /,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...


def get_text_module(model: _ModuleT) -> _ModuleT | nn.Module:
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


def _child_module(value: torch.Tensor | nn.Module) -> nn.Module:
    """Narrow an ``nn.Module.__getattr__`` result that names a child module.

    ``nn.Module.__getattr__`` also resolves registered parameters and buffers; the
    pipeline helpers read only children here (inner models, embeddings, rotary
    embeddings and norms).

    Raises:
        TypeError: If the attribute is a parameter or buffer.
    """
    if isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a submodule, got a {type(value).__name__}")
    return value


def _decoder_layers(layers: torch.Tensor | nn.Module) -> Iterable[nn.Module]:
    """Iterate decoder layers held in an ``nn.ModuleList`` or, after stage pruning, an ``nn.ModuleDict``.

    Raises:
        TypeError: If ``layers`` is not a module container.
    """
    if isinstance(layers, nn.ModuleDict):
        return layers.values()
    if isinstance(layers, (nn.ModuleList, nn.Sequential)):
        return layers
    raise TypeError(f"Expected decoder layers in an nn.ModuleList or nn.ModuleDict, got {type(layers).__name__}")


def _build_or_reuse_pp_causal_mask(
    module: _CausalMaskOwner,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | _CausalMaskMapping | None,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
) -> _CausalMaskMapping:
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

    Args:
        module: Decoder module whose config selects the mask implementation.
        inputs_embeds: Hidden states [batch, sequence, hidden]; only shape, dtype and
            device are read.
        attention_mask: Padding mask [batch, sequence], a complete mask, or an already
            built attention-type -> mask mapping.
        cache_position: Sequence offsets [sequence].
        position_ids: Positions [batch, sequence].

    Returns:
        Mapping from attention type to its mask.
    """
    # An ``attention_mask`` that is already a mask-mapping dict is used as-is.
    if isinstance(attention_mask, dict):
        return attention_mask

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    cacheable = attention_mask is None
    cache_key = (inputs_embeds.shape[1], inputs_embeds.dtype, inputs_embeds.device)
    stored_cache = get_undeclared_attr(module, "_pp_causal_mask_cache")
    cache = stored_cache if isinstance(stored_cache, MutableMapping) else None
    if cacheable and cache is not None and cache_key in cache:
        return cache[cache_key]

    # Note: inputs_embeds is only used for shape and dtype, not values.
    mask_kwargs: _CausalMaskKwargs = {
        "config": module.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,  # Training-only: no KV cache
        "position_ids": position_ids,
    }
    causal_mask_mapping: _CausalMaskMapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(module, "has_sliding_layers", False) is True:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    if cacheable:
        if cache is None:
            cache = {}
            set_undeclared_attr(module, "_pp_causal_mask_cache", cache)
        cache[cache_key] = causal_mask_mapping
    return causal_mask_mapping


def create_pipeline_forward_inner(model_class_name: str = "AutoModel") -> _InnerPipelineForward:
    """Create a pipeline-compatible forward method for HuggingFace inner models."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward(
        self: _DecoderStage,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        causal_mask_mapping: _CausalMaskMapping | None = None,
        **kwargs: object,
    ) -> torch.Tensor | BaseModelOutputWithPast:
        """Run this stage's decoder layers.

        ``input_ids`` holds token ids [batch, sequence] on the first stage, or the
        previous stage's floating-point hidden states [batch, sequence, hidden] on a
        stage without ``embed_tokens``; ``inputs_embeds`` is [batch, sequence, hidden].
        ``position_ids`` is [batch, sequence] and ``cache_position`` is [sequence].
        Returns hidden states [batch, sequence, hidden] for ``"PipelineStage"``.
        """
        # For VLM models the text components (embed_tokens, layers, norm) live on a
        # nested text module (e.g. model.language_model) rather than directly on self.
        # get_text_module returns self when no nesting exists (e.g. LlamaModel).
        text_module = get_text_module(self)

        # Embeddings handling
        if inputs_embeds is None:
            if hasattr(text_module, "embed_tokens") and text_module.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = _child_module(text_module.embed_tokens)(input_ids)
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
            position_embeddings = _child_module(rotary_emb)(hidden_states, position_ids)

        if hasattr(text_module, "layers") and text_module.layers is not None:
            # Preserve the public FlashAttention metadata for this microbatch.
            flash_kwargs = {key: kwargs[key] for key in _FLASH_VARLEN_KWARGS if key in kwargs}
            # Works for dict-like or list-like containers
            layer_iter = _decoder_layers(text_module.layers)
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
                    **flash_kwargs,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if hasattr(text_module, "norm") and text_module.norm is not None:
            hidden_states = _child_module(text_module.norm)(hidden_states)

        if model_class_name == "PipelineStage":
            return hidden_states
        else:
            # Hugging Face annotates last_hidden_state as torch.FloatTensor, the legacy float32 CPU tensor
            # class, which hidden states in the model dtype are not. ModelOutput's mapping interface stores
            # each field as the dataclass constructor does: attribute and entry, in field order, with None
            # fields left out (use_cache implies past_key_values is set above).
            output = BaseModelOutputWithPast()
            output["last_hidden_state"] = hidden_states
            if use_cache:
                output["past_key_values"] = past_key_values
            return output

    return pipeline_forward


def create_pipeline_forward_causal_lm() -> _CausalLMPipelineForward:
    """Create a pipeline-compatible forward method for causal LM wrappers."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def pipeline_forward_causal_lm(
        self: _CausalLMStage,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: object,
    ) -> torch.Tensor:
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


def create_pipeline_forward_gemma4_text() -> _Gemma4TextPipelineForward:
    """Pipeline-compatible forward for the Gemma4 text decoder backbone.

    Works for both HF Gemma4TextModel (dense path) and Gemma4MoETextModelBackend (MoE path).
    Handles:
    - Optional embed_tokens (None on non-first PP stages; hidden states arrive in input_ids slot)
    - Both full_attention and sliding_attention causal masks (Gemma4 uses mixed layer types)
    - Per-layer-type position embeddings: Gemma4RotaryEmbedding.forward(x, pos_ids, layer_type)
    """

    def pipeline_forward_gemma4_text(
        self: _Gemma4TextStage,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run retained decoder layers with native or model-prepared masks.

        Args:
            input_ids: Token IDs [batch, sequence] on the first stage, or hidden
                states [batch, sequence, hidden] on a later stage.
            attention_mask: Padding [batch, sequence], complete mask
                [batch, 1, sequence, sequence], or full/sliding mapping of complete
                masks prepared by the owning VLM. Masks are read-only.
            position_ids: Rotary positions [batch, sequence].
            inputs_embeds: Explicit hidden states [batch, sequence, hidden].
            cache_position: Sequence offsets [sequence].
            padding_mask: Optional token padding exclusions [batch, sequence].
            **kwargs: Auxiliary model inputs; no document policy is inferred here.

        Returns:
            Hidden states [batch, sequence, hidden].
        """
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

        if isinstance(attention_mask, dict):
            # The owning VLM has already composed document, image and window policy.
            causal_mask_mapping = attention_mask
        else:
            if padding_mask is None and attention_mask is not None:
                padding_mask = attention_mask.bool().logical_not()

            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs: _CausalMaskKwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # Per-layer-type rotary embeddings: Gemma4RotaryEmbedding takes (x, pos_ids, layer_type)
        position_embeddings_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            for lt in set(getattr(self.config, "layer_types", ["full_attention"])):
                try:
                    position_embeddings_map[lt] = self.rotary_emb(inputs_embeds, position_ids, lt)
                except TypeError:
                    position_embeddings_map[lt] = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        config_layer_types = getattr(self.config, "layer_types", None)
        # HF transfomers v5.8.1+ Gemma4 attention threads a single mutable mapping through every decoder
        # layer for kv-sharing: "full-length" layers (store_full_length_kv) write their
        # keys/values into it and kv-shared layers (is_kv_shared_layer) read them back.
        # The decoder-layer kwarg defaults to None, so a store layer would execute
        # ``None[layer_type] = ...`` -> TypeError. Create one per stage and thread it,
        # mirroring HF's Gemma4TextModel.forward. gemma-4-31B, 26B-A4B have num_kv_shared_layers=0
        # (no readers, store is a harmless no-op write); for a kv-sharing model split
        # across PP stages only within-stage sharing would work here.
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if hasattr(self, "layers") and self.layers is not None:
            layer_iter = _decoder_layers(self.layers)
            for decoder_layer in layer_iter:
                # Prefer config.layer_types[layer_idx] over decoder_layer attribute — the
                # attribute lookup defaults to "full_attention" and mis-assigns position
                # embeddings (wrong head_dim) to sliding-window layers. Hugging Face decoder
                # layers store their int index as a plain attribute nn.Module does not declare.
                layer_idx = get_undeclared_attr(decoder_layer, "layer_idx")
                if config_layer_types is not None and isinstance(layer_idx, int):
                    layer_type = (
                        config_layer_types[layer_idx] if layer_idx < len(config_layer_types) else "full_attention"
                    )
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
                    shared_kv_states=shared_kv_states,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if hasattr(self, "norm") and self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    return pipeline_forward_gemma4_text


def create_pipeline_forward_gemma4_vlm() -> _Gemma4VLMPipelineForward:
    """Pipeline-compatible forward for Gemma4ForConditionalGeneration (VLM top-level).

    Stage 0: embeds text tokens, merges image features from vision tower (if pixel_values
    provided or stored in _vlm_pixel_values_chunks), then calls the patched language model.
    Non-first stages: passes hidden states straight to the patched language model.
    Last stage: applies lm_head and final-logit softcapping.
    """

    def pipeline_forward_gemma4_vlm(
        self: _Gemma4VLMStage,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return logits [batch, sequence, vocab] on the last stage, else hidden states.

        ``input_ids`` holds token ids [batch, sequence] on the first stage and hidden
        states [batch, sequence, hidden] on later stages. ``pixel_values`` and
        ``image_position_ids`` use the Gemma4 processor layouts.
        """
        if (
            kwargs.get("_packed_seq_ids") is not None
            and attention_mask is None
            and not getattr(self, "_cp_enabled", False)
        ) or any(kwargs.get(key) is not None for key in ("cu_seq_lens_q", "cu_seq_lens_k")):
            validator = getattr(self, "validate_packed_attention_backend", None)
            if validator is not None:
                validator(self.config.text_config._attn_implementation)
            else:
                raise ValueError("Packed Flash Gemma4 VLM pipeline attention is unsupported; use SDPA/eager")
            if (
                kwargs.get("_packed_seq_ids") is not None
                and attention_mask is None
                and not getattr(self, "_cp_enabled", False)
            ):
                raise ValueError("Configure packed attention before forwarding compact Gemma4 attention IDs")
        lang_model = self.model.language_model
        embed_tokens = getattr(lang_model, "embed_tokens", None)
        is_first_stage = embed_tokens is not None

        if pixel_values is None and is_first_stage:
            media = next_vlm_media_chunk(self)
            if media is not None:
                pixel_values = media.pixel_values
                image_position_ids = media.image_grid

        if is_first_stage:
            if inputs_embeds is None:
                inputs_embeds = embed_tokens(input_ids)

            vision_tower = getattr(self.model, "vision_tower", None)
            if vision_tower is not None and pixel_values is not None:
                image_features = self.model.get_image_features(
                    pixel_values, image_position_ids=image_position_ids, return_dict=True
                ).pooler_output
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

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

        if kwargs.get("_packed_seq_ids") is not None:
            prepare_mask = getattr(self, "prepare_document_attention_mask", None)
            if prepare_mask is None:
                raise ValueError("Packed Gemma4 PP requires the AutoModel model-owned document-mask policy")
            attention_mask = prepare_mask(attention_mask, mm_token_type_ids, kwargs["_packed_seq_ids"])

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


def create_pipeline_forward_mistral3_vlm() -> _Mistral3VLMPipelineForward:
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
        self: _Mistral3VLMStage,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return logits [batch, sequence, vocab] on the last stage, else hidden states.

        ``input_ids`` holds token ids [batch, sequence] on the first stage and hidden
        states [batch, sequence, hidden] on later stages. ``pixel_values`` is
        [images, channels, height, width] and ``image_sizes`` is [images, 2].
        """
        inner = self.model
        lang_model = inner.language_model
        embed_tokens = getattr(lang_model, "embed_tokens", None)
        is_first_stage = embed_tokens is not None

        if pixel_values is None and is_first_stage:
            media = next_vlm_media_chunk(self)
            if media is not None:
                pixel_values = media.pixel_values
                image_sizes = media.image_grid

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


def patch_hf_model_for_pp(model: nn.Module, patch_inner_model: bool = True, patch_causal_lm_model: bool = True) -> None:
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
            # Recipes read this flag to keep the last stage's lm_head deferred to fused cross-entropy.
            set_undeclared_attr(model, "_pp_return_hidden_states_supported", True)
    else:
        if patch_inner_model:
            model.forward = types.MethodType(create_pipeline_forward_inner("PipelineStage"), model)


def init_hf_model_buffers(model: torch.nn.Module, device: torch.device) -> None:
    """Initialize HuggingFace model buffers needed before pipeline execution.

    Recomputes the rotary embedding's ``inv_freq`` on ``device`` when the embedding module exposes
    Hugging Face's ``rope_init_fn(config, device) -> (inv_freq, attention_scaling)`` attribute.

    Raises:
        TypeError: If ``rope_init_fn`` does not return a tuple.
    """
    rotary_owner = getattr(model, "model", model)
    if not hasattr(rotary_owner, "rotary_emb"):
        return
    rotary_emb = rotary_owner.rotary_emb
    if not isinstance(rotary_emb, nn.Module):
        return
    rope_init_fn = get_undeclared_attr(rotary_emb, "rope_init_fn")
    if not callable(rope_init_fn):
        return
    rope_parameters = rope_init_fn(rotary_emb.config, device)
    if not isinstance(rope_parameters, tuple):
        raise TypeError(f"rope_init_fn must return (inv_freq, attention_scaling), got {type(rope_parameters).__name__}")
    inv_freq, _ = rope_parameters
    rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


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

    # Inspect the actual alias: a composite config's nested tie flag can disagree
    # with its controlling outer flag. Identity also works before meta materialization.
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

    if config is not None:
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
