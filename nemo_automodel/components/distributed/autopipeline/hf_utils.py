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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

if TYPE_CHECKING:
    from transformers import AutoConfig

logger = logging.getLogger(__name__)


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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if getattr(self, "gradient_checkpointing", False) and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        # Embeddings handling
        if inputs_embeds is None:
            if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = self.embed_tokens(input_ids)
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

        # Attention mask
        if not isinstance((causal_mask_mapping := attention_mask), dict):
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if hasattr(self, "has_sliding_layers") and self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # Rotary embeddings
        position_embeddings = None
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if hasattr(self, "layers") and self.layers is not None:
            # Works for dict-like or list-like containers
            layer_iter = self.layers.values() if hasattr(self.layers, "values") else self.layers
            for decoder_layer in layer_iter:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

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
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        if hasattr(self, "norm") and self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if model_class_name == "PipelineStage":
            return hidden_states
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
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
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
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
            issues.append(
                "Encoder-Decoder models with cross-attention are not supported yet for pipeline parallelism."
            )

    if issues:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(issues, 1):
            error_msg += f"{i}. {issue}\n"
        raise ValueError(error_msg)
