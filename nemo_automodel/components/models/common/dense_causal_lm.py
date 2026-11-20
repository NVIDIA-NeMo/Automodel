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

"""Causal LM forward shared by the dense Llama, Qwen2 and Qwen3 models."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.utils import compute_lm_head_logits


def dense_causal_lm_forward(
    decoder: nn.Module,
    lm_head: nn.Module,
    loss_function: Callable[..., torch.FloatTensor],
    config: PreTrainedConfig,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: Cache | None,
    inputs_embeds: torch.Tensor | None,
    labels: torch.Tensor | None,
    use_cache: bool | None,
    output_attentions: bool | None,
    output_hidden_states: bool | None,
    return_dict: bool | None,
    cache_position: torch.Tensor | None,
    logits_to_keep: int | torch.Tensor,
    /,
    **kwargs: torch.Tensor | float | str | None,
) -> CausalLMOutputWithPast | tuple:
    """Run a dense decoder and its LM head for BSHD or packed THD inputs.

    Args:
        decoder: The decoder stack returning ``BaseModelOutputWithPast``.
        lm_head: The language-model head projecting hidden size ``H`` to ``V``.
        loss_function: The causal LM loss called with logits and labels.
        config: The model config supplying output defaults and ``vocab_size``.
        input_ids: Token IDs ``[B, S]`` or packed local IDs ``[T]``.
        attention_mask: Optional padded mask. THD uses ``cu_seqlens``.
        position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
        past_key_values: Optional BSHD KV cache; unsupported for THD.
        inputs_embeds: Optional hidden inputs ``[B, S, H]`` or ``[T, H]``.
        labels: Optional labels ``[B, S]`` or packed ``[T]``.
        use_cache: Whether to update the BSHD KV cache.
        output_attentions: Whether to request attention outputs.
        output_hidden_states: Whether to return per-layer hidden states.
        return_dict: Whether to return ``CausalLMOutputWithPast``.
        cache_position: Optional BSHD cache positions ``[S]``.
        logits_to_keep: Positions to project from ``H`` to ``V``.
        **kwargs: THD metadata. ``cu_seqlens`` is ``[N + 1]`` and CP adds
            ``cp_size`` and ``cp_rank``.

    Returns:
        Causal LM output with BSHD logits ``[B, S, V]``. Packed THD logits are
        restored to ``[1, T, V]`` for the shared loss contract.
    """
    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    return_dict = return_dict if return_dict is not None else config.use_return_dict

    is_thd = kwargs.get("qkv_format") == "thd"
    if is_thd:
        if position_ids is None:
            raise ValueError("Packed THD input requires position_ids.")
        kwargs.pop("padding_mask", None)
        if input_ids is not None and input_ids.ndim > 1:
            input_ids = input_ids.squeeze(0)
        if position_ids.ndim > 1:
            position_ids = position_ids.squeeze(0)
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "max_seqlen":
                kwargs[key] = value.item()
                continue
            if value.ndim > 1:
                value = value.squeeze(0)
            if key in ("cu_seqlens", "cu_seqlens_padded"):
                value = value[value != -1000].contiguous()
            kwargs[key] = value
        if inputs_embeds is not None and inputs_embeds.ndim > 2:
            inputs_embeds = inputs_embeds.squeeze(0)
        if labels is not None and labels.ndim > 1:
            labels = labels.squeeze(0)
        attention_mask = None

    # Always use return_dict internally so we can reliably access fields.
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    logits = compute_lm_head_logits(lm_head, outputs.last_hidden_state, logits_to_keep, is_thd=is_thd).logits

    loss = None
    if labels is not None:
        if is_thd and labels.ndim == 1:
            labels = labels.unsqueeze(0)
        loss = loss_function(logits=logits, labels=labels, vocab_size=config.vocab_size, **kwargs)

    out = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
    )
    if return_dict:
        return out
    return out.to_tuple()
