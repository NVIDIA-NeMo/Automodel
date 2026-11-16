# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Packed execution for the released Flash-1B modules, including memory tokens.

Uses the unmodified modules from nvidia/Nemotron-Flash-1B revision
da87f707f678b5936e67fdcfe598f76ebcc5fdbc. Each document receives its own
learned prefix; attention, short convolutions, and recurrent states share the
same expanded document boundaries. Checkpoint parameter names remain unchanged.
"""

import inspect
from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast

from nemo_automodel.components.models.common.packing import flatten_packed_sequence_metadata
from nemo_automodel.shared.import_utils import safe_import_from


@dataclass(frozen=True)
class _PackedInput:
    indices: torch.Tensor
    cu: torch.Tensor
    maximum: int


@dataclass(frozen=True)
class _PhysicalDocuments:
    cu: torch.Tensor
    seq_idx: torch.Tensor
    maximum: int


def _expand_documents(
    embeddings: torch.Tensor,
    memory: torch.Tensor,
    packed: _PackedInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _PhysicalDocuments]:
    """Insert one learned prefix per document, retaining differentiable gathers."""
    indices, cu = flatten_packed_sequence_metadata(
        packed.indices,
        packed.cu,
        batch_size=embeddings.shape[0],
        sequence_length=embeddings.shape[1],
    )
    indices = indices.to(device=embeddings.device, non_blocking=True)
    cu = cu.to(device=embeddings.device, non_blocking=True)
    lengths = cu[1:] - cu[:-1]
    count = lengths.numel()
    prefix = memory.shape[0]
    total = indices.numel() + count * prefix
    doc = torch.repeat_interleave(
        torch.arange(count, device=embeddings.device),
        lengths,
        output_size=indices.numel(),
    )
    destinations = torch.arange(indices.numel(), device=embeddings.device) + (doc + 1) * prefix
    physical_cu = (cu + torch.arange(count + 1, device=cu.device) * prefix).to(torch.int32)
    memory_indices = (physical_cu[:-1, None] + torch.arange(prefix, device=cu.device)).flatten().long()
    hidden = embeddings.new_zeros((total, embeddings.shape[-1]))
    hidden = hidden.index_copy(0, destinations, embeddings.flatten(0, 1)[indices])
    hidden = hidden.index_copy(0, memory_indices, memory.expand(count, -1, -1).reshape(-1, memory.shape[-1]))
    seq_idx = torch.repeat_interleave(
        torch.arange(count, device=cu.device, dtype=torch.int32),
        lengths + prefix,
        output_size=total,
    ).unsqueeze(0)
    starts = torch.repeat_interleave(physical_cu[:-1], lengths + prefix, output_size=total)
    positions = (torch.arange(total, device=cu.device) - starts).long().unsqueeze(0)
    return (
        hidden.unsqueeze(0),
        positions,
        destinations,
        indices,
        _PhysicalDocuments(physical_cu, seq_idx, packed.maximum + prefix),
    )


def _root_forward(
    self: nn.Module,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: object = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    calc_logits_for_entire_prompt: bool = True,
    fla_past_key_values: object = None,
    mamba_inference_params: object = None,
    *,
    packed_token_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    padding_mask: torch.Tensor | None = None,
    _packed_seq_ids: torch.Tensor | None = None,
) -> Any:
    del padding_mask, _packed_seq_ids
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        calc_logits_for_entire_prompt=calc_logits_for_entire_prompt,
        fla_past_key_values=fla_past_key_values,
        mamba_inference_params=mamba_inference_params,
    )
    if packed_token_indices is not None:
        if cu_seqlens is None or max_seqlen is None:
            raise ValueError("Flash-1B packing requires gathered-token indices, lengths, and a maximum")
        if kwargs.get("use_cache") or any(
            kwargs.get(key) is not None for key in ("past_key_values", "fla_past_key_values", "mamba_inference_params")
        ):
            raise ValueError("Packed Flash-1B execution does not support a cache")
        kwargs["use_cache"] = False
        kwargs["attention_mask"] = _PackedInput(packed_token_indices, cu_seqlens, max_seqlen)
    elif any(x is not None for x in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)):
        raise ValueError("Packed Flash-1B requires gathered-token metadata from its loader contract")
    return self._nemo_unpacked_forward(**kwargs)


def _backbone_forward(
    self: nn.Module,
    input_ids: torch.Tensor | None = None,
    attention_mask: object = None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    if not isinstance(attention_mask, _PackedInput):
        return self._nemo_unpacked_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
    if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
        raise ValueError("Packed Flash-1B does not expose intermediate attention/hidden-state outputs")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    hidden, positions, destinations, indices, physical = _expand_documents(
        inputs_embeds, self.memory_tokens, attention_mask
    )
    for index, layer in enumerate(self.layers):
        forward = partial(
            layer.__call__,
            attention_mask=physical,
            position_ids=positions,
            use_cache=False,
            use_swa=self.sliding_window is not None and index not in self.global_attn_idx,
        )
        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(forward, hidden)
        else:
            outputs = forward(hidden)
        hidden = outputs[0]
    hidden = self.final_layernorm(hidden)[:, destinations]
    restored = inputs_embeds.new_zeros(inputs_embeds.shape).flatten(0, 1)
    restored = restored.index_copy(0, indices, hidden.squeeze(0))
    output = MoeModelOutputWithPast(last_hidden_state=restored.reshape_as(inputs_embeds))
    return output if kwargs.get("return_dict", True) is not False else (output.last_hidden_state,)


def _mamba_forward(self: nn.Module, hidden_states: torch.Tensor, attention_mask: object = None, **kwargs: Any) -> Any:
    if not isinstance(attention_mask, _PhysicalDocuments):
        return self._nemo_unpacked_forward(hidden_states, attention_mask=attention_mask, **kwargs)
    if self.process_group is not None:
        raise ValueError("Packed Flash-1B does not support Mamba tensor parallelism")
    # The release's non-fused convolution drops seq_idx. Invoke the public fused
    # primitive directly, without mutating use_mem_eff_path during a forward.
    _, scan = safe_import_from("mamba_ssm.ops.triton.ssd_combined", "mamba_split_conv1d_scan_combined")
    skip = self.D.reshape(-1, self.headdim) if self.D_has_hdim else self.D
    output = scan(
        self.in_proj(hidden_states),
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        self.dt_bias,
        -self.A_log.float().exp(),
        D=skip,
        chunk_size=self.chunk_size,
        seq_idx=attention_mask.seq_idx,
        activation=self.activation,
        rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
        rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
        headdim=None if self.D_has_hdim else self.headdim,
        ngroups=self.ngroups,
        norm_before_gate=self.norm_before_gate,
        dt_limit=self.dt_limit,
    )
    # Keep the module call: its forward can own trainable LoRA adapters.
    return self.out_proj(output), kwargs.get("past_key_value")


def _delta_forward(self: nn.Module, hidden_states: torch.Tensor, attention_mask: object = None, **kwargs: Any) -> Any:
    if isinstance(attention_mask, _PhysicalDocuments):
        kwargs.update(position_ids=attention_mask, cu_seqlens=attention_mask.cu)
        attention_mask = None
    return self._nemo_unpacked_forward(hidden_states, attention_mask=attention_mask, **kwargs)


def _short_convolution_forward(self: nn.Module, *args: Any, seq_idx: object = None, **kwargs: Any) -> Any:
    if isinstance(seq_idx, _PhysicalDocuments):
        if self._nemo_packing_uses_cu:
            kwargs["cu_seqlens"] = seq_idx.cu
            return self._nemo_unpacked_forward(*args, **kwargs)
        seq_idx = seq_idx.seq_idx
    return self._nemo_unpacked_forward(*args, seq_idx=seq_idx, **kwargs)


def _attention_forward(
    self: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: object,
    query_length: int,
    **kwargs: Any,
) -> torch.Tensor:
    if not isinstance(attention_mask, _PhysicalDocuments):
        return self._nemo_unpacked_flash(query, key, value, attention_mask, query_length, **kwargs)
    _, flash = safe_import_from("flash_attn", "flash_attn_varlen_func")
    window = self.config.sliding_window if kwargs.get("use_sliding_windows") else None
    result = flash(
        query.squeeze(0),
        key.squeeze(0),
        value.squeeze(0),
        attention_mask.cu,
        attention_mask.cu,
        attention_mask.maximum,
        attention_mask.maximum,
        dropout_p=kwargs.get("dropout", 0.0),
        softmax_scale=kwargs.get("softmax_scale"),
        causal=True,
        window_size=(-1, -1) if window is None else (window, window),
    )
    return result.unsqueeze(0)


def _validate_backend(self: nn.Module, attn_implementation: str) -> None:
    if attn_implementation != "flash_attention_2" or self.model._attn_implementation != "flash_attention_2":
        raise ValueError("Packed Flash-1B currently requires flash_attention_2")


def apply_packing_adapters(model: nn.Module, *, mesh: object = None) -> nn.Module:
    """Add document-aware training while preserving released weights and generation.

    Args:
        model: Released remote Flash-1B causal LM, before distributed wrapping.
        mesh: Construction context supplied by the runtime-hook dispatcher.

    Returns:
        The original model with an opt-in packed path and unchanged parameter names.
    """
    del mesh
    if hasattr(model, "_nemo_unpacked_forward"):
        return model
    model.requires_packed_sequence_metadata = True
    model.validate_packed_attention_backend = MethodType(_validate_backend, model)
    model._nemo_unpacked_forward = model.forward
    model.forward = MethodType(_root_forward, model)
    backbone = model.model
    backbone._nemo_unpacked_forward = backbone.forward
    backbone.forward = MethodType(_backbone_forward, backbone)
    for kind, layer in zip(model.config.layer_types, backbone.layers, strict=True):
        if kind in ("m", "m2"):
            mixer = layer.mamba
            mixer._nemo_unpacked_forward = mixer.forward
            mixer.forward = MethodType(_mamba_forward, mixer)
        elif kind == "deltanet":
            mixer = layer.gla
            mixer._nemo_unpacked_forward = mixer.forward
            mixer.forward = MethodType(_delta_forward, mixer)
            for convolution in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d):
                # Modern FLA replaced seq_idx with cu_seqlens; the release's
                # legacy keyword is otherwise silently ignored through **kwargs.
                convolution._nemo_packing_uses_cu = "cu_seqlens" in inspect.signature(convolution.forward).parameters
                convolution._nemo_unpacked_forward = convolution.forward
                convolution.forward = MethodType(_short_convolution_forward, convolution)
        elif kind == "a":
            attention = layer.self_attn
            if type(attention).__name__ == "NemotronFlashFlashAttention2":
                attention._nemo_unpacked_flash = attention._flash_attention_forward
                attention._flash_attention_forward = MethodType(_attention_forward, attention)
    return model
