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
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast

from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.models.common.packing import flatten_packed_sequence_metadata
from nemo_automodel.shared.import_utils import safe_import_from

if TYPE_CHECKING:
    from mamba_ssm.utils.generation import InferenceParams


@dataclass(frozen=True)
class _PackedInput:
    """Loader metadata for the documents packed into padded rows.

    Attributes:
        indices: int64 Tensor of shape [batch, sequence] holding row-local token
            positions with ``-1`` padding, or a pre-flattened Tensor of shape
            [tokens] indexing the row-major [batch, sequence] layout.
        cu: int32 Tensor of shape [batch, max_documents + 1] holding row-local
            cumulative document lengths with ``-1`` after each row's last
            boundary, or a pre-flattened Tensor of shape [documents + 1] over
            the unpadded token stream.
        maximum: Longest document in tokens, excluding the memory prefix.
    """

    indices: torch.Tensor
    cu: torch.Tensor
    maximum: int


@dataclass(frozen=True)
class _PhysicalDocuments:
    """Boundaries of the expanded stream, where every document starts with its own memory prefix.

    Attributes:
        cu: int32 Tensor of shape [documents + 1] holding cumulative lengths of
            the prefixed documents over the [physical_tokens] stream, where
            physical_tokens = tokens + documents * prefix.
        seq_idx: int32 Tensor of shape [1, physical_tokens] holding each
            physical token's document index.
        maximum: Longest prefixed document, in tokens.
    """

    cu: torch.Tensor
    seq_idx: torch.Tensor
    maximum: int


def _expand_documents(
    embeddings: torch.Tensor,
    memory: torch.Tensor,
    packed: _PackedInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _PhysicalDocuments]:
    """Insert one learned prefix per document, retaining differentiable gathers.

    Args:
        embeddings: Tensor of shape [batch, sequence, hidden] holding the padded
            packed rows' token embeddings.
        memory: Tensor of shape [prefix, hidden] holding the learned memory
            tokens prepended to every document.
        packed: Locations of the real tokens and document boundaries in
            ``embeddings``.

    Returns:
        A tuple of:
        - Tensor of shape [1, physical_tokens, hidden], where physical_tokens =
          tokens + documents * prefix. Each document's prefix precedes its
          tokens; the tensor is new and differentiable with respect to
          ``embeddings`` and ``memory``.
        - int64 Tensor of shape [1, physical_tokens] holding positions that
          restart at zero at each prefix.
        - int64 Tensor of shape [tokens] holding each real token's physical
          position.
        - int64 Tensor of shape [tokens] holding each real token's index into
          ``embeddings.flatten(0, 1)``.
        - The prefixed documents' boundaries.
    """
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
    past_key_values: Cache | list[torch.Tensor] | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    calc_logits_for_entire_prompt: bool = True,
    fla_past_key_values: Cache | None = None,
    mamba_inference_params: "InferenceParams | None" = None,
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
    """Route loader packing metadata into the released causal-LM forward.

    Args:
        input_ids: Tensor of shape [batch, sequence] holding token IDs. Packed
            rows hold several documents followed by padding.
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask. Packed calls replace it with the packing metadata.
        position_ids: Tensor of shape [batch, sequence]. Packed calls ignore it
            and restart positions at each document's memory prefix.
        past_key_values: Release attention cache; packed calls reject it.
        inputs_embeds: Tensor of shape [batch, sequence, hidden], used instead
            of ``input_ids``.
        labels: Tensor of shape [batch, sequence] for the release's own loss.
        use_cache: Whether to return caches; packed calls reject True.
        output_attentions: Release flag; packed calls reject True.
        output_hidden_states: Release flag; packed calls reject True.
        return_dict: Whether the release returns a ``ModelOutput``.
        calc_logits_for_entire_prompt: Release flag selecting full-prompt logits.
        fla_past_key_values: Release DeltaNet cache; packed calls reject it.
        mamba_inference_params: Release Mamba cache; packed calls reject it.
        packed_token_indices: Real-token positions; see ``_PackedInput.indices``.
        cu_seqlens: Document boundaries; see ``_PackedInput.cu``.
        max_seqlen: Longest document in tokens.
        cu_seq_lens_q: int32 Tensor of shape [segments + 1] from the generic
            FlashAttention packer; rejected without ``packed_token_indices``.
        cu_seq_lens_k: int32 Tensor of shape [segments + 1]; rejected likewise.
        max_length_q: Generic FlashAttention maximum; rejected likewise.
        max_length_k: Generic FlashAttention maximum; rejected likewise.
        padding_mask: bool Tensor of shape [batch, sequence] from the collater;
            ignored.
        _packed_seq_ids: Tensor of shape [batch, sequence] holding document IDs
            from the collater; ignored.

    Returns:
        The release's causal-LM output. Its logits have shape [batch, sequence,
        vocab] in the padded input layout; packed padding positions carry zero
        hidden states.

    Raises:
        ValueError: If packing metadata is incomplete, a cache is requested
            with packing, or only generic FlashAttention metadata is supplied.
    """
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
    attention_mask: torch.Tensor | _PackedInput | None = None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    """Run the released backbone over prefixed documents when packing metadata is present.

    Args:
        input_ids: Tensor of shape [batch, sequence] holding token IDs.
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask, or the packing metadata on packed calls.
        position_ids: Tensor of shape [batch, sequence]; unused on packed calls.
        inputs_embeds: Tensor of shape [batch, sequence, hidden], used instead
            of ``input_ids``.
        **kwargs: Remaining release arguments, forwarded unchanged on unpacked
            calls.

    Returns:
        On packed calls, a ``MoeModelOutputWithPast`` whose last_hidden_state
        is a new Tensor of shape [batch, sequence, hidden] in the padded input
        layout, without memory prefixes and with zeros at padding positions; a
        one-element tuple holding that tensor when ``return_dict`` is False.
        Unpacked calls return the release's output.

    Raises:
        ValueError: If a packed call requests attention or hidden-state outputs.
    """
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


def _mamba_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | _PhysicalDocuments | None = None,
    **kwargs: Any,
) -> Any:
    """Reset the Mamba2 convolution and scan state at packed document boundaries.

    Args:
        hidden_states: Tensor of shape [batch, sequence, hidden]. Packed calls
            pass the prefixed stream as [1, physical_tokens, hidden].
        attention_mask: The release's mask, or the prefixed documents'
            boundaries on packed calls.
        **kwargs: Remaining release arguments, forwarded unchanged on unpacked
            calls.

    Returns:
        A tuple of a Tensor with the shape of ``hidden_states`` and the
        release's unchanged ``past_key_value``.

    Raises:
        ValueError: If a packed call uses Mamba tensor parallelism.
    """
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


def _delta_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | _PhysicalDocuments | None = None,
    **kwargs: Any,
) -> Any:
    """Pass packed document boundaries to the released DeltaNet kernels.

    Args:
        hidden_states: Tensor of shape [batch, sequence, hidden]. Packed calls
            pass the prefixed stream as [1, physical_tokens, hidden].
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask, or the prefixed documents' boundaries. On packed
            calls the boundaries reach the delta rule as ``cu_seqlens`` and the
            short convolutions through the release's ``position_ids`` argument.
        **kwargs: Remaining release arguments.

    Returns:
        The release's DeltaNet output tuple, whose first element has the shape
        of ``hidden_states``.
    """
    if isinstance(attention_mask, _PhysicalDocuments):
        kwargs.update(position_ids=attention_mask, cu_seqlens=attention_mask.cu)
        attention_mask = None
    return self._nemo_unpacked_forward(hidden_states, attention_mask=attention_mask, **kwargs)


def _short_convolution_forward(
    self: nn.Module,
    *args: Any,
    seq_idx: torch.Tensor | _PhysicalDocuments | None = None,
    **kwargs: Any,
) -> Any:
    """Translate packed boundaries into the installed FLA short-convolution API.

    Args:
        *args: Positional release arguments.
        seq_idx: int32 Tensor of shape [batch, sequence] holding document
            indices, or the prefixed documents' boundaries on packed calls.
            Those become ``cu_seqlens`` when the installed convolution accepts
            it and otherwise its ``seq_idx`` tensor.
        **kwargs: Remaining release arguments, including ``x``: Tensor of shape
            [batch, sequence, channels].

    Returns:
        The convolution's output tuple, whose first element has the shape of
        ``x``.
    """
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
    attention_mask: torch.Tensor | _PhysicalDocuments | None,
    query_length: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Run released FlashAttention-2 inputs with variable-length document boundaries.

    Args:
        query: Tensor of shape [batch, sequence, heads, head_dim]. Packed calls
            pass [1, physical_tokens, heads, head_dim].
        key: Tensor of shape [batch, sequence, heads, head_dim], already
            repeated to the query heads by the release.
        value: Tensor of shape [batch, sequence, heads, value_head_dim].
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask, or the prefixed documents' boundaries.
        query_length: Query sequence length.
        **kwargs: Release options: ``dropout``, ``softmax_scale`` and
            ``use_sliding_windows``.

    Returns:
        Tensor of shape [batch, sequence, heads, value_head_dim].
    """
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


def apply_packing_adapters(model: nn.Module, *, mesh: MeshContext | None = None) -> nn.Module:
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
