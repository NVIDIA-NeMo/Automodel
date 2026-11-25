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

"""Thread public FlashAttention boundaries through the released DeciLM API.

The publisher's classes and checkpoint layout are retained. Reference interface:
nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 at
420ba7d28211abf116b8b103ab700d92619daf98 (Apache-2.0 modeling_decilm.py).
No publisher implementation is copied here.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.shared.import_utils import safe_import_from


@dataclass(frozen=True)
class _FlashBoundaries:
    """Generic FlashAttention varlen boundaries for packed rows.

    Attributes:
        cu_q: int32 Tensor of shape [segments + 1] holding cumulative query
            lengths over the row-major flattened [batch * sequence] stream.
            Physical padding runs are segments of their own.
        cu_k: int32 Tensor of shape [segments + 1] holding the matching key
            boundaries.
        max_q: Longest query segment, in tokens.
        max_k: Longest key segment, in tokens.
    """

    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int


def _packed_attention(self: nn.Module, hidden: torch.Tensor, metadata: _FlashBoundaries, **kwargs: Any) -> Any:
    """Use the released projections/RoPE with explicit, padding-aware segments.

    As in the release, FP32 query, key and value return to the autocast dtype
    before FlashAttention; without autocast, to the pre-quantization or
    query-projection weight dtype.

    Args:
        hidden: Tensor of shape [batch, sequence, hidden] holding normalized
            packed rows.
        metadata: Segment boundaries over the flattened rows.
        **kwargs: Release attention arguments. ``position_embeddings`` is a
            tuple of cos and sin Tensors of shape [batch, sequence, head_dim];
            without it, ``position_ids`` (Tensor of shape [batch, sequence])
            rebuilds them.

    Returns:
        A tuple of a Tensor of shape [batch, sequence, hidden] holding the
        attention output after ``o_proj``, and two ``None`` placeholders for
        the release's attention weights and cache.

    Raises:
        ValueError: If a KV cache or sink attention is requested.
    """
    if kwargs.get("use_cache") or kwargs.get("past_key_value") is not None:
        raise ValueError("Packed DeciLM attention does not support a KV cache")
    if self.attention_config.is_sink:
        raise ValueError("Packed DeciLM sink attention requires a model-specific mask policy")
    batch, sequence, _ = hidden.shape
    query = self.q_proj(hidden).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
    key = self.k_proj(hidden).view(batch, sequence, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value = self.v_proj(hidden).view(batch, sequence, self.num_key_value_heads, self.head_dim)
    position_embeddings = kwargs.get("position_embeddings")
    if position_embeddings is None:
        position_embeddings = self.rotary_emb(value, kwargs.get("position_ids"))
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    if query.dtype == torch.float32:
        # As in the release's FlashAttention-2 path: RoPE tables built from FP32
        # activations promote Q/K, but FlashAttention needs matching 16-bit inputs.
        if torch.is_autocast_enabled(query.device.type):
            target = torch.get_autocast_dtype(query.device.type)
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target = self.config._pre_quantization_dtype
        else:
            target = self.q_proj.weight.dtype
        query, key, value = query.to(target), key.to(target), value.to(target)
    _, flash = safe_import_from("flash_attn", "flash_attn_varlen_func")
    window = self.sliding_window
    output = flash(
        query.transpose(1, 2).flatten(0, 1),
        key.transpose(1, 2).flatten(0, 1),
        value.flatten(0, 1),
        metadata.cu_q.to(device=query.device, non_blocking=True),
        metadata.cu_k.to(device=query.device, non_blocking=True),
        metadata.max_q,
        metadata.max_k,
        dropout_p=self.attention_dropout if self.training else 0.0,
        causal=True,
        window_size=(-1, -1) if window is None else (window, window),
    )
    output = self.o_proj(output.reshape(batch, sequence, self.num_heads * self.head_dim))
    return output, None, None


def _root_forward(
    self: nn.Module,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.Tensor | None = None,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    padding_mask: torch.Tensor | None = None,
    _packed_seq_ids: torch.Tensor | None = None,
) -> Any:
    """Route generic FlashAttention packing metadata into the released DeciLM forward.

    Args:
        input_ids: Tensor of shape [batch, sequence] holding token IDs. Packed
            rows hold several documents followed by padding.
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask. Packed calls replace it with segment boundaries.
        position_ids: Tensor of shape [batch, sequence] holding positions that
            restart at each packed document.
        past_key_values: Release KV cache; packed calls reject it.
        inputs_embeds: Tensor of shape [batch, sequence, hidden], used instead
            of ``input_ids``.
        labels: Tensor of shape [batch, sequence] for the release's own loss.
        use_cache: Whether to return a cache; packed calls reject True.
        output_attentions: Release flag.
        output_hidden_states: Release flag.
        return_dict: Whether the release returns a ``ModelOutput``.
        cache_position: Tensor of shape [sequence] holding cache positions.
        cu_seq_lens_q: Query boundaries; see ``_FlashBoundaries.cu_q``.
        cu_seq_lens_k: Key boundaries; see ``_FlashBoundaries.cu_k``.
        max_length_q: Longest query segment, in tokens.
        max_length_k: Longest key segment, in tokens.
        padding_mask: bool Tensor of shape [batch, sequence] from the collater;
            ignored.
        _packed_seq_ids: Tensor of shape [batch, sequence] holding document IDs
            from the collater; ignored.

    Returns:
        The release's causal-LM output, whose logits have shape [batch,
        sequence, vocab].

    Raises:
        ValueError: If packing metadata is incomplete or a cache is requested
            with packing.
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
        cache_position=cache_position,
    )
    if any(x is not None for x in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)):
        if cu_seq_lens_q is None or cu_seq_lens_k is None or max_length_q is None or max_length_k is None:
            raise ValueError("DeciLM packing requires both cumulative lengths and maxima")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("Packed DeciLM execution does not support a KV cache")
        kwargs["use_cache"] = False
        kwargs["attention_mask"] = _FlashBoundaries(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    return self._nemo_unpacked_forward(**kwargs)


def _causal_mask(
    self: nn.Module,
    attention_mask: torch.Tensor | _FlashBoundaries | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Keep packed boundaries instead of building the release's causal mask.

    Args:
        attention_mask: Tensor of shape [batch, sequence] holding the release's
            0/1 padding mask, or packed segment boundaries.
        *args: Remaining release mask arguments.
        **kwargs: Remaining release mask arguments.

    Returns:
        The boundaries unchanged on packed calls; otherwise the release's mask.
    """
    if isinstance(attention_mask, _FlashBoundaries):
        return attention_mask
    return self._nemo_unpacked_causal_mask(attention_mask, *args, **kwargs)


def _attention_forward(
    self: nn.Module,
    *args: Any,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    **kwargs: Any,
) -> Any:
    """Dispatch packed calls to varlen attention and others to the released forward.

    Args:
        *args: Positional release arguments; the first is ``hidden_states``.
        cu_seq_lens_q: Query boundaries forwarded by the shared pipeline
            forward; see ``_FlashBoundaries.cu_q``.
        cu_seq_lens_k: Key boundaries; see ``_FlashBoundaries.cu_k``.
        max_length_q: Longest query segment, in tokens.
        max_length_k: Longest key segment, in tokens.
        **kwargs: Release arguments, including ``hidden_states``: Tensor of
            shape [batch, sequence, hidden] and ``attention_mask``: Tensor of
            shape [batch, sequence] or packed segment boundaries.

    Returns:
        The release's attention tuple, whose first element is a Tensor of shape
        [batch, sequence, hidden].

    Raises:
        ValueError: If pipeline metadata is incomplete or a cache is requested
            with packing.
    """
    # PP preserves these tensor kwargs even though the legacy top-level forward
    # has been replaced by the shared pipeline forward.
    if cu_seq_lens_q is not None:
        if cu_seq_lens_k is None or max_length_q is None or max_length_k is None:
            raise ValueError("Packed DeciLM attention requires complete Flash metadata")
        if kwargs.get("use_cache") or kwargs.get("past_key_value") is not None:
            raise ValueError("Packed DeciLM attention does not support a KV cache")
        kwargs["attention_mask"] = _FlashBoundaries(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    metadata = kwargs.get("attention_mask")
    if isinstance(metadata, _FlashBoundaries):
        hidden = args[0] if args else kwargs.pop("hidden_states")
        return _packed_attention(self, hidden, metadata, **kwargs)
    return self._nemo_unpacked_forward(*args, **kwargs)


def _validate_backend(self: nn.Module, attn_implementation: str) -> None:
    if attn_implementation != "flash_attention_2":
        raise ValueError("Packed DeciLM currently requires flash_attention_2")


def apply_packing_adapters(model: nn.Module, *, mesh: MeshContext | None = None) -> nn.Module:
    """Add explicit packed boundaries without changing unpacked/generation calls.

    Args:
        model: Released remote DeciLM causal LM, before FSDP/PP wrapping.
        mesh: Construction context supplied by the runtime-hook dispatcher.

    Returns:
        The same module, with its parameter names and ordinary forwards intact.
    """
    del mesh
    if hasattr(model, "_nemo_unpacked_forward"):
        return model
    model._nemo_unpacked_forward = model.forward
    model.forward = MethodType(_root_forward, model)
    model.validate_packed_attention_backend = MethodType(_validate_backend, model)
    backbone = model.model
    backbone._nemo_unpacked_causal_mask = backbone._update_causal_mask
    backbone._update_causal_mask = MethodType(_causal_mask, backbone)
    for attention in model.modules():
        if type(attention).__name__ != "DeciLMFlashAttention2":
            continue
        attention._nemo_unpacked_forward = attention.forward
        attention.forward = MethodType(_attention_forward, attention)
    return model
