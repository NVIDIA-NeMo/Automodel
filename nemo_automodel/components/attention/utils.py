# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.attention.flex_attention import FlexAttention
from nemo_automodel.shared.import_utils import safe_import


def _flatten_packed_sequence_metadata(
    packed_token_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt batch-major dataset metadata to FA4's flat varlen layout."""
    if packed_token_indices.ndim == 1 and cu_seqlens.ndim == 1:
        if (
            cu_seqlens.numel() < 2
            or int(cu_seqlens[0].item()) != 0
            or int(cu_seqlens[-1].item()) != packed_token_indices.numel()
        ):
            raise ValueError("Flat packed FA4 metadata must start at zero and cover every token index")
        return packed_token_indices, cu_seqlens
    if packed_token_indices.shape != (batch_size, sequence_length) or cu_seqlens.ndim != 2:
        raise ValueError(
            "Packed FA4 metadata does not match the current [batch, sequence] layout: "
            f"indices={tuple(packed_token_indices.shape)}, cu_seqlens={tuple(cu_seqlens.shape)}, "
            f"batch={batch_size}, sequence={sequence_length}."
        )

    valid = packed_token_indices >= 0
    row_offsets = torch.arange(batch_size, device=packed_token_indices.device)[:, None] * sequence_length
    flat_indices = (packed_token_indices + row_offsets)[valid].to(torch.long)
    lengths: list[torch.Tensor] = []
    for row_idx, row in enumerate(cu_seqlens):
        boundaries = row[row >= 0]
        if boundaries.numel() and (
            int(boundaries[0].item()) != 0
            or int(boundaries[-1].item()) != int(valid[row_idx].sum().item())
            or bool((boundaries[1:] < boundaries[:-1]).any().item())
        ):
            raise ValueError("Each packed FA4 metadata row must start at zero and cover its valid tokens")
        if boundaries.numel() > 1:
            lengths.append(boundaries[1:] - boundaries[:-1])
    if not lengths:
        raise ValueError("Packed FA4 metadata must describe at least one document")
    document_lengths = torch.cat(lengths)
    flat_cu_seqlens = F.pad(torch.cumsum(document_lengths, dim=0, dtype=cu_seqlens.dtype), (1, 0))
    if int(flat_cu_seqlens[-1].item()) != flat_indices.numel():
        raise ValueError("Packed FA4 token indices and cumulative lengths describe different token counts")
    return flat_indices, flat_cu_seqlens


def initialize_attn_module_and_func(
    attn_impl: str,
    num_attention_heads: int,
    num_qk_channels: int,
    num_v_channels: int,
    softmax_scale: float,
    attn_mask_type: str = "causal",
    qkv_format: str = "bshd",
    num_gqa_groups: int | None = None,
    **kwargs: Any,
) -> tuple[nn.Module | None, Callable[..., torch.Tensor]]:
    """Initialize an attention backend module and callable."""
    if attn_impl == "te":
        from transformer_engine.pytorch.attention import DotProductAttention

        attn_module = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=(num_qk_channels, num_v_channels),
            attn_mask_type=attn_mask_type,
            qkv_format=qkv_format,
            softmax_scale=softmax_scale,
            num_gqa_groups=num_gqa_groups,
            **kwargs,
        )
        attn_func = attn_module.__call__
        return attn_module, attn_func
    elif attn_impl == "sdpa":
        supported_sdpa_kwargs = {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
        unexpected_kwargs = kwargs.keys() - supported_sdpa_kwargs
        if unexpected_kwargs:
            raise TypeError(f"Unsupported SDPA attention kwargs: {sorted(unexpected_kwargs)}")

        default_attn_mask = cast(torch.Tensor | None, kwargs.get("attn_mask", None))
        default_dropout_p = cast(float, kwargs.get("dropout_p", 0.0))
        default_is_causal = cast(bool, kwargs.get("is_causal", attn_mask_type == "causal"))
        default_scale = cast(float | None, kwargs.get("scale", softmax_scale))
        default_enable_gqa = cast(bool, kwargs.get("enable_gqa", num_gqa_groups is not None))

        def attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **call_kwargs: Any) -> torch.Tensor:
            unexpected_call_kwargs = call_kwargs.keys() - supported_sdpa_kwargs
            if unexpected_call_kwargs:
                raise TypeError(f"Unsupported SDPA attention kwargs: {sorted(unexpected_call_kwargs)}")

            attn_mask = cast(torch.Tensor | None, call_kwargs.get("attn_mask", default_attn_mask))
            dropout_p = cast(float, call_kwargs.get("dropout_p", default_dropout_p))
            is_causal = cast(bool, call_kwargs.get("is_causal", default_is_causal))
            scale = cast(float | None, call_kwargs.get("scale", default_scale))
            enable_gqa = cast(bool, call_kwargs.get("enable_gqa", default_enable_gqa))
            if enable_gqa and attn_mask is not None:
                groups = q.shape[-3] // k.shape[-3]
                k = k.repeat_interleave(groups, dim=-3)
                v = v.repeat_interleave(groups, dim=-3)
                enable_gqa = False
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return None, attn_func
    elif attn_impl == "fa4":
        # FlashAttention-4 (CuTe). Consumes the native [b, s, nh, hd] (bshd) / [t, nh, hd]
        # (thd) layout directly, like TE -- no transpose on the way in or out. FA4 has no
        # dense-mask entry point by design: `causal` plus varlen `cu_seqlens` are its only
        # mask forms, which is what makes it fast. preprocess_args_and_kwargs_for_attn
        # rejects an explicit mask rather than silently materializing one.
        try:
            have_fa4, flash_attn_cute = safe_import("flash_attn.cute")
        except Exception as exc:
            # Deliberately broad: flash_attn.cute imports cutlass, which probes the CUDA
            # toolchain at module scope and raises FileNotFoundError (no nvcc) or a cutlass
            # RuntimeError rather than ImportError when the install is incomplete. `from exc`
            # keeps the original traceback.
            raise ImportError(
                "attn_impl='fa4' requires a working FlashAttention-4 (flash_attn.cute) "
                f"install; importing it failed with {type(exc).__name__}: {exc}. Build the "
                "container with INSTALL_FA4=true (docker/Dockerfile)."
            ) from exc
        if not have_fa4:
            raise ImportError(
                "attn_impl='fa4' requires FlashAttention-4 (flash_attn.cute); "
                "build the container with INSTALL_FA4=true (docker/Dockerfile)."
            )
        flash_attn_func = flash_attn_cute.flash_attn_func
        flash_attn_varlen_func = flash_attn_cute.flash_attn_varlen_func

        if qkv_format not in ("bshd", "thd"):
            raise ValueError(f"attn_impl='fa4' supports qkv_format 'bshd' or 'thd', got {qkv_format!r}")

        supported_fa4_kwargs = {
            "causal",
            "window_size",
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "max_seqlen_q",
            "max_seqlen_kv",
            "softcap",
            "learnable_sink",
            "packed_token_indices",
            "_fa4_padded_output_shape",
        }
        default_causal = attn_mask_type == "causal"

        def attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **call_kwargs: Any) -> torch.Tensor:
            """Run dense or varlen FA4 and restore a padded BSHD result when requested.

            Args:
                q: Query tensor of shape [batch, sequence, heads, head_dim] for
                    dense attention or [tokens, heads, head_dim] for varlen attention.
                k: Key tensor of shape [batch, sequence, kv_heads, head_dim] for
                    dense attention or [tokens, kv_heads, head_dim] for varlen attention.
                v: Value tensor with the same layout as ``k``.
                **call_kwargs: FA4 options and optional packed-sequence metadata.
                    ``packed_token_indices`` has shape [tokens] and
                    ``_fa4_padded_output_shape`` is [batch, sequence, heads, head_dim].

            Returns:
                Attention output matching the input query layout, or a restored
                tensor of shape [batch, sequence, heads, head_dim] when packed
                inputs were unpadded before the kernel call.
            """
            unexpected_call_kwargs = call_kwargs.keys() - supported_fa4_kwargs
            if unexpected_call_kwargs:
                raise TypeError(f"Unsupported FA4 attention kwargs: {sorted(unexpected_call_kwargs)}")

            common: dict[str, Any] = {
                "softmax_scale": softmax_scale,
                "causal": cast(bool, call_kwargs.get("causal", default_causal)),
                "window_size": call_kwargs.get("window_size", (None, None)),
            }
            for opt in ("softcap", "learnable_sink"):
                if call_kwargs.get(opt) is not None:
                    common[opt] = call_kwargs[opt]

            cu_seqlens_q = call_kwargs.get("cu_seqlens_q")
            if cu_seqlens_q is None:
                return flash_attn_func(q, k, v, **common)

            cu_seqlens_kv = call_kwargs.get("cu_seqlens_kv", cu_seqlens_q)
            max_seqlen_q = call_kwargs.get("max_seqlen_q")
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=call_kwargs.get("max_seqlen_kv", max_seqlen_q),
                **common,
            )
            padded_output_shape = call_kwargs.get("_fa4_padded_output_shape")
            if padded_output_shape is None:
                return output

            unpad_indices = call_kwargs.get("packed_token_indices")
            if unpad_indices is None:
                return output.reshape(padded_output_shape)
            padded_output = output.new_zeros((padded_output_shape[0] * padded_output_shape[1], *output.shape[1:]))
            padded_output = padded_output.index_copy(0, unpad_indices, output)
            return padded_output.reshape(padded_output_shape)

        return None, attn_func
    elif attn_impl == "flex":
        attn_module = FlexAttention()
        # We still return the module and a reference to its call for parity with other backends
        attn_func = attn_module.__call__
        return attn_module, attn_func
    elif attn_impl == "magi":
        # MagiAttention (Flex-Flash-Attention / context-parallel). The FFA kernel
        # requires q/k/v to share a single head_dim <= 128, so MLA-style attention
        # (e.g. DeepSeek-V3, where num_qk_channels != num_v_channels) is unsupported.
        if num_qk_channels != num_v_channels:
            raise ValueError(
                f"attn_impl='magi' requires equal q/k and v head_dim, got "
                f"num_qk_channels={num_qk_channels} != num_v_channels={num_v_channels}. "
                "MLA-style attention (e.g. DeepSeek-V3 / Moonlight) is not supported by MagiAttention."
            )
        if num_qk_channels > 128:
            raise ValueError(
                f"attn_impl='magi' supports head_dim <= 128, got {num_qk_channels} "
                "(e.g. Gemma3 / Qwen3.5 full-attention layers use 256)."
            )
        # requires magi_attention; the guards above are exercised on CPU but the
        # kernel build is not, so exclude it from coverage.
        from nemo_automodel.components.distributed.context_parallel.magi import (  # pragma: no cover - requires magi_attention
            make_magi_attn_func,
        )

        attn_func = make_magi_attn_func(softmax_scale=softmax_scale)  # pragma: no cover - requires magi_attention
        return None, attn_func  # pragma: no cover - requires magi_attention
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")


def preprocess_args_and_kwargs_for_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    attn_impl: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Preprocess attention inputs based on backend requirements.

    Args:
        q: Query tensor of shape [batch, sequence, heads, head_dim] or
            [tokens, heads, head_dim] for THD input.
        k: Key tensor of shape [batch, sequence, kv_heads, head_dim] or
            [tokens, kv_heads, head_dim] for THD input.
        v: Value tensor with the same layout as ``k``.
        attention_mask: Optional tensor of shape [batch, sequence] for padding
            or indexed packing, or [batch, 1, sequence, sequence] for an
            explicit dense mask.
        attn_impl: Attention backend name.
        **kwargs: Backend metadata. Packed FA4 accepts ``cu_seqlens`` of shape
            [documents + 1] and ``packed_token_indices`` of shape [tokens].

    Returns:
        Query, key, and value tensors in the backend layout plus its keyword
        arguments. Packed BSHD FA4 tensors are unpadded to [tokens, heads,
        head_dim]; the FA4 callable restores its output to BSHD.
    """
    attn_kwargs: dict[str, Any]
    # Create attention kwargs based on backend
    if attn_impl == "te":
        attn_kwargs = {
            "window_size": kwargs.get("window_size", (-1, 0)),
        }
        if attention_mask is not None:
            padding_mask = attention_mask.logical_not()
            attn_kwargs.update(
                {
                    "attn_mask_type": "padding_causal",
                    "attention_mask": padding_mask.unsqueeze(1).unsqueeze(2),
                }
            )
        elif "cu_seqlens" in kwargs:
            attn_kwargs.update(
                {
                    "qkv_format": "thd",
                    "attn_mask_type": "padding_causal",
                    "cu_seqlens_q": kwargs["cu_seqlens"],
                    "cu_seqlens_kv": kwargs["cu_seqlens"],
                }
            )
            if "cu_seqlens_padded" in kwargs:
                attn_kwargs.update(
                    {
                        "cu_seqlens_q_padded": kwargs["cu_seqlens_padded"],
                        "cu_seqlens_kv_padded": kwargs["cu_seqlens_padded"],
                        "pad_between_seqs": True,
                    }
                )
            if "max_seqlen" in kwargs:
                attn_kwargs.update(
                    {
                        "max_seqlen_q": kwargs["max_seqlen"],
                        "max_seqlen_kv": kwargs["max_seqlen"],
                    }
                )
        elif "cu_seqlens_q" in kwargs and "cu_seqlens_kv" in kwargs:
            attn_kwargs.update(
                {
                    "qkv_format": "thd",
                    "attn_mask_type": "padding_causal",
                    "cu_seqlens_q": kwargs["cu_seqlens_q"],
                    "cu_seqlens_kv": kwargs["cu_seqlens_kv"],
                }
            )
            if "cu_seqlens_q_padded" in kwargs:
                attn_kwargs.update(
                    {
                        "cu_seqlens_q_padded": kwargs["cu_seqlens_q_padded"],
                        "pad_between_seqs": True,
                    }
                )
            if "cu_seqlens_kv_padded" in kwargs:
                attn_kwargs["cu_seqlens_kv_padded"] = kwargs["cu_seqlens_kv_padded"]
            if "max_seqlen_q" in kwargs:
                attn_kwargs["max_seqlen_q"] = kwargs["max_seqlen_q"]
            if "max_seqlen_kv" in kwargs:
                attn_kwargs["max_seqlen_kv"] = kwargs["max_seqlen_kv"]

    elif attn_impl == "fa4":
        # FA4 consumes the native [b, s, nh, hd] / [t, nh, hd] layout -- no transpose.
        # Window convention differs from the rest of the codebase: here (-1, 0) means
        # "causal, unbounded left context", whereas FA4 spells unbounded as None and
        # derives `local` from a non-None left window (_resolve_causal_local_window).
        attn_kwargs = {"causal": True}
        window_size = kwargs.get("window_size", (-1, 0))
        left_window, right_window = window_size if isinstance(window_size, tuple) else (window_size, 0)
        attn_kwargs["window_size"] = (
            None if left_window is None or left_window < 0 else left_window,
            None if right_window is None or right_window <= 0 else right_window,
        )
        for opt in ("softcap", "learnable_sink"):
            if kwargs.get(opt) is not None:
                attn_kwargs[opt] = kwargs[opt]

        cu_seqlens = kwargs.get("cu_seqlens")
        unpad_indices = kwargs.get("packed_token_indices")
        max_seqlen = kwargs.get("max_seqlen")
        if cu_seqlens is None and attention_mask is not None and attention_mask.ndim == 2:
            if attention_mask.numel() == 0:
                raise ValueError("FA4 attention_mask must contain at least one valid token")
            if attention_mask.dtype != torch.bool and int(attention_mask.max().item()) > 1:
                raise ValueError(
                    "Packed FA4 inputs require dataset-provided cu_seqlens, max_seqlen, and packed_token_indices."
                )
            valid_tokens = attention_mask.bool()
            seqlens = valid_tokens.sum(dim=-1)
            seqlens = seqlens[seqlens > 0]
            if seqlens.numel() == 0:
                raise ValueError("FA4 attention_mask must contain at least one valid token")
            unpad_indices = torch.nonzero(valid_tokens.flatten(), as_tuple=False).flatten()
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            max_seqlen = int(seqlens.max().item())

        if cu_seqlens is not None and unpad_indices is not None and unpad_indices.ndim == 2:
            if q.ndim != 4:
                raise ValueError("Batch-major packed FA4 metadata requires BSHD query/key/value tensors")
            unpad_indices, cu_seqlens = _flatten_packed_sequence_metadata(
                unpad_indices,
                cu_seqlens,
                batch_size=q.shape[0],
                sequence_length=q.shape[1],
            )

        if cu_seqlens is not None:
            if max_seqlen is None:
                raise ValueError("FA4 cu_seqlens requires max_seqlen")
            attn_kwargs["cu_seqlens_q"] = cu_seqlens
            attn_kwargs["cu_seqlens_kv"] = cu_seqlens
            attn_kwargs["max_seqlen_q"] = max_seqlen
            attn_kwargs["max_seqlen_kv"] = max_seqlen

            if q.ndim == 4:
                padded_output_shape = tuple(q.shape)
                flat_q = q.reshape(-1, *q.shape[2:])
                flat_k = k.reshape(-1, *k.shape[2:])
                flat_v = v.reshape(-1, *v.shape[2:])
                if unpad_indices is None:
                    if int(cu_seqlens[-1].item()) != flat_q.shape[0]:
                        raise ValueError(
                            "Packed BSHD FA4 inputs require packed_token_indices when cu_seqlens "
                            "does not cover every padded token."
                        )
                    q, k, v = flat_q, flat_k, flat_v
                else:
                    unpad_indices = unpad_indices.to(device=q.device, dtype=torch.long)
                    if int(cu_seqlens[-1].item()) != unpad_indices.numel():
                        raise ValueError(
                            "FA4 cu_seqlens and packed_token_indices disagree: "
                            f"{int(cu_seqlens[-1].item())} tokens vs {unpad_indices.numel()} indices."
                        )
                    q = flat_q.index_select(0, unpad_indices)
                    k = flat_k.index_select(0, unpad_indices)
                    v = flat_v.index_select(0, unpad_indices)
                    attn_kwargs["packed_token_indices"] = unpad_indices
                attn_kwargs["_fa4_padded_output_shape"] = padded_output_shape
        elif "cu_seqlens_q" in kwargs and "cu_seqlens_kv" in kwargs:
            attn_kwargs["cu_seqlens_q"] = kwargs["cu_seqlens_q"]
            attn_kwargs["cu_seqlens_kv"] = kwargs["cu_seqlens_kv"]
            if "max_seqlen_q" in kwargs:
                attn_kwargs["max_seqlen_q"] = kwargs["max_seqlen_q"]
            if "max_seqlen_kv" in kwargs:
                attn_kwargs["max_seqlen_kv"] = kwargs["max_seqlen_kv"]
        elif attention_mask is not None:
            # Anything left is either a padding mask (needs unpadding to varlen) or a dense
            # block-causal mask. Silently dropping it would attend across documents/padding;
            # materializing it is exactly the SDPA slow path FA4 exists to avoid.
            raise ValueError(
                "attn_impl='fa4' cannot consume an explicit attention_mask "
                f"(got shape {tuple(attention_mask.shape)}). Pass packed sequences so the "
                "model supplies cu_seqlens (varlen), or use attn='te'/'sdpa' for masked batches."
            )

    elif attn_impl == "flex":
        attn_kwargs = kwargs
        # Transpose for SDPA
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif attn_impl == "magi":  # pragma: no cover - requires magi_attention
        # magi's attn_func consumes the native [b, s, nh, hd] / [t, nh, hd] layout
        # directly (no transpose). Forward the genuine mask metadata so the FFA key
        # matches what the other backends would build: an explicit ``magi_attn_spec``
        # (arbitrary AttnSlice mask, e.g. a prefix tree) takes priority, else
        # ``cu_seqlens`` selects a varlen/block-diagonal (packed) mask; absence of
        # both means a single causal sequence.
        attn_kwargs = {}
        for _k in ("magi_attn_spec", "cu_seqlens", "cu_seqlens_q", "window_size"):
            if kwargs.get(_k) is not None:
                attn_kwargs[_k] = kwargs[_k]
    else:  # sdpa
        attn_kwargs = {}
        # Transpose for SDPA
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        window_size = kwargs.get("window_size", (-1, 0))
        left_window, right_window = window_size if isinstance(window_size, tuple) else (window_size, 0)
        has_local_window = (left_window is not None and left_window >= 0) or (
            right_window is not None and right_window > 0
        )
        key_mask = None
        explicit_mask = None
        if attention_mask is not None:
            if attention_mask.dim() <= 2:
                key_mask = attention_mask.to(device=q.device, dtype=torch.bool)
                has_padding_mask = not bool(key_mask.all().item())
            else:
                explicit_mask = attention_mask.to(device=q.device)
                has_padding_mask = False
        else:
            has_padding_mask = False

        if has_local_window or has_padding_mask:
            q_len = q.shape[-2]
            kv_len = k.shape[-2]
            kv_offset = max(kv_len - q_len, 0)
            q_pos = torch.arange(q_len, device=q.device) + kv_offset
            kv_pos = torch.arange(kv_len, device=q.device)
            causal_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

            if left_window is not None and left_window >= 0:
                causal_mask = causal_mask & (kv_pos.unsqueeze(0) > q_pos.unsqueeze(1) - left_window)
            if right_window is not None and right_window > 0:
                causal_mask = causal_mask & (kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + right_window)

            if has_padding_mask:
                assert key_mask is not None
                if key_mask.shape[-1] != kv_len:
                    key_mask = key_mask[..., -kv_len:]
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) & key_mask[:, None, None, :]

            attn_kwargs["attn_mask"] = causal_mask
            attn_kwargs["is_causal"] = False
        elif explicit_mask is not None:
            attn_kwargs["attn_mask"] = explicit_mask
            attn_kwargs["is_causal"] = False
        else:
            attn_kwargs["is_causal"] = True

    return q, k, v, attn_kwargs


def postprocess_output_for_attn(x: torch.Tensor, attn_impl: str) -> torch.Tensor:
    """Postprocess attention output based on attn_impl requirements."""
    if attn_impl in ("sdpa", "flex"):
        x = x.transpose(1, 2).contiguous()
    return x
