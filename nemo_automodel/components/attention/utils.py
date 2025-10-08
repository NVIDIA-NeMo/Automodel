import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.attention.flex_attention import FlexAttention


def initialize_attn_module_and_func(
    attn_impl: str,
    num_attention_heads: int,
    num_qk_channels: int,
    num_v_channels: int,
    softmax_scale: float,
    attn_mask_type: str = "causal",
    qkv_format: str = "bshd",
    num_gqa_groups: int | None = None,
    **kwargs,
) -> tuple[nn.Module | None, Callable]:
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
        attn_func = functools.partial(
            F.scaled_dot_product_attention,
            scale=softmax_scale,
            is_causal=attn_mask_type == "causal",
            enable_gqa=num_gqa_groups is not None,
            **kwargs,
        )
        return None, attn_func
    elif attn_impl == "flex":
        attn_module = FlexAttention()
        # We still return the module and a reference to its call for parity with other backends
        attn_func = attn_module.__call__
        return attn_module, attn_func
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")


def preprocess_args_and_kwargs_for_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None, attn_impl: str, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Preprocess attention inputs based on backend requirements."""
    # Create attention kwargs based on backend
    if attn_impl == "te":
        attn_kwargs = {}
        if attention_mask is not None:
            padding_mask = attention_mask.logical_not()
            attn_kwargs = {
                "attn_mask_type": "padding_causal",
                "window_size": (-1, 0),
                "attention_mask": padding_mask.unsqueeze(1).unsqueeze(2),
            }
        elif "seq_lens" in kwargs:
            cu_seqlens = torch.cat(
                [torch.tensor([0], device=kwargs["seq_lens"].device), torch.cumsum(kwargs["seq_lens"], dim=0)]
            ).to(dtype=torch.int32, device=q.device)
            attn_kwargs = {
                "qkv_format": "thd",
                "attn_mask_type": "padding_causal",
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_kv": cu_seqlens,
            }
            if "seq_lens_padded" in kwargs:
                cu_seqlens_padded = torch.cat(
                    [
                        torch.tensor([0], device=kwargs["seq_lens_padded"].device),
                        torch.cumsum(kwargs["seq_lens_padded"], dim=0),
                    ]
                ).to(dtype=torch.int32, device=q.device)
                attn_kwargs["cu_seqlens_q_padded"] = cu_seqlens_padded
                attn_kwargs["cu_seqlens_kv_padded"] = cu_seqlens_padded
        elif "cu_seqlens" in kwargs:
            attn_kwargs = {
                "qkv_format": "thd",
                "attn_mask_type": "padding_causal",
                "cu_seqlens_q": kwargs["cu_seqlens"],
                "cu_seqlens_kv": kwargs["cu_seqlens"],
            }
            if "cu_seqlens_padded" in kwargs:
                attn_kwargs["cu_seqlens_q_padded"] = kwargs["cu_seqlens_padded"]
                attn_kwargs["cu_seqlens_kv_padded"] = kwargs["cu_seqlens_padded"]
        elif "cu_seqlens_q" in kwargs and "cu_seqlens_kv" in kwargs:
            attn_kwargs = {
                "qkv_format": "thd",
                "attn_mask_type": "padding_causal",
                "cu_seqlens_q": kwargs["cu_seqlens_q"],
                "cu_seqlens_kv": kwargs["cu_seqlens_kv"],
            }
            if "cu_seqlens_q_padded" in kwargs:
                attn_kwargs["cu_seqlens_q_padded"] = kwargs["cu_seqlens_q_padded"]
            if "cu_seqlens_kv_padded" in kwargs:
                attn_kwargs["cu_seqlens_kv_padded"] = kwargs["cu_seqlens_kv_padded"]
    else:  # sdpa
        attn_kwargs = {}
        # Transpose for SDPA
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        attn_kwargs["is_causal"] = True

    return q, k, v, attn_kwargs


def postprocess_output_for_attn(x: torch.Tensor, attn_impl: str) -> torch.Tensor:
    """Postprocess attention output based on attn_impl requirements."""
    if attn_impl == "sdpa":
        x = x.transpose(1, 2).contiguous()
    return x
