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


def process_input_for_thd(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    seq_lens_padded: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """
    Process inputs for THD (total, hidden, depth) format.

    This function converts batched inputs from BSHD format to THD format for packed sequence
    processing. In THD format, the batch dimension is collapsed and all sequences are
    concatenated along the sequence dimension. This supports both token IDs and embeddings
    for pipeline parallelism scenarios.

    Args:
        input_ids: Input tensor of shape [batch_size, seq_len] for token IDs or
            [batch_size, seq_len, hidden_dim] for embeddings (in PP scenarios)
        position_ids: Position IDs tensor of shape [batch_size, seq_len] or None
        seq_lens: Sequence lengths tensor of shape [batch_size, num_packs] containing
            actual sequence lengths (excluding padding/separators). Values of -1000
            indicate padding in the seq_lens dimension.
        seq_lens_padded: Padded sequence lengths tensor of shape [batch_size, num_packs]
            containing lengths including separator tokens. Values of -1000 indicate
            padding in the seq_lens dimension.

    Returns:
        tuple containing:
            - input_ids: Reshaped tensor of shape [total_tokens] for token IDs or
                [total_tokens, hidden_dim] for embeddings
            - position_ids: Reshaped tensor of shape [total_tokens] or None
            - attn_kwargs: Dictionary containing:
                - 'cu_seqlens_q': Cumulative sequence lengths for queries
                - 'cu_seqlens_kv': Cumulative sequence lengths for keys/values
                - 'cu_seqlens_q_padded': Cumulative padded sequence lengths (if seq_lens_padded provided)
                - 'cu_seqlens_kv_padded': Cumulative padded sequence lengths (if seq_lens_padded provided)

    Example:
        >>> batch_size, seq_len = 2, 6
        >>> # 2D Token IDs case - results in [batch*seq, 1] shape
        >>> input_ids = torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]])
        >>> position_ids = torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]])
        >>> seq_lens = torch.tensor([[3, 2], [6, -1000]])  # -1000 is padding
        >>> seq_lens_padded = torch.tensor([[4, 2], [6, -1000]])
        >>>
        >>> input_ids_thd, position_ids_thd, attn_kwargs = process_input_for_thd(
        ...     input_ids, position_ids, seq_lens, seq_lens_padded
        ... )
        >>> # input_ids_thd shape: [12, 1] for 2D input (token IDs)
        >>> # For 3D embeddings [batch, seq, hidden], would be [12, hidden]
        >>> # position_ids_thd shape: [12]
        >>> # cu_seqlens: [0, 3, 5, 11] (cumulative: 0, +3, +2, +6)
        >>> # cu_seqlens_padded: [0, 4, 6, 12] (cumulative: 0, +4, +2, +6)
    """
    # Reshape to THD format: collapse batch dimension
    position_ids_thd = position_ids.reshape(-1) if position_ids is not None else None
    input_ids_thd = input_ids.reshape(position_ids_thd.shape[0], -1).squeeze(-1)

    attn_kwargs = {}

    if seq_lens is not None:
        # Filter out padding values (-1000) and flatten
        # seq_lens shape: [batch_size, num_packs] -> flatten and remove -1000 values
        seq_lens_flat = seq_lens.reshape(-1)
        valid_seq_lens = seq_lens_flat[seq_lens_flat != -1000]

        # Compute cumulative sequence lengths for attention
        cu_seqlens = torch.cat([torch.tensor([0], device=valid_seq_lens.device), torch.cumsum(valid_seq_lens, dim=0)])
        cu_seqlens = cu_seqlens.to(dtype=torch.int32)

        attn_kwargs["cu_seqlens_q"] = cu_seqlens
        attn_kwargs["cu_seqlens_kv"] = cu_seqlens

        if seq_lens_padded is not None:
            # Same processing for padded sequence lengths
            seq_lens_padded_flat = seq_lens_padded.reshape(-1)
            valid_seq_lens_padded = seq_lens_padded_flat[seq_lens_padded_flat != -1000]

            cu_seqlens_padded = torch.cat(
                [torch.tensor([0], device=valid_seq_lens_padded.device), torch.cumsum(valid_seq_lens_padded, dim=0)]
            )
            cu_seqlens_padded = cu_seqlens_padded.to(dtype=torch.int32)

            attn_kwargs["cu_seqlens_q_padded"] = cu_seqlens_padded
            attn_kwargs["cu_seqlens_kv_padded"] = cu_seqlens_padded

    return input_ids_thd, position_ids_thd, attn_kwargs
