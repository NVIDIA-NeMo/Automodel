# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(kw_only=True)
class BackendConfig:
    attn: Literal["te", "sdpa"] = "te"
    linear: Literal["torch", "te"] = "torch"
    rms_norm: Literal["torch", "te"] = "te"
    enable_deepep: bool = False
    fake_balanced_gate: bool = False
    enable_hf_state_dict_adapter: bool = False


def initialize_attn_module_and_func(
    attn_impl: str,
    num_attention_heads: int,
    num_qk_channels: int,
    num_v_channels: int,
    softmax_scale: float,
    attn_mask_type: str = "causal",
    qkv_format: str = "bshd",
) -> tuple[nn.Module | None, Callable]:
    if attn_impl == "te":
        from transformer_engine.pytorch.attention import DotProductAttention

        attn_module = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=(num_qk_channels, num_v_channels),
            attn_mask_type=attn_mask_type,
            qkv_format=qkv_format,
            softmax_scale=softmax_scale,
        )
        attn_func = attn_module.__call__
        return attn_module, attn_func
    elif attn_impl == "sdpa":
        attn_func = functools.partial(
            F.scaled_dot_product_attention, scale=softmax_scale, is_causal=attn_mask_type == "causal"
        )
        return None, attn_func
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str = "meta",
) -> nn.Module:
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        rms_norm_module = TransformerEngineRMSNorm(normalized_shape=dim, eps=eps, device=device)
    elif rms_norm_impl == "torch":
        rms_norm_module = nn.RMSNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")
    return rms_norm_module


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str = "meta",
) -> nn.Module:
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        return TransformerEngineLinear(in_features, out_features, bias=bias, device=device)
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")
