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

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from nemo_automodel.shared.utils import dtype_from_str

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None


@dataclass(kw_only=True)
class BackendConfig:
    attn: Literal["te", "sdpa", "flex"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    enable_deepep: bool = HAVE_DEEP_EP
    fake_balanced_gate: bool = False
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    gate_precision: str | torch.dtype | None = None

    def __post_init__(self):
        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        rms_norm_module = TransformerEngineRMSNorm(normalized_shape=dim, eps=eps, device=device, params_dtype=dtype)
    elif rms_norm_impl == "torch":
        rms_norm_module = nn.RMSNorm(dim, eps=eps, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")
    return rms_norm_module


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        return TransformerEngineLinear(in_features, out_features, bias=bias, device=device, params_dtype=dtype)
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")
