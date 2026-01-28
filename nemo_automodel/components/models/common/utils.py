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
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
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
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize RMSNorm module with the specified backend.
    
    For TE backend, creates TE module directly on meta device (following GroupedExpertsTE pattern).
    Call reset_parameters() to materialize weights.
    
    Args:
        rms_norm_impl: Backend implementation ("te" or "torch")
        dim: Normalized dimension
        eps: Epsilon for numerical stability
        device: Device to create module on (meta for TE to support lazy init)
        dtype: Parameter dtype
        
    Returns:
        RMSNorm module
    """
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm
        
        # Create TE module directly on meta device (same as GroupedExpertsTE)
        return TransformerEngineRMSNorm(
            normalized_shape=dim,
            eps=eps,
            device="meta",
            params_dtype=dtype
        )
    elif rms_norm_impl == "torch":
        return nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize Linear module with the specified backend.
    
    For TE backend, creates TE module directly on meta device (following GroupedExpertsTE pattern).
    Call reset_parameters() to materialize weights.
    
    Args:
        linear_impl: Backend implementation ("te" or "torch")
        in_features: Input features
        out_features: Output features
        bias: Whether to use bias
        device: Device to create module on (meta for TE to support lazy init)
        dtype: Parameter dtype
        
    Returns:
        Linear module
    """
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear
        
        # Create TE module directly on meta device (same as GroupedExpertsTE)
        return TransformerEngineLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device="meta",
            params_dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def initialize_attention_module(
    attn_impl: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    qkv_bias: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize attention qkv projection module with the specified backend.

    Args:
        attn_impl: Backend implementation ("te", "sdpa", or "torch")
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads
        head_dim: Dimension of each attention head
        hidden_size: Hidden dimension of the model
        qkv_bias: Whether to use bias in qkv projection
        device: Device to create module on (None for meta device)
        dtype: Parameter dtype

    Returns:
        Module for qkv projection, depending on the backend:
        - TE: Returns TE Linear module for fused qkv projection
        - SDPA/Torch: Returns regular nn.Linear for separate q,k,v projections
    """
    # Calculate projection dimensions
    num_query_groups = num_heads // num_kv_heads
    qkv_hidden_size = (num_heads + 2 * num_kv_heads) * head_dim

    if attn_impl == "te":
        # Use wrapper for auto-materialization during PP shape inference
        return TELinearWrapper(
            in_features=hidden_size,
            out_features=qkv_hidden_size,
            bias=qkv_bias,
            device=device,
            params_dtype=dtype
        )
    elif attn_impl in ("sdpa", "torch"):
        return nn.Linear(hidden_size, qkv_hidden_size, bias=qkv_bias, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")


def materialize_te_weights(model: nn.Module, device: torch.device | str = "cuda") -> None:
    """Materialize Transformer Engine module weights from meta device.
    
    This function walks through all modules and calls reset_parameters() on any
    Transformer Engine modules that still have weights on meta device.
    This is necessary before using the model with pipeline parallelism, as PP's shape
    inference requires weights to be allocated.
    
    Args:
        model: The model to materialize weights for
        device: Target device to materialize weights on
    """
# #region agent log
    import json, time
    with open('/lustre/fs1/portfolios/coreai/users/zhiyul/Automodel/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"location":"utils.py:materialize_te_weights", "message":"Materializing weights", "data":{"model_type": str(type(model))}, "timestamp":time.time(), "hypothesisId":"C"}) + "\n")
    # #endregion
    with torch.device(device):
        for name, module in model.named_modules():
            # Check if this is a TE module with weights on meta device
            is_te_module = module.__class__.__module__.startswith('transformer_engine')
            
            if is_te_module:
                is_meta = hasattr(module, 'weight') and hasattr(module.weight, 'is_meta') and module.weight.is_meta
                # #region agent log
                with open('/lustre/fs1/portfolios/coreai/users/zhiyul/Automodel/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"location":"utils.py:materialize_te_weights", "message":"Found TE module", "data":{"name": name, "class": str(module.__class__), "is_meta": is_meta}, "timestamp":time.time(), "hypothesisId":"C"}) + "\n")
                # #endregion
                if is_meta and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()


__all__ = [
    "BackendConfig",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "initialize_attention_module",
    "materialize_te_weights",
]
