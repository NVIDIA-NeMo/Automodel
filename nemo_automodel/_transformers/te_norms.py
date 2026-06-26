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

"""Transformer Engine norm replacement helpers."""

from __future__ import annotations

import torch


def _is_llama_rms_norm(module: torch.nn.Module) -> bool:
    """Return whether *module* is a HuggingFace Llama RMSNorm-style module."""
    cls = module.__class__
    if cls.__name__ != "LlamaRMSNorm":
        return False
    return hasattr(module, "weight") and (hasattr(module, "variance_epsilon") or hasattr(module, "eps"))


def replace_norms_with_te(module: torch.nn.Module) -> tuple[int, int]:
    """Replace supported PyTorch/HF norm modules with Transformer Engine norms.

    Replaces standard 1-D ``torch.nn.LayerNorm`` modules and HuggingFace
    ``LlamaRMSNorm``-style modules. We intentionally do not replace arbitrary
    custom normalization modules, because equivalent eps/weight semantics are
    model-specific.

    Args:
        module: Module tree to mutate in place.

    Returns:
        Tuple of ``(layer_norm_count, rms_norm_count)``.
    """
    from transformer_engine.pytorch import LayerNorm as TELayerNorm
    from transformer_engine.pytorch import RMSNorm as TERMSNorm

    layer_norms = 0
    rms_norms = 0
    for child_name, child in list(module.named_children()):
        replacement = None
        if (
            isinstance(child, torch.nn.LayerNorm)
            and len(child.normalized_shape) == 1
            and child.weight is not None
            and child.bias is not None
        ):
            dtype = child.weight.dtype
            replacement = TELayerNorm(
                hidden_size=child.normalized_shape[0],
                eps=child.eps,
                device=child.weight.device,
                params_dtype=dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            layer_norms += 1
        elif _is_llama_rms_norm(child):
            eps = child.variance_epsilon if hasattr(child, "variance_epsilon") else child.eps
            replacement = TERMSNorm(
                hidden_size=child.weight.shape[0],
                eps=eps,
                device=child.weight.device,
                params_dtype=child.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
            rms_norms += 1

        if replacement is None:
            child_layer_norms, child_rms_norms = replace_norms_with_te(child)
            layer_norms += child_layer_norms
            rms_norms += child_rms_norms
        else:
            setattr(module, child_name, replacement)

    return layer_norms, rms_norms
