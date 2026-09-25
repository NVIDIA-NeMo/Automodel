# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MiMo-specific Transformer Engine context-parallel setup."""

from __future__ import annotations

import torch


def _unwrap_checkpoint_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return the attention module beneath any activation-checkpoint wrappers."""
    while hasattr(module, "_checkpoint_wrapped_module"):
        module = module._checkpoint_wrapped_module
    return module


def setup_mimo_te_context_parallel(model: torch.nn.Module, cp_mesh) -> None:
    """Configure every MiMo attention layer through its model-owned CP hook.

    The attention module owns TE backend validation, head-partition validation,
    and the a2a transport choice. This traversal only finds pipeline-local
    attention layers and shares one CUDA communication stream across them.

    Args:
        model: A complete MiMo model or one pipeline-local model part.
        cp_mesh: One-dimensional context-parallel device mesh.
    """
    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    if cp_size <= 1:
        model._mimo_te_cp_configured_group = None
        return

    cp_group = cp_mesh.get_group()
    cp_stream = torch.cuda.Stream()
    for name, wrapped_attention in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        self_attn = _unwrap_checkpoint_module(wrapped_attention)
        setup_cp_attention = getattr(self_attn, "setup_cp_attention", None)
        if not callable(setup_cp_attention):
            raise ValueError(f"MiMo context parallelism requires {name} to expose setup_cp_attention(cp_mesh).")
        setup_cp_attention(cp_mesh, cp_stream=cp_stream)

    model._mimo_te_cp_configured_group = cp_group


def ensure_mimo_te_context_parallel(model: torch.nn.Module, cp_mesh) -> None:
    """Install MiMo's TE a2a transport once for the active CP process group."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return
    if getattr(model, "_mimo_te_cp_configured_group", None) is cp_mesh.get_group():
        return
    setup_mimo_te_context_parallel(model, cp_mesh)
