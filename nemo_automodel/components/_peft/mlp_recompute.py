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

"""MLP-scoped activation recompute for SwiGLU MLPs.

A standard LoRA SwiGLU MLP holds three ``(tokens, intermediate)`` activations for
backward: ``gate`` and ``up`` (saved by liger's ``LigerSiLUMulFunction``) and the
SiLU-Mul output ``h`` (saved by the ``down_proj`` LoRA). At Llama-3.1-8B scale that
is ~3 x 112 MiB per layer, accumulating with depth.

:func:`apply_mlp_activation_recompute` wraps each SwiGLU MLP's ``forward`` in
non-reentrant activation checkpointing, so only the block input ``x``
``(tokens, hidden)`` is saved and the projections + SiLU-Mul are recomputed in
backward. Because the recompute replays the real ``LinearLoRA``/liger forward, all
gradients -- LoRA ``A``/``B`` factors and FSDP2/DTensor-sharded base weights -- are
produced by the existing autograd machinery; there is no hand-written backward and
no DTensor-placement reasoning. This is the cross-cutting alternative to a custom
fused MLP Function and works under FSDP2 today.
"""

import logging

import torch
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

_SWIGLU_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj")


def _make_recompute_forward(orig_forward):
    """Wrap ``orig_forward`` (a bound MLP forward) so it is recomputed in backward.

    ``orig_forward`` is already bound to its module, so ``self`` is not re-passed.
    """

    def forward(x, *args, **kwargs):
        return checkpoint.checkpoint(orig_forward, x, *args, use_reentrant=False, **kwargs)

    return forward


def apply_mlp_activation_recompute(model: torch.nn.Module) -> int:
    """Wrap each SwiGLU MLP's ``forward`` in non-reentrant activation checkpointing.

    A module is treated as a SwiGLU MLP if it exposes ``gate_proj``, ``up_proj``, and
    ``down_proj`` (matches HF ``LlamaMLP``/``Qwen2MLP``/``MistralMLP`` and liger's
    ``LigerSwiGLUMLP``). The projections and SiLU-Mul are recomputed in backward
    instead of saving the ``(tokens, intermediate)`` activations; only the block
    input is saved.

    Call this after the model is built (PEFT, liger, and sharding applied). It is
    agnostic to dropout, DoRA, quantization, and DTensor sharding because it replays
    the real forward rather than reimplementing it.

    Args:
        model: The model (or pipeline stage) to patch in place.

    Returns:
        The number of MLP modules that were wrapped.
    """
    count = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in _SWIGLU_MLP_ATTRS):
            continue
        # Capture the current (possibly liger-bound) forward before replacing it.
        module.forward = _make_recompute_forward(module.forward)
        count += 1
    logger.info("Wrapped %d SwiGLU MLP module(s) with activation recompute", count)
    return count
