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

"""LoRA target-module matching for the Wan-Animate-2 transformer layout.

Wan-Animate-2 keeps the original Wan 2.1 research naming: the self-attention
projections are ``q``, ``k`` and ``v`` on a ``self_attn`` submodule nested
under ``blocks.<n>.block``. The Diffusers-style ``*.to_q`` / ``*.to_v`` patterns
used by the other Wan recipes therefore match nothing here, which would silently
train zero adapters.

The module tree below mirrors ``WanAnimate2Transformer3DModel`` only where the
patterns can be fooled: its ``blocks`` are ``IncontextAttentionBlock``s that wrap
an ``AttentionBlock`` in ``.block``, and ``CrossAttention`` subclasses
``SelfAttention``, so cross attention exposes the same ``q``/``k``/``v``/``o``
leaf names and must stay unpatched. No forward pass is run: the upstream
attention kernels hard-assert CUDA.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.config.loader import load_yaml_config

# LoRA target patterns for Wan-Animate-2 SFT recipes.
WAN_ANIMATE2_TARGET_MODULES = ["*.self_attn.q", "*.self_attn.k", "*.self_attn.v"]
# Patterns used by the Diffusers-style Wan recipes.
DIFFUSERS_WAN_TARGET_MODULES = ["*.to_q", "*.to_v"]

REPO_ROOT = Path(__file__).resolve().parents[4]
LORA_RECIPE_PATH = REPO_ROOT / "examples/diffusion/finetune/wan_animate2_flow_lora.yaml"

NUM_BLOCKS = 2
DIM = 8
# Linear modules that must stay unpatched: cross attention repeats the
# self-attention leaf names, and the feed-forward and head projections are the
# usual over-matching targets.
DECOYS = {"blocks.0.block.cross_attn.q", "blocks.0.block.cross_attn.k_img", "blocks.0.block.ffn", "head"}


class _SelfAttention(nn.Module):
    """Self-attention leaf names of the upstream ``SelfAttention`` module."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _CrossAttention(_SelfAttention):
    """Upstream ``CrossAttention`` subclasses ``SelfAttention`` and adds image projections."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)


class _AttentionBlock(nn.Module):
    """Inner block held by every ``IncontextAttentionBlock`` as ``.block``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttention(dim)
        self.cross_attn = _CrossAttention(dim)
        self.ffn = nn.Linear(dim, dim)


class _IncontextAttentionBlock(nn.Module):
    """Outer block that owns the driving-branch cache and wraps ``_AttentionBlock``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = _AttentionBlock(dim)


class _FakeWanAnimate2Transformer(nn.Module):
    """Structural stand-in for ``WanAnimate2Transformer3DModel``."""

    def __init__(self, *, num_blocks: int = NUM_BLOCKS, dim: int = DIM) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_IncontextAttentionBlock(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, dim)


def _expected_self_attention_projections() -> set[str]:
    """Return the module names LoRA must patch on the fake transformer.

    The output projection is absent by design; it is a decoy that must stay
    unpatched, matching the reference recipe.
    """
    return {
        f"blocks.{index}.block.self_attn.{projection}" for index in range(NUM_BLOCKS) for projection in ("q", "k", "v")
    }


def _apply_lora(model: nn.Module, target_modules: list[str]) -> tuple[int, set[str]]:
    """Apply LoRA to ``model`` in place with the given target patterns.

    Args:
        model: Module to patch.
        target_modules: LoRA target-module patterns.

    Returns:
        Tuple of the count reported by ``apply_lora_to_linear_modules`` and the
        dotted names of the modules it replaced with ``LinearLoRA``.
    """
    count = apply_lora_to_linear_modules(model, PeftConfig(target_modules=list(target_modules), dim=4, alpha=8))
    return count, {name for name, module in model.named_modules() if isinstance(module, LinearLoRA)}


def test_raw_self_attention_patterns_patch_every_block_projection() -> None:
    """The raw q/k/v patterns train adapters on exactly the self-attention input projections.

    The output projection is deliberately excluded: the reference training recipe
    adapts only q, k and v.
    """
    model = _FakeWanAnimate2Transformer()
    linear_names = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    expected = _expected_self_attention_projections()
    # The decoys that must stay unpatched are really present in the tree.
    assert DECOYS <= linear_names

    count, patched = _apply_lora(model, WAN_ANIMATE2_TARGET_MODULES)

    assert patched == expected
    assert count == 3 * NUM_BLOCKS
    assert patched.isdisjoint(DECOYS)
    trainable_owners = {name.rsplit(".", 2)[0] for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_owners == expected


def test_diffusers_wan_patterns_patch_no_module() -> None:
    """``*.to_q`` / ``*.to_v`` select nothing on the Wan-Animate-2 naming."""
    model = _FakeWanAnimate2Transformer()

    count, patched = _apply_lora(model, DIFFUSERS_WAN_TARGET_MODULES)

    assert (count, patched) == (0, set())
    assert not any(param.requires_grad for param in model.parameters())


def test_lora_recipe_ships_the_raw_self_attention_patterns() -> None:
    """The shipped LoRA recipe targets the raw names, so the patterns cannot drift apart."""
    config = load_yaml_config(LORA_RECIPE_PATH)

    assert config.peft.target_modules == WAN_ANIMATE2_TARGET_MODULES
