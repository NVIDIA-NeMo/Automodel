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

"""CPU coverage for vision-tower activation checkpointing on the expert-parallel MoE path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.moe import parallelizer as moe_parallelizer

_DIM = 8


class _VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(_DIM, 3 * _DIM)
        self.proj = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SDPA over a flattened patch sequence.

        Args:
            x: Patch embeddings of shape ``[S, H]`` (``S`` = patches, ``H`` = hidden).

        Returns:
            Attention output of shape ``[S, H]``.
        """
        q, k, v = (t.unsqueeze(0).unsqueeze(0) for t in self.qkv(x).chunk(3, dim=-1))
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.squeeze(0).squeeze(0))


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _VisionAttention()
        self.mlp = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention + MLP to patch embeddings of shape ``[S, H]``, returning ``[S, H]``."""
        return x + self.mlp(self.attn(x))


class _VisionTower(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(_VisionBlock() for _ in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all vision blocks to patch embeddings of shape ``[S, H]``, returning ``[S, H]``."""
        for block in self.blocks:
            x = block(x)
        return x


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(_DIM, _DIM)
        self.mlp = nn.Linear(_DIM, _DIM)


class _LanguageModel(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(_DecoderLayer() for _ in range(num_layers))


class _InnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.visual = _VisionTower()


class Qwen3VLMoeForConditionalGeneration(nn.Module):
    """Tiny stand-in named after the real EP-MoE VLM so the layer-group mapping applies."""

    def __init__(self):
        super().__init__()
        self.model = _InnerModel()


class _TextOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LanguageModel()


def _assert_submodule_wrapped(visual: nn.Module) -> None:
    """Assert the default spec: blocks stay unwrapped, attention + MLP checkpoint-wrapped."""
    for block in visual.blocks:
        assert not isinstance(block, CheckpointWrapper)
        assert isinstance(block.attn, CheckpointWrapper)
        assert isinstance(block.mlp, CheckpointWrapper)


def test_apply_ac_checkpoints_trainable_vision_submodules_by_default():
    model = Qwen3VLMoeForConditionalGeneration()

    moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2)

    # Default spec matches the generic FSDP2/DDP path: per-submodule wrapping
    # (attention included -- SDPA/flash replay inside checkpoint backward is safe).
    _assert_submodule_wrapped(model.model.visual)
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.model.language_model.layers)

    # The wrapped tower must recompute cleanly: forward + backward produce grads.
    out = model.model.visual(torch.randn(4, _DIM))
    out.sum().backward()
    qkv_weight = model.model.visual.blocks[0].attn._checkpoint_wrapped_module.qkv.weight
    assert qkv_weight.grad is not None


def test_apply_ac_pinned_block_spec_wraps_whole_vision_blocks():
    model = Qwen3VLMoeForConditionalGeneration()

    moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2, vision_checkpoint_spec="pinned_block")

    assert all(isinstance(block, CheckpointWrapper) for block in model.model.visual.blocks)
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.model.language_model.layers)

    out = model.model.visual(torch.randn(4, _DIM))
    out.sum().backward()
    qkv_weight = model.model.visual.blocks[0]._checkpoint_wrapped_module.attn.qkv.weight
    assert qkv_weight.grad is not None


def test_apply_ac_rejects_unknown_vision_checkpoint_spec():
    model = Qwen3VLMoeForConditionalGeneration()

    with pytest.raises(ValueError, match="vision_checkpoint_spec"):
        moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2, vision_checkpoint_spec="mlp_only")


def test_apply_ac_skips_frozen_vision_tower():
    model = Qwen3VLMoeForConditionalGeneration()
    for param in model.model.visual.parameters():
        param.requires_grad_(False)

    moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2)

    for block in model.model.visual.blocks:
        assert not isinstance(block, CheckpointWrapper)
        assert not isinstance(block.attn, CheckpointWrapper)
        assert not isinstance(block.mlp, CheckpointWrapper)
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.model.language_model.layers)


def test_apply_ac_selective_also_checkpoints_vision_submodules():
    model = Qwen3VLMoeForConditionalGeneration()

    moe_parallelizer.apply_ac(model, selective=True)

    _assert_submodule_wrapped(model.model.visual)
    assert all(isinstance(layer, CheckpointWrapper) for layer in model.model.language_model.layers)


def test_apply_ac_flash_attention_vision_tower_also_wraps_attention():
    """FA replay inside checkpoint backward is safe; FA-configured towers get the full default spec."""
    model = Qwen3VLMoeForConditionalGeneration()
    model.config = SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation="flash_attention_2"))

    moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2)

    _assert_submodule_wrapped(model.model.visual)


def test_apply_ac_leaves_text_only_models_unchanged():
    model = _TextOnlyModel()

    moe_parallelizer.apply_ac(model, hidden_size=_DIM, num_experts=2)

    assert all(isinstance(layer, CheckpointWrapper) for layer in model.model.layers)
