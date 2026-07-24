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

"""Tests for the model-owned Qwen image-edit distributed strategy."""

from __future__ import annotations

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed.parallelizer import (
    PARALLELIZATION_STRATEGIES,
    DefaultParallelizationStrategy,
)
from nemo_automodel.components.models.qwen_image_edit.fsdp import (
    QwenImageEditParallelizationStrategy,
    _apply_block_activation_checkpointing,
    _validate_transformer_blocks,
    register_qwen_image_edit_parallel_strategy,
)


def _tiny_transformer():
    """Build a one-block upstream Qwen transformer without downloading weights."""
    diffusers = pytest.importorskip("diffusers")
    return diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )


def test_whole_block_checkpointing_preserves_canonical_state_dict() -> None:
    """Keep upstream Diffusers keys and every dual-stream branch parameter."""
    torch.manual_seed(9)
    model = _tiny_transformer()
    expected_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    _apply_block_activation_checkpointing(model)
    _apply_block_activation_checkpointing(model)

    assert isinstance(model.transformer_blocks[0], CheckpointWrapper)
    actual_state = model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected)

    expected_branch_parameters = {
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.add_q_proj.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
    }
    assert expected_branch_parameters <= set(actual_state)


def test_strategy_checkpoints_blocks_before_standard_fsdp_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover complete Qwen blocks before delegating to repository FSDP2."""
    model = _tiny_transformer()
    delegated: dict[str, object] = {}

    def fake_parallelize(self, **kwargs):
        """Capture the model handed to the standard distributed strategy."""
        delegated.update(kwargs)
        return kwargs["model"]

    monkeypatch.setattr(DefaultParallelizationStrategy, "parallelize", fake_parallelize)
    result = QwenImageEditParallelizationStrategy().parallelize(
        model=model,
        device_mesh=object(),
        activation_checkpointing=True,
    )

    assert result is model
    assert delegated["model"] is model
    assert delegated["activation_checkpointing"] is False
    wrapped_block = model.transformer_blocks[0]
    assert isinstance(wrapped_block, CheckpointWrapper)
    inner_block = wrapped_block._checkpoint_wrapped_module
    assert isinstance(inner_block.attn, torch.nn.Module)
    assert isinstance(inner_block.img_mlp, torch.nn.Module)
    assert isinstance(inner_block.txt_mlp, torch.nn.Module)


def test_registration_is_idempotent_and_model_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install exactly one strategy under the upstream transformer class name."""
    key = "QwenImageTransformer2DModel"
    monkeypatch.delitem(PARALLELIZATION_STRATEGIES, key, raising=False)

    register_qwen_image_edit_parallel_strategy()
    first = PARALLELIZATION_STRATEGIES[key]
    register_qwen_image_edit_parallel_strategy()

    assert PARALLELIZATION_STRATEGIES[key] is first
    assert isinstance(first, QwenImageEditParallelizationStrategy)


def test_strategy_rejects_blocks_missing_text_branch() -> None:
    """Prevent silent omission of Qwen text-MLP parameters from sharding."""

    class IncompleteBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Linear(2, 2)
            self.img_mlp = torch.nn.Linear(2, 2)

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([IncompleteBlock()])

    with pytest.raises(TypeError, match="txt_mlp"):
        _validate_transformer_blocks(model)
