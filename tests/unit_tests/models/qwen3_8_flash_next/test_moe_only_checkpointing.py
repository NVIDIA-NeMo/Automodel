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

"""MoE-only activation checkpointing wraps just the MoE sub-block and leaves the numerics unchanged."""

import copy

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.qwen3_8_flash_next.test_qsa_route_replay import _build_model, _run_step


def _layers(model):
    return model.model.language_model.layers


def test_config_exposes_checkpoint_moe_only_with_default_off() -> None:
    assert MoEParallelizerConfig().checkpoint_moe_only is False
    assert MoEParallelizerConfig(checkpoint_moe_only=True).to_dict()["checkpoint_moe_only"] is True


def test_moe_only_rejects_selective_and_unpinned_router() -> None:
    model = _build_model(reuse_routes=True)
    with pytest.raises(ValueError, match="checkpoint_moe_only"):
        apply_ac(model, ignore_router=True, selective=True, moe_only=True)
    with pytest.raises(ValueError, match="checkpoint_moe_only"):
        apply_ac(model, ignore_router=False, moe_only=True)


@pytest.mark.runtime_budget(60, hard_timeout=180, reason="three CPU forward/backward passes of the tiny HC decoder")
def test_moe_only_wraps_only_the_moe_sub_block_and_is_bitwise_identical() -> None:
    """Blocks stay unwrapped, each block's ``mlp`` becomes the checkpoint unit; loss and grads are unchanged."""
    input_ids = torch.randint(2, 64, (2, 6), generator=torch.Generator().manual_seed(11))
    reference = _build_model(reuse_routes=True)
    ref_loss, ref_grads = _run_step(reference, input_ids)

    moe_only = copy.deepcopy(reference)
    apply_ac(moe_only, ignore_router=True, moe_only=True)
    for _, block in _layers(moe_only).items():
        assert not isinstance(block, CheckpointWrapper)
        if hasattr(block, "mlp"):
            assert isinstance(block.mlp, CheckpointWrapper)
    loss, grads = _run_step(moe_only, input_ids)
    assert torch.equal(loss, ref_loss)
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        assert torch.equal(grads[name], ref_grads[name]), name

    block_ac = copy.deepcopy(reference)
    apply_ac(block_ac, ignore_router=True, moe_only=False)
    assert any(isinstance(block, CheckpointWrapper) for _, block in _layers(block_ac).items())
    loss_block, grads_block = _run_step(block_ac, input_ids)
    assert torch.equal(loss_block, ref_loss)
    for name in ref_grads:
        assert torch.equal(grads_block[name], ref_grads[name]), name


def test_moe_lookup_sees_through_the_checkpoint_wrapper() -> None:
    """apply_fsdp separates expert parameters via _get_moe_module; it must find a wrapped MoE."""
    from nemo_automodel.components.moe.layers import MoE
    from nemo_automodel.components.moe.parallelizer import _get_moe_module

    model = _build_model(reuse_routes=True)
    apply_ac(model, ignore_router=True, moe_only=True)
    for _, block in _layers(model).items():
        if hasattr(block, "mlp"):
            assert isinstance(block.mlp, CheckpointWrapper)
            assert isinstance(_get_moe_module(block), MoE)
