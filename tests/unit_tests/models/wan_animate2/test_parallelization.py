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

"""Test strategy selection and whole-block checkpoint boundaries on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed.parallelizer import get_parallelization_strategy
from nemo_automodel.components.models.wan_animate2.parallelization import WanAnimate2ParallelizationStrategy


@pytest.mark.parametrize("checkpointing", [False, True, "selective"])
def test_strategy_installs_forward_before_shared_sharding(tiny_model, checkpointing):
    strategy = get_parallelization_strategy(tiny_model)
    assert isinstance(strategy, WanAnimate2ParallelizationStrategy)
    mesh = SimpleNamespace(mesh_dim_names=())
    with patch(
        "nemo_automodel.components.distributed.parallelizer.DefaultParallelizationStrategy.parallelize",
        return_value=tiny_model,
    ) as shard:
        result = strategy.parallelize(tiny_model, mesh, activation_checkpointing=checkpointing)
    assert result is tiny_model
    assert tiny_model._wan_animate2_training
    assert all(isinstance(block, CheckpointWrapper) == bool(checkpointing) for block in tiny_model.blocks)
    assert shard.call_args.kwargs["activation_checkpointing"] is False


@pytest.mark.parametrize("axis", ["tp", "cp", "pp"])
def test_unsupported_parallelism_fails_before_model_surgery(tiny_model, axis):
    class Mesh:
        mesh_dim_names = (axis,)

        def __getitem__(self, name):
            return SimpleNamespace(size=lambda: 2)

    with pytest.raises(ValueError, match=f"does not support {axis}"):
        WanAnimate2ParallelizationStrategy().parallelize(tiny_model, Mesh())
    assert not getattr(tiny_model, "_wan_animate2_training", False)
