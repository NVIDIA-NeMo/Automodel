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


from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from nemo_automodel.components.distributed import model_parallelizer
from nemo_automodel.components.distributed.config import MoEParallelizerConfig


def test_parallelize_moe_forwards_the_moe_checkpointing_mode() -> None:
    """The activation-checkpointing mode and the router knob must reach parallelize_model unchanged."""
    model = nn.Linear(2, 2)
    moe = MoEParallelizerConfig(
        ignore_router_for_ac=True, fsdp2_forward_prefetch_depth=2, fsdp2_backward_prefetch_depth=3
    )
    mesh_context = SimpleNamespace(
        device_mesh=object(),
        moe_mesh=object(),
        moe_parallel_config=moe,
        strategy_config=None,
        activation_checkpointing="moe",
        reapply_trainability=None,
        parallelize_axis_kwargs=lambda: {},
    )
    parallelizer = SimpleNamespace(_customizes_moe_fsdp=False)

    with patch("nemo_automodel.components.moe.parallelizer.parallelize_model", return_value=model) as parallelize:
        assert model_parallelizer._parallelize_moe(model, mesh_context, parallelizer=parallelizer) is model

    kwargs = parallelize.call_args.kwargs
    assert kwargs["activation_checkpointing"] == "moe"
    assert kwargs["ignore_router_for_ac"] is True
    assert kwargs["fsdp2_forward_prefetch_depth"] == 2
    assert kwargs["fsdp2_backward_prefetch_depth"] == 3
    assert "checkpoint_moe_only" not in kwargs


def test_moe_parallelizer_config_keeps_positional_slots() -> None:
    """The prefetch depths are keyword-only, so older positional calls bind as before."""
    moe = MoEParallelizerConfig(True, False, "float32", False)

    assert moe.ignore_router_for_ac is True
    assert moe.reshard_after_forward is False
    assert moe.lm_head_precision == "float32"
    assert moe.wrap_outer_model is False
    assert moe.fsdp2_forward_prefetch_depth == 0
    assert moe.fsdp2_backward_prefetch_depth == 0
