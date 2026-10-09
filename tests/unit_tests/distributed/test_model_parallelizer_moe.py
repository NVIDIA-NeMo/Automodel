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


def test_parallelize_moe_forwards_checkpoint_moe_only() -> None:
    """Every MoE activation-checkpointing knob on the typed config must reach parallelize_model."""
    model = nn.Linear(2, 2)
    moe = MoEParallelizerConfig(ignore_router_for_ac=True, checkpoint_moe_only=True)
    mesh_context = SimpleNamespace(
        device_mesh=object(),
        moe_mesh=object(),
        moe_parallel_config=moe,
        strategy_config=None,
        activation_checkpointing=True,
        reapply_trainability=None,
        parallelize_axis_kwargs=lambda: {},
    )
    parallelizer = SimpleNamespace(_customizes_moe_fsdp=False)

    with patch("nemo_automodel.components.moe.parallelizer.parallelize_model", return_value=model) as parallelize:
        assert model_parallelizer._parallelize_moe(model, mesh_context, parallelizer=parallelizer) is model

    kwargs = parallelize.call_args.kwargs
    assert kwargs["activation_checkpointing"] is True
    assert kwargs["ignore_router_for_ac"] is True
    assert kwargs["checkpoint_moe_only"] is True
