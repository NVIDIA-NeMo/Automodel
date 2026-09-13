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

"""CPU checks for unsupported MXFP4 routing semantics before loading weights."""

import pytest
import torch

from nemo_automodel.components._peft.lora_experts_mxfp4 import (
    GroupedExpertsDeepEPLoRAMXFP4,
    GroupedExpertsLoRAMXFP4,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsDeepEPMXFP4, GroupedExpertsMXFP4


@pytest.mark.parametrize("passthrough", (False, True))
@pytest.mark.parametrize(
    "cls",
    (
        GroupedExpertsMXFP4,
        GroupedExpertsLoRAMXFP4,
        GroupedExpertsDeepEPMXFP4,
        GroupedExpertsDeepEPLoRAMXFP4,
    ),
)
def test_rejects_router_weight_after_down(cls: type[torch.nn.Module], passthrough: bool) -> None:
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        norm_topk_prob=False,
        apply_router_weight_after_down=True,
    )
    base = GroupedExpertsDeepEP if issubclass(cls, GroupedExpertsDeepEP) else GroupedExperts
    with torch.device("meta"):
        orig = base(config, BackendConfig(experts="torch_mm"))
        with pytest.raises(NotImplementedError, match="apply_router_weight_after_down=True"):
            cls(orig, passthrough=passthrough)
