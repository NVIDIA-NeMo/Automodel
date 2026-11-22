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
"""CPU checks of the LoRA set-up helpers of the Praxis-1 fine-tuning recipe."""

import torch

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import _init_peft_adapters
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from nemo_automodel.recipes.wam.train import lora_trains_backbone
from tests.unit_tests.models.praxis.test_praxis_model import TINY_CONFIG, cpu_backend

EXPERT_TARGETS = ["*backbone.*.ffn.*"]
ACTION_TARGETS = ["*action_expert.blocks.*.to_q"]


def _adapted_policy(targets: list[str]) -> PraxisPolicy:
    """A tiny policy built on the meta device with adapters on ``targets``, as the recipe builds it."""
    with torch.device("meta"):
        policy = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend())
    assert apply_lora_to_linear_modules(policy, PeftConfig(target_modules=targets, dim=2, alpha=4)) > 0
    return policy


def test_expert_adapters_alone_keep_the_backbone_graph() -> None:
    assert lora_trains_backbone(_adapted_policy(EXPERT_TARGETS))
    assert not lora_trains_backbone(_adapted_policy(ACTION_TARGETS))


def test_grouped_expert_adapters_are_initialized_after_materialization() -> None:
    policy = _adapted_policy(EXPERT_TARGETS + ACTION_TARGETS)
    policy.to_empty(device=torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if ".lora_" in name:
                parameter.zero_()
    _init_peft_adapters(policy, "xavier")
    seen = set()
    for name, parameter in policy.named_parameters():
        if ".lora_" not in name:
            continue
        seen.add(name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1])
        if name.endswith(("lora_A.weight", "_A")):
            assert parameter.abs().sum() > 0, name
        else:
            assert not parameter.abs().sum(), name
    assert {"lora_gate_and_up_A", "lora_down_A", "lora_A"} <= seen
