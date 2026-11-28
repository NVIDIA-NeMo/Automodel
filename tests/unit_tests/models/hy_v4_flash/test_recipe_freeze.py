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

"""Text-loss recipes freeze the vision tower (``vit``), which the shared modality aliases do not match."""

from pathlib import Path

import pytest
import torch
import yaml

from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing
from tests.unit_tests.models.hy_v4_flash.conftest import CPU_BACKEND, TOKEN_RANGE, tiny_config

REPO = Path(__file__).resolve().parents[4]
RECIPES = [
    "examples/llm_finetune/hy_v4_flash/hy4_flash_tulu3_128k_cp16_ep64_100steps.yaml",
    "examples/vlm_finetune/hy_v4_flash/hy4_flash_medpix_4k_ep16_lora.yaml",
]


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_freeze_section_freezes_every_vision_parameter(recipe: str) -> None:
    with (REPO / recipe).open() as handle:
        freeze_config = yaml.safe_load(handle)["freeze_config"]
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND).float()
    model.initialize_weights(dtype=torch.float32)
    apply_parameter_freezing(model, freeze_config)

    vision = {name: p for name, p in model.named_parameters() if name.startswith("vit.")}
    assert vision and not any(p.requires_grad for p in vision.values())
    assert any(p.requires_grad for name, p in model.named_parameters() if name.startswith("model."))

    before = {name: p.detach().clone() for name, p in vision.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
    ids = torch.randint(*TOKEN_RANGE, (1, 24), generator=torch.Generator().manual_seed(1))
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    for name, parameter in vision.items():
        assert torch.equal(parameter, before[name]), name
