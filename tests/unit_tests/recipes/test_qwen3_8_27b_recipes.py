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

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "examples/vlm_finetune/qwen3_8"
MODEL_ID = "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize(
    ("filename", "expects_peft"),
    [
        ("qwen3_8_27b.yaml", False),
        ("qwen3_8_27b_lora.yaml", True),
    ],
)
def test_qwen3_8_27b_recipe_contract(filename: str, expects_peft: bool) -> None:
    config = yaml.safe_load((RECIPE_DIR / filename).read_text(encoding="utf-8"))

    assert config["recipe"] == "FinetuneRecipeForVLM"
    assert config["model"]["pretrained_model_name_or_path"] == MODEL_ID
    assert config["processor"]["pretrained_model_name_or_path"] == MODEL_ID
    assert config["dataset"]["path_or_dataset"] == "mmoukouba/MedPix-VQA"
    assert config["step_scheduler"]["max_steps"] == 100
    assert config["distributed"]["activation_checkpointing"] is True
    assert ("peft" in config) is expects_peft


def test_qwen3_8_27b_lora_recipe_keeps_vision_and_output_modules_frozen() -> None:
    config = yaml.safe_load((RECIPE_DIR / "qwen3_8_27b_lora.yaml").read_text(encoding="utf-8"))

    assert config["freeze_config"]["freeze_vision_tower"] is True
    assert config["peft"]["match_all_linear"] is False
    assert "*vision_tower*" in config["peft"]["exclude_modules"]
    assert "*lm_head*" in config["peft"]["exclude_modules"]
