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

from __future__ import annotations

from pathlib import Path

import yaml


RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "vlm_finetune" / "gemma4"


def _load_recipe(name: str) -> dict:
    with (RECIPE_DIR / name).open(encoding="utf-8") as recipe_file:
        return yaml.safe_load(recipe_file)


def test_afriinstruct_recipes_preserve_gemma4_e2b_text_only_contract() -> None:
    for name in ("gemma4_e2b_afriinstruct_peft.yaml", "gemma4_e2b_afriinstruct_full.yaml"):
        recipe = _load_recipe(name)

        assert recipe["recipe"] == "FinetuneRecipeForVLM"
        assert recipe["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
        assert recipe["model"]["pretrained_model_name_or_path"] == "google/gemma-4-E2B-it"
        assert recipe["model"]["text_config"]["use_cache"] is False
        assert recipe["distributed"]["strategy"] == "fsdp2"
        assert recipe["distributed"]["tp_size"] == 1
        assert recipe["distributed"]["activation_checkpointing"] is False
        assert recipe["freeze_config"]["freeze_vision_tower"] is True
        assert recipe["freeze_config"]["freeze_audio_tower"] is True
        assert recipe["dataset"]["inject_fake_images"] is False
        assert recipe["dataset"]["_target_"].endswith("make_meta_dataset")


def test_afriinstruct_peft_and_full_recipes_have_distinct_training_and_checkpoint_policies() -> None:
    peft_recipe = _load_recipe("gemma4_e2b_afriinstruct_peft.yaml")
    full_recipe = _load_recipe("gemma4_e2b_afriinstruct_full.yaml")

    assert peft_recipe["peft"]["_target_"] == "nemo_automodel.components._peft.lora.PeftConfig"
    assert peft_recipe["optimizer"]["lr"] == 2.0e-4
    assert peft_recipe["checkpoint"]["save_consolidated"] is False

    assert "peft" not in full_recipe
    assert full_recipe["optimizer"]["lr"] == 2.0e-5
    assert full_recipe["checkpoint"]["save_consolidated"] == "final"
