# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Muse Spark Open MedPix recipes."""

from pathlib import Path

import yaml

_SFT_RECIPE = Path("examples/vlm_finetune/muse_spark/muse_spark_open_medpix_nonpacked4k_ep64_pp4_100steps.yaml")
_LORA_RECIPE = Path("examples/vlm_finetune/muse_spark/muse_spark_open_medpix_nonpacked4k_lora_ep32.yaml")
_MODEL_ID = "meta-models/Muse-Spark-open-816B-A42B"


def _load_recipe(path: Path) -> dict:
    """Load a checked-in Muse recipe."""
    return yaml.safe_load(path.read_text())


def test_muse_spark_medpix_sft_recipe_contract() -> None:
    recipe = _load_recipe(_SFT_RECIPE)

    assert recipe["recipe"] == "FinetuneRecipeForVLM"
    assert recipe["model"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert recipe["processor"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert recipe["step_scheduler"]["max_steps"] == 100
    assert recipe["packed_sequence"]["packed_sequence_size"] == 0
    assert recipe["dataset"] == {
        "_target_": "nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset",
        "path_or_dataset": "mmoukouba/MedPix-VQA",
        "split": "train",
    }
    distributed = recipe["distributed"]
    assert (distributed["ep_size"], distributed["pp_size"], distributed["cp_size"]) == (64, 4, 1)
    assert distributed["pipeline"]["pp_schedule"] == "interleaved1f1b"
    assert recipe["freeze_config"]["freeze_vision_tower"] is True
    assert recipe["freeze_config"]["freeze_audio_tower"] is True


def test_muse_spark_medpix_lora_recipe_contract() -> None:
    recipe = _load_recipe(_LORA_RECIPE)

    assert recipe["recipe"] == "FinetuneRecipeForVLM"
    assert recipe["model"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert recipe["processor"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert recipe["step_scheduler"]["max_steps"] == 100
    assert recipe["distributed"]["ep_size"] == 32
    peft = recipe["peft"]
    assert peft["_target_"].endswith("PeftConfig")
    assert peft["dim"] == 8
    assert "model.layers.*.feed_forward.experts" in peft["target_modules"]
    optimizer = recipe["optimizer"]
    assert optimizer["master_weights"] is True
    assert optimizer["master_weight_dtype"] == "float32"
    assert optimizer["store_param_remainders"] is True
