# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from pathlib import Path

import yaml

_RECIPES = Path("examples/vlm_finetune/glm5_next")
_RECIPE_NAME = "glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml"


def _load() -> dict:
    return yaml.safe_load((_RECIPES / _RECIPE_NAME).read_text())


def test_glm5_next_medpix_ep72_cp2_recipe_contract():
    recipe = _load()

    assert recipe["recipe"] == "FinetuneRecipeForVLM"
    assert recipe["model"]["pretrained_model_name_or_path"] == "zai-org/GLM-5.3-Flash"
    assert recipe["processor"]["_target_"].endswith("build_glm5_next_processor")
    assert recipe["dataset"] == {
        "_target_": "nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset",
        "path_or_dataset": "mmoukouba/MedPix-VQA",
        "split": "train",
    }
    assert recipe["step_scheduler"]["max_steps"] == 100
    assert recipe["step_scheduler"]["global_batch_size"] == 144
    assert recipe["step_scheduler"]["local_batch_size"] == 1
    assert recipe["distributed"]["strategy"] == "fsdp2"
    assert recipe["distributed"]["ep_size"] == 72
    assert recipe["distributed"]["cp_size"] == 2
    assert recipe["distributed"]["pp_size"] == 1
    assert recipe["distributed"]["tp_size"] == 1
    assert recipe["distributed"]["defer_fsdp_grad_sync"] is False
    assert recipe["distributed"]["moe"]["wrap_outer_model"] is True
    assert recipe["packed_sequence"]["packing_format"] == "neat"
    assert recipe["packed_sequence"]["max_length"] == 2048
    assert recipe["packed_sequence"]["pack_size"] == 2048
    assert recipe["packed_sequence"]["collate_max_length"] == 2048
    assert recipe["wandb"]["enable"] is True
    assert recipe["wandb"]["name"] == "glm5_3_flash_medpix_packed2k_ep72_cp2_100steps"
    assert "ep72" in recipe["wandb"]["tags"]
    assert "cp2" in recipe["wandb"]["tags"]
    assert recipe["ci"]["nodes"] == 9
    assert recipe["ci"]["time"] == "01:00:00"
