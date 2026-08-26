# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from pathlib import Path

import yaml

_RECIPES = Path("examples/vlm_finetune/glm5_next")


def _load(cp_size: int) -> dict:
    path = _RECIPES / f"glm5_3_flash_medpix_packed2k_ep144_cp{cp_size}_100steps.yaml"
    return yaml.safe_load(path.read_text())


def test_glm5_next_medpix_parity_recipe_contract():
    cp1 = _load(1)
    cp8 = _load(8)

    for cp_size, recipe in ((1, cp1), (8, cp8)):
        assert recipe["recipe"] == "FinetuneRecipeForVLM"
        assert recipe["model"]["pretrained_model_name_or_path"] == "zai-org/GLM-5.3-Flash"
        assert recipe["processor"]["_target_"].endswith("build_glm5_next_processor")
        assert recipe["dataset"]["path_or_dataset"] == "mmoukouba/MedPix-VQA"
        assert recipe["dataset"]["split"] == "train"
        assert recipe["step_scheduler"]["max_steps"] == 100
        assert recipe["step_scheduler"]["global_batch_size"] == 144
        assert recipe["distributed"]["ep_size"] == 144
        assert recipe["distributed"]["pp_size"] == 1
        assert recipe["distributed"]["tp_size"] == 1
        assert recipe["distributed"]["cp_size"] == cp_size
        assert recipe["distributed"]["moe"]["wrap_outer_model"] is True
        assert recipe["packed_sequence"]["packing_format"] == "neat"
        assert recipe["packed_sequence"]["pack_size"] == 2048
        assert recipe["packed_sequence"]["collate_max_length"] == 2048
        assert recipe["wandb"]["enable"] is True
        assert recipe["wandb"]["group"] == "glm5_3_flash_medpix_packed2k_ep144_cp1_cp8_parity"

    parity_fields = (
        "seed",
        "rng",
        "step_scheduler.global_batch_size",
        "step_scheduler.local_batch_size",
        "step_scheduler.max_steps",
        "dataset",
        "packed_sequence",
        "optimizer",
        "lr_scheduler",
        "freeze_config",
    )
    for field in parity_fields:
        left = cp1
        right = cp8
        for part in field.split("."):
            left = left[part]
            right = right[part]
        assert left == right, field
