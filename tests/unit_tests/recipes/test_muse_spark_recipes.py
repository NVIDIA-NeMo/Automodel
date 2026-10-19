# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Muse Spark Open training recipes."""

from pathlib import Path

import yaml

_SFT_RECIPE = Path("examples/vlm_finetune/muse_spark/muse_spark_open_medpix_nonpacked4k_ep64_pp4_100steps.yaml")
_LORA_RECIPE = Path("examples/vlm_finetune/muse_spark/muse_spark_open_medpix_nonpacked4k_lora_ep32.yaml")
_PACKED_RECIPE = Path("examples/llm_finetune/muse_spark/muse_spark_open_tulu3_packed128k_ep32_pp2_cp32_magi.yaml")
_MODEL_ID = "meta-models/Muse-Spark-open-816B-A42B"
_OC_ENV = "$" + "{oc.env:"
_MUSE_RECIPE_DIRS = (
    Path("examples/llm_finetune/muse_spark"),
    Path("examples/vlm_finetune/muse_spark"),
)


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


def test_muse_spark_packed128k_magi_pp_recipe_contract() -> None:
    raw_recipe = _PACKED_RECIPE.read_text()
    assert _OC_ENV not in raw_recipe

    recipe = yaml.safe_load(raw_recipe)
    assert recipe["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"
    assert recipe["seed"] == 1234
    assert recipe["model"]["pretrained_model_name_or_path"] == "meta-models/Muse-Spark-open-816B-A42B"

    scheduler = recipe["step_scheduler"]
    assert scheduler["max_steps"] == 50
    assert scheduler["local_batch_size"] == scheduler["global_batch_size"] == 8

    backend = recipe["model"]["backend"]
    assert backend["attn"] == "magi"
    assert backend["compile_norm"] is True
    assert backend["experts"] == "torch_mm"
    assert backend["dispatcher"] == "hybridep"
    assert backend["enable_fsdp_optimizations"] is False

    distributed = recipe["distributed"]
    assert (distributed["tp_size"], distributed["ep_size"], distributed["cp_size"], distributed["pp_size"]) == (
        1,
        32,
        32,
        2,
    )
    assert distributed["activation_checkpointing"] is True
    assert distributed["reshard_after_forward"] is True
    assert distributed["defer_fsdp_grad_sync"] is False
    assert distributed["moe"]["reshard_after_forward"] is True
    pipeline = distributed["pipeline"]
    assert scheduler["local_batch_size"] // pipeline["pp_microbatch_size"] == 8
    assert pipeline == {
        "pp_schedule": "interleaved1f1b",
        "pp_microbatch_size": 1,
        "layers_per_stage": 16,
        "pp_recv_buffer_pool": False,
    }

    assert recipe["checkpoint"]["load_metadata_with_gloo"] is True
    assert recipe["loss_fn"]["_target_"].endswith("FusedLinearCrossEntropy")
    assert recipe["loss_fn"]["impl"] == "cce"

    assert recipe["dataset"] == {
        "_target_": "datasets.load_from_disk",
        "dataset_path": "data/tulu3_128k_longdoc_shuffle_v2/train",
    }
    packed = recipe["packed_sequence"]
    assert packed == {
        "packed_sequence_size": 131072,
        "packing_strategy": "thd",
        "prepacked": True,
        "pad_to_multiple_of": 64,
    }
    assert recipe["dataloader"]["batch_size"] == 8
    assert recipe["dataloader"]["shuffle"] is False

    optimizer = recipe["optimizer"]
    assert optimizer["master_weights"] is True
    assert optimizer["master_weight_dtype"] == "float32"
    assert optimizer["store_param_remainders"] is True
    assert optimizer["exp_avg_dtype"] == optimizer["exp_avg_sq_dtype"] == "bfloat16"
    assert recipe["clip_grad_norm"] == {"max_norm": 1.0, "backend": "triton"}
    assert recipe["wandb"]["enable"] is False
    assert recipe["ci"]["nodes"] == 16
    ci = recipe["ci"]
    assert ci["nproc_per_node"] == 4
    assert ci["cluster_tag"] == "gb200"
    assert ci["time"] == "02:00:00"
    assert ci["max_steps"] == 2
    assert ci["release"]["dataset"]["path_or_dataset_id"] == "allenai/tulu-3-sft-mixture"
    assert ci["release"]["packed_sequence"]["prepacked"] is False
    assert ci["release"]["packed_sequence"]["max_packs"] == 16
    assert ci["release"]["dataloader.shuffle"] is True
    assert ci["env_vars"]["SBATCH_SEGMENT_SIZE"] == "16"
    assert ci["env_vars"]["RDZV_TIMEOUT"] == "900"


def test_muse_spark_recipes_do_not_use_omegaconf_env_interpolation() -> None:
    """Keep every Muse example usable without an OmegaConf env resolver."""
    offenders = [
        path for directory in _MUSE_RECIPE_DIRS for path in directory.glob("*.yaml") if _OC_ENV in path.read_text()
    ]
    assert offenders == []
