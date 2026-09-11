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

import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).parents[3]
RECIPE_PATH = REPO_ROOT / "examples/llm_pretrain/qwen3_8b_pretrain.yaml"
FINEWEB_RECIPE_PATH = REPO_ROOT / "examples/llm_pretrain/qwen3_8b_fineweb_edu_pretrain.yaml"


def test_qwen3_8b_pretrain_recipe_uses_full_official_architecture() -> None:
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))

    assert recipe["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    model_config = recipe["model"]["config"]
    assert model_config == {
        "_target_": "transformers.Qwen3Config",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "hidden_act": "silu",
        "rms_norm_eps": 1.0e-6,
        "initializer_range": 0.02,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "use_cache": False,
    }
    assert recipe["model"]["load_base_model"] is False
    assert recipe["model"]["force_hf"] is False
    assert recipe["dataset"]["tokenizer"] is None
    assert recipe["validation_dataset"]["tokenizer"] is None
    assert recipe["dataset"]["vocab_size"] == 151936
    assert recipe["dataset"]["eod_token_id"] == 151645

    config_kwargs = dict(model_config)
    config_kwargs.pop("_target_")
    with torch.device("meta"):
        model = Qwen3ForCausalLM(Qwen3Config(**config_kwargs))
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_190_735_360


def test_qwen3_8b_pretrain_recipe_has_reproducible_eight_gpu_smoke_shape() -> None:
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))

    assert recipe["seed"] == 1234
    assert recipe["step_scheduler"] == {
        "global_batch_size": 8,
        "local_batch_size": 1,
        "max_steps": 10,
        "val_every_steps": 5,
        "ckpt_every_steps": 10,
    }
    assert recipe["dataset"]["seq_length"] == 2048
    assert recipe["distributed"]["strategy"] == "fsdp2"
    assert recipe["distributed"]["tp_size"] == 2
    assert recipe["distributed"]["activation_checkpointing"] is True
    assert recipe["ci"]["nodes"] == 1
    assert recipe["ci"]["nproc_per_node"] == 8


def test_qwen3_8b_fineweb_recipe_uses_real_qwen_tokenized_pretraining_data() -> None:
    recipe = yaml.safe_load(FINEWEB_RECIPE_PATH.read_text(encoding="utf-8"))
    smoke_recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))

    assert recipe["model"]["config"] == smoke_recipe["model"]["config"]
    assert recipe["step_scheduler"]["max_steps"] == 100000
    assert recipe["step_scheduler"]["num_epochs"] is None
    assert recipe["step_scheduler"]["val_every_steps"] == 1000
    assert recipe["step_scheduler"]["ckpt_every_steps"] == 10000
    assert recipe["dataset"]["_target_"].endswith("MegatronPretrainingConfig")
    assert recipe["validation_dataset"]["_target_"].endswith("MegatronPretrainingConfig")
    assert "mock" not in recipe["dataset"]["_target_"].lower()
    assert recipe["dataset"]["tokenizer"]["pretrained_model_name_or_path"] == "Qwen/Qwen3-8B"
    assert recipe["dataset"]["seq_length"] == 2048
    assert recipe["dataset"]["splits_to_build"] == "train"
    assert recipe["validation_dataset"]["splits_to_build"] == "validation"
    assert recipe["dataloader"]["dataloader_type"] == "cyclic"
    assert recipe["dataloader"]["num_workers"] == 0
    assert recipe["validation_dataloader"]["num_workers"] == 0
    assert recipe["checkpoint"]["restore_from"] == "LATEST"
    assert recipe["checkpoint"]["is_async"] is True
    assert recipe["optimizer"]["master_weights"] is True
    assert recipe["optimizer"]["store_param_remainders"] is True
    assert recipe["optimizer"]["exp_avg_dtype"] == "torch.float32"
    assert recipe["optimizer"]["exp_avg_sq_dtype"] == "torch.float32"
