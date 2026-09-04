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

"""Configuration contracts for the Kimi K3 SFT examples."""

from pathlib import Path

import yaml

from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.recipes._dist_utils import parse_distributed_section

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/llm_finetune/kimi/k3_hellaswag_lora.yaml"
SFT_CONFIG_PATH = REPO_ROOT / "examples/llm_finetune/kimi/k3_hellaswag.yaml"


def test_k3_full_sft_recipe_uses_hybridep_with_resharding() -> None:
    """The full-SFT recipe keeps HybridEP dispatch and reshard-after-forward enabled."""
    raw_config = yaml.safe_load(SFT_CONFIG_PATH.read_text(encoding="utf-8"))
    config = load_yaml_config(SFT_CONFIG_PATH)
    distributed = parse_distributed_section(raw_config["distributed"])

    backend = config.model.backend.instantiate()

    assert isinstance(backend, BackendConfig)
    assert backend.experts == "torch_mm"
    assert backend.dispatcher == "hybridep"
    assert distributed["moe_parallel_config"].reshard_after_forward is True


def test_k3_lora_recipe_declares_expert_lora_scaling_contract() -> None:
    """The recipe keeps the validated K3 LoRA backend, PEFT, and topology settings."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = load_yaml_config(CONFIG_PATH)
    distributed = parse_distributed_section(raw_config["distributed"])

    backend = config.model.backend.instantiate()
    peft = config.peft.instantiate()

    assert isinstance(backend, BackendConfig)
    assert backend.attn == "te"
    assert backend.experts == "torch_mm"
    assert backend.dispatcher == "hybridep"

    assert isinstance(peft, PeftConfig)
    assert peft.target_modules == ["*"]
    assert peft.dim == 32
    assert peft.alpha == 32
    assert peft.use_memory_efficient_lora is False
    assert peft.use_triton is False
    assert peft.moe_rank_scaling is True

    assert config.model.pretrained_model_name_or_path == "moonshotai/Kimi-K3"
    assert config.step_scheduler.max_steps == 100
    assert config.checkpoint.enabled is True
    assert config.checkpoint.save_consolidated == "final"

    assert raw_config["distributed"]["pp_size"] == 4
    assert raw_config["distributed"]["ep_size"] == 32
    assert distributed["activation_checkpointing"] == "selective"
    assert distributed["moe_parallel_config"].ignore_router_for_ac is True
    assert distributed["moe_parallel_config"].reshard_after_forward is True

    assert raw_config["ci"] == {
        "recipe_owner": "huiyingl",
        "nodes": 32,
        "cluster_tag": "gb200",
        "time": "01:00:00",
    }
