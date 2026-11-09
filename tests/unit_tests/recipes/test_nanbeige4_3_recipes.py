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
RECIPE_DIR = REPO_ROOT / "examples/llm_finetune/nanbeige"
MODEL_ID = "Nanbeige/Nanbeige4.3-33B"


@pytest.mark.parametrize(
    ("filename", "cp_size", "nodes", "expects_peft"),
    [
        ("nanbeige4_3_33b_tulu3_packed128k_cp16.yaml", 16, 2, False),
        ("nanbeige4_3_33b_tulu3_packed128k_cp8_lora.yaml", 8, 1, True),
    ],
)
def test_nanbeige4_3_33b_packed_cp_recipe_contract(filename: str, cp_size: int, nodes: int, expects_peft: bool) -> None:
    """Check that each recipe shuffles Tulu-3 before packing and shards whole THD packs over CP.

    Args:
        filename: Recipe file name under ``examples/llm_finetune/nanbeige``.
        cp_size: Expected context-parallel size.
        nodes: Expected number of 8-GPU nodes.
        expects_peft: Whether the recipe trains LoRA adapters.
    """
    config = yaml.safe_load((RECIPE_DIR / filename).read_text(encoding="utf-8"))

    assert config["model"]["pretrained_model_name_or_path"] == MODEL_ID
    assert config["model"]["backend"]["attn"] == "te"
    assert config["dataset"]["path_or_dataset_id"] == "allenai/tulu-3-sft-mixture"
    assert config["dataset"]["shuffle_seed"] is not None
    assert config["dataloader"]["shuffle"] is False

    pack_size = config["packed_sequence"]["packed_sequence_size"]
    assert config["packed_sequence"]["packing_strategy"] == "thd"
    assert pack_size == config["dataset"]["seq_length"] == 131072
    assert config["distributed"]["cp_size"] == cp_size
    assert config["ci"]["nodes"] == nodes
    assert pack_size % (2 * cp_size) == 0

    step_scheduler = config["step_scheduler"]
    world_size = 8 * config["ci"]["nodes"]
    dp_size = world_size // cp_size
    assert step_scheduler["global_batch_size"] % (step_scheduler["local_batch_size"] * dp_size) == 0
    assert config["packed_sequence"]["max_packs"] >= step_scheduler["max_steps"] * step_scheduler["global_batch_size"]
    assert ("peft" in config) is expects_peft
