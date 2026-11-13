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

"""Contract tests for the Inkling-2-Small MedPix recipes."""

from pathlib import Path

import pytest
import yaml

from nemo_automodel.components._peft.module_matcher import wildcard_match

pytestmark = pytest.mark.cpu_only

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "vlm_finetune" / "inkling2"
_SFT_RECIPE = _RECIPE_DIR / "inkling2_small_medpix_sft_ep128.yaml"
_LORA_RECIPE = _RECIPE_DIR / "inkling2_small_medpix_lora_ep32.yaml"
_MODEL_ID = "thinkingmachines/Inkling-2-Small"
_NUM_EXPERTS = 384
_GPUS_PER_NODE = 8


def _load_recipe(path: Path) -> dict:
    """Load a checked-in Inkling 2 recipe."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", [_SFT_RECIPE, _LORA_RECIPE], ids=["sft", "lora"])
def test_inkling2_recipe_layout_matches_declared_hardware(path: Path) -> None:
    recipe = _load_recipe(path)

    assert recipe["recipe"] == "FinetuneRecipeForVLM"
    assert recipe["model"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert recipe["processor"]["pretrained_model_name_or_path"] == _MODEL_ID
    backend = recipe["model"]["backend"]
    assert (backend["experts"], backend["dispatcher"]) == ("torch_mm", "hybridep")
    assert recipe["checkpoint"]["dequantize_base_checkpoint"] is True

    distributed = recipe["distributed"]
    world_size = recipe["ci"]["nodes"] * _GPUS_PER_NODE
    assert (distributed["tp_size"], distributed["cp_size"], distributed["pp_size"]) == (1, 1, 1)
    # EP spans every rank, so each rank owns whole experts and FSDP does not shard them further.
    assert distributed["ep_size"] == world_size
    assert _NUM_EXPERTS % distributed["ep_size"] == 0
    scheduler = recipe["step_scheduler"]
    assert scheduler["global_batch_size"] % (scheduler["local_batch_size"] * world_size) == 0


def test_inkling2_lora_targets_language_attention_and_experts_only() -> None:
    peft = _load_recipe(_LORA_RECIPE)["peft"]
    patterns = peft["target_modules"]

    def matched(name: str) -> bool:
        return any(wildcard_match(pattern, name) for pattern in patterns)

    for name in (
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.5.self_attn.k_proj",
        "model.language_model.layers.11.self_attn.v_proj",
        "model.language_model.layers.47.self_attn.o_proj",
        "model.language_model.layers.47.mlp.experts",
    ):
        assert matched(name), name
    for name in (
        "model.vision_tower.layers.0.self_attn.q_proj",
        "model.vision_tower.layers.0.self_attn.out_proj",
        "model.vision_tower.merger.linear_1",
        "model.language_model.layers.0.self_attn.gate_proj",
        "model.language_model.layers.0.self_attn.sink_proj",
        "model.language_model.layers.0.self_attn.r_proj",
        "model.language_model.layers.5.self_attn.k_sconv.weight_proj",
        "model.language_model.layers.0.mlp.gate",
        "lm_head",
    ):
        assert not matched(name), name
    assert peft["moe_rank_scaling"] is False
