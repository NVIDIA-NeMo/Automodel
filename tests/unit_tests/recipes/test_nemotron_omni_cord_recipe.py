# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration contract for the Nemotron Omni CORD-v2 example."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml"


def test_nemotron_omni_cord_recipe_uses_fp32_master_weights() -> None:
    """The BF16 model keeps small optimizer updates in FP32 master weights."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    optimizer = config["optimizer"]
    assert optimizer["_target_"] == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
    assert optimizer["lr"] == 6e-6
    assert optimizer["master_weights"] is True
    assert optimizer["master_weight_dtype"] == "torch.float32"

    assert config["distributed"]["activation_checkpointing"] is True

    freeze_config = config["freeze_config"]
    assert freeze_config["freeze_vision_tower"] is True
    assert freeze_config["freeze_vision_projector"] is False
    assert freeze_config["freeze_audio_tower"] is True
    assert freeze_config["freeze_audio_projector"] is False
