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

"""Configuration contract for the dense Qwen3-VL CP2 example."""

from pathlib import Path

import yaml

from nemo_automodel.components.distributed.cp_vision_frame_shard import CpVisionFrameShardingConfig
from nemo_automodel.recipes._dist_utils import parse_distributed_section

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "examples/vlm_finetune/qwen3/qwen3_vl_8b_cp2_vision_frame_shard.yaml"


def test_qwen3_vl_cp2_example_enables_typed_vision_frame_sharding() -> None:
    """The runnable example selects dense Qwen3-VL, CP2, and the typed policy."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    distributed = parse_distributed_section(raw_config["distributed"])

    assert raw_config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert raw_config["distributed"]["cp_size"] == 2
    assert distributed["strategy_config"].multimodal.vision.frame_sharding == CpVisionFrameShardingConfig(
        enabled=True,
        mesh_dims=("cp",),
        min_tokens=0,
    )
