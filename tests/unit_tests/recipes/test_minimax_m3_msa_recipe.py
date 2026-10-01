# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import yaml

_RECIPE = Path(__file__).parents[3] / "examples/vlm_finetune/minimax_m3/minimax_m3_vl_sft_tulu3_text_msa_16k.yaml"


def test_minimax_m3_msa_pipeline_has_one_stage_per_rank() -> None:
    config = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    pipeline = config["distributed"]["pipeline"]

    assert pipeline["pp_schedule"] == "1f1b"
    assert "layers_per_stage" not in pipeline
    assert "round_virtual_stages_to_pp_multiple" not in pipeline


def test_minimax_m3_msa_preserves_effective_global_batch() -> None:
    config = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    scheduler = config["step_scheduler"]
    distributed = config["distributed"]
    data_parallel_size = 128 // distributed["pp_size"]

    assert distributed["pp_size"] == 4
    assert distributed["ep_size"] == 32
    assert scheduler["local_batch_size"] == 4
    assert scheduler["global_batch_size"] == scheduler["local_batch_size"] * data_parallel_size
