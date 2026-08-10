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
REGRESSION_BENCHMARKS = (
    "examples/llm_benchmark/glm/glm47_lora.yaml",
    "examples/llm_benchmark/qwen/qwen3.5_moe_lora.yaml",
    "examples/llm_benchmark/qwen/qwen3.5_moe_te_deepep_lora.yaml",
    "examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml",
    "examples/llm_benchmark/qwen/qwen3_moe_30b_torch.yaml",
    "examples/llm_benchmark/qwen/qwen3_next_te_deepep.yaml",
)


@pytest.mark.parametrize("relative_path", REGRESSION_BENCHMARKS)
def test_regressed_fake_gate_benchmarks_recompute_router_under_ac(relative_path: str):
    """Keep the rerun regression set on deterministic router recomputation."""
    config = yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))

    assert config["distributed"]["activation_checkpointing"]
    assert config["model"]["backend"]["fake_balanced_gate"]
    assert config["distributed"]["moe"]["ignore_router_for_ac"] is False
