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

"""Keep dLLM AdamW parameter storage separate from compute precision."""

from pathlib import Path

import pytest
import torch
import yaml

from nemo_automodel.recipes._dist_utils import parse_distributed_section

RECIPE_DIR = Path(__file__).resolve().parents[4] / "examples" / "dllm_sft"


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", sorted(RECIPE_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_dllm_recipes_preserve_fp32_adamw_storage(path):
    config = yaml.safe_load(path.read_text())
    assert config["optimizer"]["_target_"] == "torch.optim.AdamW"
    model = config["model"]
    storage_dtype = model.get("dtype", model.get("torch_dtype"))
    assert storage_dtype == "float32"
    peft = config.get("peft")
    if peft is not None:
        assert (peft.get("lora_dtype") or storage_dtype) == "float32"

    strategy = parse_distributed_section(config["distributed"])["strategy_config"]
    assert strategy.mp_policy.param_dtype == torch.bfloat16
    assert strategy.mp_policy.reduce_dtype == torch.float32
    assert strategy.mp_policy.output_dtype == torch.float32
    if model["_target_"].startswith("transformers."):
        assert strategy.mp_policy.cast_forward_inputs
