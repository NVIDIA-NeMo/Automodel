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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DIFFUSION_RECIPE_DIR = REPO_ROOT / "examples" / "dllm_sft"


def test_diffusion_recipes_keep_fp32_master_parameters() -> None:
    diffusion_configs: list[tuple[Path, dict]] = []
    for path in sorted(DIFFUSION_RECIPE_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if str(config.get("recipe", "")).startswith("Diffusion"):
            diffusion_configs.append((path, config))

    assert diffusion_configs, "No diffusion recipes found"
    for path, config in diffusion_configs:
        model_dtype = config["model"].get("torch_dtype", config["model"].get("dtype"))
        assert model_dtype == "float32", f"{path.name} must load FP32 resident parameters"
        mixed_precision = config["distributed"]["mp_policy"]
        assert mixed_precision["param_dtype"] == "float32", f"{path.name} must retain FP32 master parameters"
        assert config["distributed"]["autocast_dtype"] == "bfloat16", f"{path.name} must keep BF16 autocast compute"
