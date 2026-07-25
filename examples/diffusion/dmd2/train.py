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

"""Launch the AutoModel-native Qwen-Image DMD2 trainer."""

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.diffusion import DMD2DiffusionRecipe


def main(default_config_path: str = "examples/diffusion/dmd2/qwen_image_dmd2.yaml") -> None:
    """Train Qwen-Image with the configured Model Optimizer DMD2 method."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = DMD2DiffusionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
