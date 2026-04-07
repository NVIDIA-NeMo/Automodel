# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Example launcher for dLLM (diffusion LLM) SFT fine-tuning.

Usage (single GPU):
    python examples/dllm_sft/finetune.py -c examples/dllm_sft/mdlm_sft.yaml

Usage (multi-GPU):
    python -m torch.distributed.run --nproc-per-node=8 \
        examples/dllm_sft/finetune.py -c examples/dllm_sft/mdlm_sft.yaml

When run without ``-c`` it defaults to the MDLM YAML.
"""

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe


def main(default_config_path="examples/dllm_sft/mdlm_sft.yaml") -> None:
    """Entry-point for dLLM SFT training."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = DiffusionLMSFTRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main()
