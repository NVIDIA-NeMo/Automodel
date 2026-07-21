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

"""Thin entry point for multimodal fine-tuning.

Recommended invocation (uses AM's CLI launcher):

    automodel examples/multimodal_finetune/bagel/bagel_sft.yaml --nproc-per-node 8

This script remains available as a direct entry point:

    python -m torch.distributed.run --nproc_per_node=8 \\
        examples/multimodal_finetune/finetune.py \\
        -c examples/multimodal_finetune/bagel/bagel_sft.yaml
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Running recipes via examples/ scripts is deprecated. "
    "Use: automodel <config.yaml> [--nproc-per-node N]\n"
    "See BREAKING_CHANGES.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal


def main(config: str = "examples/multimodal_finetune/bagel/bagel_sft.yaml") -> None:
    """Run the multimodal fine-tuning recipe from a YAML config."""
    cfg = parse_args_and_load_config(config)
    recipe = FinetuneRecipeForMultimodal(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
