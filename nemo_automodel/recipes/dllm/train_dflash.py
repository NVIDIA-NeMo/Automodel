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

"""DFlash SFT entry point.

DFlash training uses the common :class:`~nemo_automodel.recipes.dllm.train_ft.DiffusionLMSFTRecipe`
with ``dllm.mode: dflash`` in the YAML.  All DFlash-specific logic lives in
:class:`~nemo_automodel.recipes.dllm.strategy.DFlashStrategy`.

Usage (8-GPU)::

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/dllm/train_dflash.py \\
        -c examples/dllm_sft/dflash_sft.yaml
"""

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe


def main(config_path=None):
    """Main entry point for DFlash SFT recipe."""
    if config_path is None:
        config_path = "examples/dllm_sft/dflash_sft.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = DiffusionLMSFTRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
