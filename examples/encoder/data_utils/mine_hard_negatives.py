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

# To run this script, use the following command:
# torchrun --nproc_per_node=8 --master_port=29500 ./examples/encoder/data_utils/mine_hard_negatives.py \
#     --config examples/encoder/data_utils/mining_config.yaml \
#     --mining.model_name_or_path /path/to/encoder/checkpoint \
#     --mining.train_qa_file_path /path/to/input.json \
#     --mining.train_file_output_path /path/to/output.json \
#     --mining.cache_embeddings_dir /path/to/cache \
#     --mining.hard_neg_margin 0.95
#
# The model is loaded directly from the checkpoint path (--mining.model_name_or_path),
# so no model architecture config is needed. This allows mining with any saved
# encoder checkpoint without requiring the original training config.
#
# The mining_config.yaml contains only mining parameters and dist_env settings,
# not the model architecture. All mining parameters can also be overridden via
# command line arguments.

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.encoder import MineHardNegativesRecipe


def main(default_config_path="examples/encoder/data_utils/mining_config.yaml"):
    cfg = parse_args_and_load_config(default_config_path)
    recipe = MineHardNegativesRecipe(cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
