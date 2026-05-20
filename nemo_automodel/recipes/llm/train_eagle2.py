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

"""Minimal Llama-only EAGLE-2 training recipe.

EAGLE-2 reuses the same draft-model training objective as EAGLE-1. The
difference is in the speculative decoding tree policy at inference time.
This recipe exists as an explicit entrypoint so users can train a draft
checkpoint intended for an EAGLE-2 deployment workflow without having to
remember that the underlying optimization is identical to EAGLE-1.
"""

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe


class TrainEagle2Recipe(TrainEagle1Recipe):
    """Recipe alias for EAGLE-2 draft training."""


def main(config_path: str | None = None):
    """Entrypoint for ``TrainEagle2Recipe``."""
    if config_path is None:
        raise ValueError("config_path is required for TrainEagle2Recipe")
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle2Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()
