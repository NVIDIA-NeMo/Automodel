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

"""Single-GPU Uno (diffusion-adapter) training smoke test."""

from __future__ import annotations

import sys

import datasets
import pytest
import torch

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loss.dllm_loss import UnoDistillLoss
from nemo_automodel.recipes.dllm.strategy import UnoStrategy
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

datasets.disable_caching()


def _get_cfg_path() -> str:
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            return argv[i + 1]
    raise ValueError("Expected --config/-c to be provided by the functional-test launcher")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Uno functional test requires CUDA")
def test_uno_smoke():
    """End-to-end smoke test of ``dllm.mode: uno``:

    - build the dLLM recipe from a config that selects the Uno strategy on a LoRA model
    - assert: the Uno strategy and loss are the ones wired up, the model carries the LoRA adapter the
      gate needs, and the block-size curriculum is parsed from the config
    - run the training loop and assert the curriculum reached its last stage
    """
    cfg = parse_args_and_load_config(_get_cfg_path())
    recipe = DiffusionLMSFTRecipe(cfg)
    recipe.setup()

    assert isinstance(recipe.dllm_strategy, UnoStrategy)
    assert isinstance(recipe.dllm_loss_fn, UnoDistillLoss)
    assert any(isinstance(m, LinearLoRA) for m in recipe.model_parts[0].modules()), "Uno trains a LoRA adapter"
    assert recipe.dllm_strategy.block_size == 2, "the curriculum starts at its first stage"

    # Per-step loss/gradient correctness is asserted by the CPU unit tests in
    # tests/unit_tests/recipes/dllm/test_uno_recipe_smoke.py; here the loop itself is the check.
    recipe.run_train_validation_loop()
    assert recipe.dllm_strategy.block_size == 8, "the curriculum did not reach its last stage"
