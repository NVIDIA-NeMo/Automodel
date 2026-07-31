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

"""Single-GPU SCDD (self-correcting discrete diffusion) training smoke test."""

from __future__ import annotations

import sys

import datasets
import pytest
import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loss.dllm_loss import SCDDLoss
from nemo_automodel.recipes.dllm.strategy import SCDDStrategy
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SCDD functional test requires CUDA")
def test_scdd_smoke():
    """End-to-end smoke test of ``dllm.mode: scdd``:

    - build the dLLM recipe from a config that selects the SCDD strategy
    - assert: the SCDD strategy and loss are the ones actually wired up, with the
      mask token id resolved and the schedule taken from the config
    - run a couple of training steps
    """
    cfg = parse_args_and_load_config(_get_cfg_path())
    recipe = DiffusionLMSFTRecipe(cfg)
    recipe.setup()

    assert isinstance(recipe.dllm_strategy, SCDDStrategy)
    loss_fn = recipe.dllm_loss_fn
    assert isinstance(loss_fn, SCDDLoss)
    # setup_extra installs the resolved id; a stale 0 here would corrupt with a
    # real token and train garbage without ever failing loudly.
    assert loss_fn.mask_token_id == recipe.mask_token_id
    assert loss_fn.max_ratio > 0, "uniform_ratio must be > 0 or SCDD degenerates to MDLM"
    # The ELBO scores uncorrupted positions too, so the denominator is the
    # supervised-token count.
    assert recipe.dllm_strategy.normalization_mode == "supervised"

    # Run a very short training loop (max_steps is controlled by the config/CLI overrides).
    # Per-step loss/grad_norm finiteness is asserted by the CPU unit tests in
    # tests/unit_tests/loss/test_dllm_loss.py; here the loop itself is the check.
    recipe.run_train_validation_loop()
