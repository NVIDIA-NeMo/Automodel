# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tiny QLoRA smoke test (4-bit + LoRA) for the functional test suite."""

from __future__ import annotations

import sys

import pytest
import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.quantization.qlora import verify_qlora_quantization
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

import datasets

datasets.disable_caching()


def _get_cfg_path() -> str:
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            return argv[i + 1]
    raise ValueError("Expected --config/-c to be provided by the functional-test launcher")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QLoRA functional test requires CUDA")
def test_qlora_tiny_smoke():
    """
    End-to-end smoke test:
    - load a tiny HF model in 4-bit (bitsandbytes)
    - apply LoRA adapters
    - run a couple of training steps
    - assert: model is quantized and LoRA params are trainable
    """
    pytest.importorskip("bitsandbytes")

    cfg = parse_args_and_load_config(_get_cfg_path())
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    # Single-stage model in this config
    model = trainer.model_parts[0]

    is_quantized = bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(getattr(model, "config", None), "quantization_config", None)
    )
    is_quantized = is_quantized or verify_qlora_quantization(model)
    assert is_quantized, "Expected 4-bit quantization to be active for QLoRA"

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora" in n.lower() for n in trainable), f"Expected LoRA trainable params, got: {trainable[:20]}"

    # Run a very short training loop (max_steps is controlled by the config/CLI overrides)
    trainer.run_train_validation_loop()


