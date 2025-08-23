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

# pylint: disable=line-too-long
"""Tests for consolidated HF safetensors checkpointing for LLM."""

from pathlib import Path
import sys

import torch
import torch.distributed.tensor

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.finetune import FinetuneRecipeForNextTokenPrediction
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


def get_cfg_path() -> str:
    """
    Parses CLI args, pulls out --config and returns the config path.

    Raises:
        ValueError: if there's no --config and cfg_path = None
        ValueError: if there's --config but not yaml file passed

    Returns:
        str: the config path
    """
    argv = sys.argv[1:]
    i = 0

    while i < len(argv):
        tok = argv[i]

        # --config or -c
        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            cfg_path = argv[i + 1]
            i += 2
            return cfg_path

        i += 1

def test_consolidated_llm_checkpoint():
    """
    Tests HF consolidated checkpoint for LLM.
    """
    cfg_path = get_cfg_path()
    if "llm" in cfg_path:
        recipe_cls = FinetuneRecipeForNextTokenPrediction
        default_cfg_path = Path(__file__).parents[3] / "examples" / "llm_finetune" / "llama3_2" / "llama3_2_1b_hellaswag.yaml"
    elif "vlm" in cfg_path:
        recipe_cls = FinetuneRecipeForVLM
        default_cfg_path = Path(__file__).parents[3] / "examples" / "vlm_finetune" / "gemma3" / "gemma3_vl_4b_cord_v2.yaml"
    else:
        raise ValueError(f"Unable to infer trainer from config path: {cfg_path}")

    cfg = parse_args_and_load_config(default_cfg_path)
    cfg.model.is_meta_device = False
    trainer = recipe_cls(cfg)
    trainer.setup()

    cfg.model.is_meta_device = True
    meta_trainer = recipe_cls(cfg)
    meta_trainer.setup()

    for (n, p), (meta_n, meta_p) in zip(trainer.model.named_parameters(), meta_trainer.model.named_parameters()):
        assert n == meta_n
        if isinstance(p, torch.distributed.tensor.DTensor):
            p, meta_p = p.full_tensor(), meta_p.full_tensor()
        assert torch.allclose(p, meta_p)
        assert p.device == meta_p.device
        assert p.dtype == meta_p.dtype
        assert p.shape == meta_p.shape
        assert p.requires_grad == meta_p.requires_grad

    for (n, b), (meta_n, meta_b) in zip(trainer.model.named_buffers(), meta_trainer.model.named_buffers()):
        assert n == meta_n
        if isinstance(b, torch.distributed.tensor.DTensor):
            b, meta_b = b.full_tensor(), meta_b.full_tensor()
        assert torch.allclose(b, meta_b)
        assert b.device == meta_b.device
        assert b.dtype == meta_b.dtype
        assert b.shape == meta_b.shape
        assert b.requires_grad == meta_b.requires_grad
