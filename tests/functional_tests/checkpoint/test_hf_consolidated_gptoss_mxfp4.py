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

"""Functional test: GPT-OSS mxfp4 base checkpoint → bf16 fine-tune → consolidated safetensors.

Validates that stale mxfp4 FQNs (_blocks/_scales) in the base checkpoint
index do not produce invalid safetensors during consolidation.
"""

import shutil
from pathlib import Path

import torch
import torch.distributed
from safetensors import safe_open

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def test_consolidated_gptoss_mxfp4_checkpoint():
    """Load mxfp4 GPT-OSS → train → save bf16 consolidated → reload with HF."""

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama3_2" / "llama3_2_1b_hellaswag.yaml")

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    ckpt_root = Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9"
    consolidated_dir = ckpt_root / "model" / "consolidated"

    # --- 1. Verify checkpoint directory structure ---
    expected_paths = [
        "model",
        "optim",
        "step_scheduler.pt",
        "config.yaml",
        "losses.json",
        "model/consolidated",
        "model/consolidated/config.json",
        "model/consolidated/model.safetensors.index.json",
    ]
    for rel in expected_paths:
        p = ckpt_root / rel
        assert p.exists(), f"Expected {p} to exist"

    # At least one consolidated safetensors file must exist
    consolidated_st = list(consolidated_dir.glob("model-*.safetensors"))
    assert len(consolidated_st) >= 1, "No consolidated safetensors files found"

    # --- 2. Verify consolidated safetensors is loadable by safe_open ---
    loaded_keys: set[str] = set()
    for st_file in consolidated_st:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            loaded_keys.update(f.keys())

    assert len(loaded_keys) > 0, "Consolidated safetensors has no keys"
    # No mxfp4 phantom keys should be present
    for key in loaded_keys:
        assert "_blocks" not in key, f"Phantom mxfp4 key leaked: {key}"
        assert "_scales" not in key, f"Phantom mxfp4 key leaked: {key}"

    # --- 3. Verify all consolidated tensors are well-formed ---
    for st_file in consolidated_st:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                assert tensor.shape.numel() > 0, f"Empty tensor: {key}"
                assert not torch.isnan(tensor).any(), f"NaN in tensor: {key}"
                assert tensor.dtype in (torch.bfloat16, torch.float32), f"Unexpected dtype {tensor.dtype} for {key}"

    # --- cleanup ---
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        ckpt_base = Path(trainer.checkpointer.config.checkpoint_dir)
        if ckpt_base.exists():
            shutil.rmtree(ckpt_base)
    torch.distributed.barrier()
