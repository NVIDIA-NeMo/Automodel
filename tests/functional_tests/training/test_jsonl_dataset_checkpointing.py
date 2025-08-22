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

import os
from pathlib import Path
import shutil

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.finetune import build_distributed, build_dataloader
from nemo_automodel.components.checkpoint.checkpointing import save_dataloader, load_dataloader

"""
This test is to make sure that JSONL dataset can be checkpointed and loaded correctly.
"""

def test_jsonl_dataset_checkpointing():
    cfg_path = Path(__file__).parents[4] / "examples" / "llm" / "llama_3_2_1b_fineweb_edu.yaml"
    cfg = parse_args_and_load_config(cfg_path)
    dist_env = build_distributed(cfg.get("dist_env", {}))
    model_wrapper = cfg.distributed.instantiate(world_size=dist_env.world_size)
    device_mesh = getattr(model_wrapper, "device_mesh", None)

    dataset = build_dataloader(cfg.dataset, {}, cfg.model, {}, device_mesh, 42)[0]
    
    # fast-forward. not necessary, but we want to make sure the dataset is not at the beginning.
    for i, batch in enumerate(dataset):
        if i == 2:
            # save checkpoint
            save_dataloader(dataset, cfg.checkpoint.checkpoint_dir, device_mesh)
        elif i == 3:
            expected_batch = batch
            break

    torch.distributed.barrier()
    del dataset

    # assert the correct paths exist
    output_files = [
        "dataloader/dataloader_dp_rank_0.pt",
        "dataloader/dataloader_dp_rank_1.pt",
    ]

    for file in output_files:
        path = Path(cfg.checkpoint.checkpoint_dir) / file
        assert path.exists(), f"Expected {path} to exist"
        assert path.is_file(), f"Expected {path} to be a file"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"

    dataset = build_dataloader(cfg.dataset, {}, cfg.model, {}, device_mesh, 42)[0]

    initial_batch = next(iter(dataset))
    for k in ["input_ids", "labels"]:
        assert torch.any(initial_batch[k] != expected_batch[k]), f"Initial batch key {k, initial_batch[k]} should not be equal to expected batch key {k, expected_batch[k]}"

    # load checkpoint
    load_dataloader(dataset, cfg.checkpoint.checkpoint_dir, device_mesh)

    for i, batch in enumerate(dataset):
        for k in batch.keys():
            assert torch.all(batch[k] == expected_batch[k]), f"Batch key {k, batch[k]} is not equal to expected batch key {k, expected_batch[k]}"
        break

    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path("checkpoints/").exists():
            shutil.rmtree(Path("checkpoints/"))
    torch.distributed.barrier()
