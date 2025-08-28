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
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_dataloader
from nemo_automodel.components.checkpoint.checkpointing import save_dataloader, load_dataloader

"""
This test is to make sure that JSONL dataset can be checkpointed and loaded correctly.
"""

def test_megatron_dataset_checkpointing():
    cfg_path = Path(__file__).parents[4] / "examples" / "llm_pretrain" / "megatron_pretrain.yaml"
    cfg = parse_args_and_load_config(cfg_path)
    dist_env = build_distributed(cfg.get("dist_env", {}))
    model_wrapper = cfg.distributed.instantiate(world_size=dist_env.world_size)
    device_mesh = getattr(model_wrapper, "device_mesh", None)
    dp_rank = device_mesh["dp"].get_local_rank()
    dp_world_size = device_mesh["dp"].size()
    tp_rank = device_mesh["tp"].get_local_rank()

    dataset = build_dataloader(
        cfg_ds=cfg.dataset,
        cfg_dl=cfg.dataloader,
        cfg_model=cfg.model,
        cfg_ps={},
        seed=42,
        local_batch_size=cfg.step_scheduler.local_batch_size,
        global_batch_size=cfg.step_scheduler.global_batch_size,
        max_steps=None,
        val_check_interval=10,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        pp_enabled=False,
    )[0]
    
    # fast-forward. not necessary, but we want to make sure the dataset is not at the beginning.
    for i, batch in enumerate(dataset):
        if i == 2:
            # save checkpoint
            save_dataloader(dataset, cfg.checkpoint.checkpoint_dir, dp_rank, tp_rank)
        elif i == 3:
            expected_batch = batch
            break

    torch.distributed.barrier(device_mesh["dp"].get_group())
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

    dataset = build_dataloader(
        cfg_ds=cfg.dataset,
        cfg_dl=cfg.dataloader,
        cfg_model=cfg.model,
        cfg_ps={},
        seed=42,
        local_batch_size=cfg.step_scheduler.local_batch_size,
        global_batch_size=cfg.step_scheduler.global_batch_size,
        max_steps=None,
        val_check_interval=10,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        pp_enabled=False,
    )[0]

    initial_batch = next(iter(dataset))
    for k in ["input_ids", "labels"]:
        assert torch.any(initial_batch[k] != expected_batch[k]), f"Initial batch key {k, initial_batch[k]} should not be equal to expected batch key {k, expected_batch[k]}"

    # load checkpoint
    load_dataloader(dataset, cfg.checkpoint.checkpoint_dir, dp_rank)

    for i, batch in enumerate(dataset):
        for k in batch.keys():
            assert torch.all(batch[k] == expected_batch[k]), f"Batch key {k, batch[k]} is not equal to expected batch key {k, expected_batch[k]}"
        break

    torch.distributed.barrier(device_mesh["dp"].get_group())
    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path("checkpoints/").exists():
            shutil.rmtree(Path("checkpoints/"))
    torch.distributed.barrier(device_mesh["dp"].get_group())
