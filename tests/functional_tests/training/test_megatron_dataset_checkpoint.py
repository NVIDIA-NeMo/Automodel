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
import shutil
from pathlib import Path

import torch

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.llm.train_ft import _build_tokenizer

"""
This test is to make sure that JSONL dataset can be checkpointed and loaded correctly.
"""


def test_megatron_dataset_checkpointing():
    cfg_path = Path(__file__).parents[4] / "examples" / "llm_pretrain" / "megatron_pretrain_gpt2.yaml"
    cfg = parse_args_and_load_config(cfg_path)
    dist_env = initialize_distributed(
        backend=cfg.get("dist_env", {}).get("backend", "nccl"),
        timeout_minutes=cfg.get("dist_env", {}).get("timeout_minutes", 1),
    )
    mesh_context = create_distributed_setup_from_config(cfg, world_size=dist_env.world_size).mesh_context
    device_mesh = mesh_context.device_mesh
    dp_rank = device_mesh["dp"].get_local_rank()
    dp_world_size = device_mesh["dp"].size()
    tp_rank = device_mesh["tp"].get_local_rank()
    pp_rank = device_mesh["pp"].get_local_rank()

    # mock checkpoint config and checkpointer
    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=cfg.checkpoint.checkpoint_dir,
        model_save_format="safetensors",
        model_cache_dir="",
        model_repo_id="",
        save_consolidated=False,
        is_peft=False,
        model_state_dict_keys=[],
    )
    checkpointer = Checkpointer(
        config=checkpoint_config,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
    )

    # Override the example config's batch sizes / schedule with the small values this test needs.
    cfg.step_scheduler.local_batch_size = 2
    cfg.step_scheduler.global_batch_size = 4
    cfg.step_scheduler.max_steps = None
    cfg.step_scheduler.val_every_steps = 10
    # Megatron datasets require a tokenizer; the recipe supplies it via runtime (build(tokenizer=...)),
    # so build it here the same way (from the dataset/model config) instead of relying on the default.
    _, tokenizer = _build_tokenizer(cfg.model, cfg.dataset)
    dataset = RecipeConfig(cfg).dataloader.build(
        dp_rank=dp_rank, dp_world_size=dp_world_size, pp_enabled=False, tokenizer=tokenizer
    )

    # fast-forward. not necessary, but we want to make sure the dataset is not at the beginning.
    for i, batch in enumerate(dataset):
        if i == 2:
            # save checkpoint
            checkpointer.save_on_dp_ranks(dataset, "dataloader", cfg.checkpoint.checkpoint_dir)
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

    dataset = RecipeConfig(cfg).dataloader.build(
        dp_rank=dp_rank, dp_world_size=dp_world_size, pp_enabled=False, tokenizer=tokenizer
    )

    initial_batch = next(iter(dataset))
    for k in ["input_ids", "labels"]:
        assert torch.any(initial_batch[k] != expected_batch[k]), (
            f"Initial batch key {k, initial_batch[k]} should not be equal to expected batch key {k, expected_batch[k]}"
        )

    # load checkpoint
    checkpointer.load_on_dp_ranks(dataset, "dataloader", cfg.checkpoint.checkpoint_dir)

    for i, batch in enumerate(dataset):
        for k in batch.keys():
            assert torch.all(batch[k] == expected_batch[k]), (
                f"Batch key {k, batch[k]} is not equal to expected batch key {k, expected_batch[k]}"
            )
        break

    torch.distributed.barrier(device_mesh["dp"].get_group())
    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path("checkpoints/").exists():
            shutil.rmtree(Path("checkpoints/"))
    torch.distributed.barrier(device_mesh["dp"].get_group())
