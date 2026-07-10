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

from pathlib import Path

import torch
import torch.distributed as dist

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes._typed_config import RecipeConfig

"""
This test is to make sure that JSONL dataset can be checkpointed and loaded correctly.
"""


def gather_helper(input_tensor):
    tensor_list = [torch.zeros_like(input_tensor) for _ in range(2)]
    dist.all_gather(tensor_list, input_tensor)
    return tensor_list


def test_megatron_data_sharding():
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
    tp_world_size = device_mesh["tp"].size()

    for key, value in {"max_steps": None, "val_every_steps": 10}.items():
        if not hasattr(cfg.step_scheduler, key):
            raise ValueError(f"step_scheduler config has no field {key!r}")
        setattr(cfg.step_scheduler, key, value)
    # Megatron datasets require a tokenizer; the recipe supplies it via runtime (build(tokenizer=...)),
    # so build it here the same way (from the dataset/model config) instead of relying on the default.
    recipe_config = RecipeConfig(cfg)
    tokenizer = recipe_config.tokenizer.build()
    dataset = recipe_config.dataloader.build(
        dp_rank=dp_rank, dp_world_size=dp_world_size, pp_enabled=False, tokenizer=tokenizer
    )

    # fast-forward. not necessary, but we want to make sure the dataset is not at the beginning.
    for i, batch in enumerate(dataset):
        if i == 2:
            batch_to_test = batch
            break

    batch_to_test = {k: v.to(dist.get_rank()) for k, v in batch_to_test.items()}

    # ensure that labels are inputs left shifted by 1
    assert torch.all(batch_to_test["labels"][:, :-1] == batch_to_test["input_ids"][:, 1:]), (
        "Labels are not inputs left shifted by 1"
    )

    dist.barrier()
    del dataset

    for key in ("input_ids", "labels"):
        gathered_tensors = gather_helper(batch_to_test[key])
        if tp_world_size > 1:
            assert torch.all(gathered_tensors[0] == gathered_tensors[1]), "Expected the same tensors for TP > 1"
        else:
            assert torch.any(gathered_tensors[0] != gathered_tensors[1]), "Expected different tensors for DP > 1"

    dist.barrier()
