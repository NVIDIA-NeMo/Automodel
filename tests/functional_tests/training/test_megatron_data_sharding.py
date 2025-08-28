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
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_dataloader

"""
This test is to make sure that JSONL dataset can be checkpointed and loaded correctly.
"""

def gather_helper(input_tensor):
    tensor_list = [torch.zeros_like(input_tensor) for _ in range(2)]
    dist.all_gather(tensor_list, input_tensor)
    return tensor_list

def test_megatron_data_sharding():
    cfg_path = Path(__file__).parents[4] / "examples" / "llm_pretrain" / "megatron_pretrain.yaml"
    cfg = parse_args_and_load_config(cfg_path)
    dist_env = build_distributed(cfg.get("dist_env", {}))
    model_wrapper = cfg.distributed.instantiate(world_size=dist_env.world_size)
    device_mesh = getattr(model_wrapper, "device_mesh", None)
    dp_rank = device_mesh["dp"].get_local_rank()
    dp_world_size = device_mesh["dp"].size()
    tp_world_size = device_mesh["tp"].size()

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
    )[0]
    
    # fast-forward. not necessary, but we want to make sure the dataset is not at the beginning.
    for i, batch in enumerate(dataset):
        if i == 2:
            batch_to_test = batch
            break

    batch_to_test = {k: v.to(dist.get_rank()) for k, v in batch_to_test.items()}

    # ensure that labels are inputs left shifted by 1
    assert torch.all(batch_to_test["labels"][:, :-1] == batch_to_test["input_ids"][:, 1:]), f"Labels are not inputs left shifted by 1"

    dist.barrier()
    del dataset

    for key in ("input_ids", "labels"):
        gathered_tensors = gather_helper(batch_to_test[key])
        if tp_world_size > 1:
            assert torch.all(gathered_tensors[0] == gathered_tensors[1]), f"Expected the same tensors for TP > 1"
        else:
            assert torch.any(gathered_tensors[0] != gathered_tensors[1]), f"Expected different tensors for DP > 1"

    dist.barrier()
