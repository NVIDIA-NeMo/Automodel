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

import os

import torch
import torch.distributed as dist

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.recipes._dist_setup import setup_distributed

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
torch.manual_seed(1111)

dist_setup = setup_distributed(
    {
        "strategy": "fsdp2",
        "dp_size": None,
        "dp_replicate_size": None,
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 8,
    },
    world_size=dist.get_world_size(),
)
kwargs = {
    "device_mesh": dist_setup.device_mesh,
    "moe_mesh": dist_setup.moe_mesh,
    "distributed_config": dist_setup.strategy_config,
    "moe_config": dist_setup.moe_config,
}
model = NeMoAutoModelForCausalLM.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", **kwargs)
print(model)
assert model is not None
dist.destroy_process_group()
