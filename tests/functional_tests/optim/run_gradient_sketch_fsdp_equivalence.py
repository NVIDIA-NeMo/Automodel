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

"""Run with torch.distributed.run --nproc-per-node=2; compare the sketched update to a dense reference."""

import copy
import os

# This test checks fp32 equivalence to 1e-5; TF32 matmuls (forced by the CI image through this
# variable) do not meet that. The override is read once at first use, so clear it before torch loads.
os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from nemo_automodel.components.optim.gradient_sketch import GradientSketchOptimizer, SketchSettings


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(19)
    reference = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).cuda()
    sharded = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    fully_shard(sharded, mesh=mesh)
    optimizer = GradientSketchOptimizer(
        sharded,
        torch.optim.AdamW(sharded.parameters(), lr=0.01, weight_decay=0),
        SketchSettings(rank=3, target_modules=("0",), optimizer="sgd"),
    )
    baseline = torch.optim.AdamW(
        [p for n, p in reference.named_parameters() if n != "0.weight"], lr=0.01, weight_decay=0
    )
    torch.manual_seed(22)
    x = torch.randn(8, 7, device="cuda", requires_grad=True)
    reference(x).square().sum().div(8).backward()
    sharded(x.chunk(world)[rank]).square().sum().mul(world / 8).backward()
    a = optimizer.layers[0].projection
    projected = reference[0].weight.grad @ a.T @ a
    expected_norm = (
        projected.square().sum()
        + sum(p.grad.square().sum() for n, p in reference.named_parameters() if n != "0.weight")
    ).sqrt()
    norm = optimizer.synchronize_and_clip(dist.group.WORLD, 0.3)
    torch.testing.assert_close(norm, expected_norm, rtol=1e-5, atol=1e-5)
    scale = min(1.0, 0.3 / (expected_norm.item() + 1e-6))
    with torch.no_grad():
        reference[0].weight.add_(projected, alpha=-0.01 * scale)
        for n, p in reference.named_parameters():
            if n != "0.weight":
                p.grad.mul_(scale)
    baseline.step()
    optimizer.step()
    for (name, actual), expected in zip(sharded.named_parameters(), reference.parameters()):
        torch.testing.assert_close(actual.full_tensor(), expected, rtol=1e-5, atol=1e-6, msg=name)
    assert sharded[0].weight.grad is None
    if rank == 0:
        print("GRADIENT_SKETCH_FSDP_EQUIVALENCE_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
