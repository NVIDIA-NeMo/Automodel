# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEBUG (NVBug 6599894 / AMINT-274): what size does FSDP2 reduce for EP-split experts?

On GLM-5.2 the reduce-scatter input was the GLOBAL 256-expert gradient (36 GiB fp32)
rather than the per-rank EP shard. This runs the same object graph -- ExpertParallel
over an ep mesh, then fully_shard over an ep_shard mesh -- on 4 gloo/CPU ranks, so it
executes the container's torch without needing a 64-node allocation.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_MARKER = "EP_REDUCE_PROBE "
N_EXPERTS, EXPERT_DIM, MOE_INTER_DIM = 8, 32, 16


def _worker() -> None:
    import torch
    import torch.distributed as dist
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_automodel.components.moe.parallelizer import ExpertParallel, _moe_shard_placement

    seen = []
    original = collectives.foreach_reduce

    def spy(fsdp_params, unsharded_grads, *args, **kwargs):
        seen.append(
            [
                (tuple(g.shape), g.numel(), type(g).__name__, getattr(p, "is_dtensor", None))
                for p, g in zip(fsdp_params, unsharded_grads)
            ]
        )
        return original(fsdp_params, unsharded_grads, *args, **kwargs)

    collectives.foreach_reduce = spy
    param_group.foreach_reduce = spy

    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_and_up_projs = nn.Parameter(torch.randn(N_EXPERTS, EXPERT_DIM, 2 * MOE_INTER_DIM))
            self.down_projs = nn.Parameter(torch.randn(N_EXPERTS, MOE_INTER_DIM, EXPERT_DIM))

        def forward(self) -> torch.Tensor:
            """Consume both params via ``.to_local()``, as GroupedExperts does."""
            gate = self.gate_and_up_projs
            down = self.down_projs
            gate = gate.to_local() if isinstance(gate, DTensor) else gate
            down = down.to_local() if isinstance(down, DTensor) else down
            return gate.float().pow(2).sum() + down.float().pow(2).sum()

    dist.init_process_group("gloo")
    world = dist.get_world_size()
    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names=("ep_shard", "ep"))
    ep_size = mesh["ep"].size()

    torch.manual_seed(0)
    experts = Experts()
    parallelize_module(module=experts, device_mesh=mesh["ep"], parallelize_plan=ExpertParallel())
    fully_shard(
        experts,
        mesh=mesh["ep_shard"],
        shard_placement_fn=_moe_shard_placement,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    )
    experts().backward()

    if dist.get_rank() == 0:
        per_expert = EXPERT_DIM * 2 * MOE_INTER_DIM + MOE_INTER_DIM * EXPERT_DIM
        glob = N_EXPERTS * per_expert
        for rows in seen:
            total = sum(r[1] for r in rows)
            verdict = "EP-LOCAL" if total == glob // ep_size else "GLOBAL-INFLATED" if total == glob else "OTHER"
            print(
                f"{_MARKER}torch={torch.__version__} ep_size={ep_size} numel={total} "
                f"global={glob} ep_local={glob // ep_size} verdict={verdict} rows={rows}",
                flush=True,
            )
    dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="requires gloo")
def test_ep_split_experts_reduce_local_shard() -> None:
    """The reduce-scatter input must be the per-rank EP shard, not the global parameter."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    completed = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, check=False
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    lines = [line for line in completed.stdout.splitlines() if _MARKER in line]
    assert lines, completed.stdout
    assert all("verdict=EP-LOCAL" in line for line in lines), "\n".join(lines)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
