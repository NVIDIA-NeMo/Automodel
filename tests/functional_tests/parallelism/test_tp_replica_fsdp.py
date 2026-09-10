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

"""Experimental native-FSDP replica ownership versus a CPU FP32 reference.

Run with torchrun (1 or 2 GPUs for focused validation; 8 for TP2 x DP4).
No tp_replicas helper is imported or called, including during initialization.
This intentionally tests PyTorch ownership before adding a production API.
"""

import itertools
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.training.utils import clip_grad_norm


class _TinyModel(nn.Module):
    def __init__(self, *, device: str, recompute: bool) -> None:
        super().__init__()
        self.recompute = recompute
        self.replica = nn.Linear(4, 4, device=device)
        self.col = nn.Linear(4, 8, bias=False, device=device)
        self.row = nn.Linear(8, 4, bias=False, device=device)
        self.unused = nn.Parameter(torch.empty(4, device=device))
        self.frozen = nn.Parameter(torch.empty(4, device=device), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a redundant block followed by a TP-sharded MLP.

        Args:
            inputs: Local tensor of shape [batch, hidden=4], replicated on TP
                and containing different samples on DP.

        Returns:
            Local tensor of shape [batch, hidden=4], replicated on TP.
        """
        hidden = checkpoint(self.replica, inputs, use_reentrant=False) if self.recompute else self.replica(inputs)
        return self.row(self.col(hidden.sigmoid()).sigmoid())


def _check_case(mesh: DeviceMesh, *, window: int, max_norm: float, recompute: bool) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dp_size, tp_size = mesh.shape
    dp_rank, _ = mesh.get_coordinate()
    replica_mesh = DeviceMesh("cuda", mesh.mesh.T.contiguous(), mesh_dim_names=("tp", "dp"))
    # Explicit common initialization seed, separate from the training RNG.
    # DTensor placement owns disjoint DP/TP shards and equal TP replicas.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(42)
        model = _TinyModel(device="meta", recompute=recompute)
        if tp_size > 1:
            parallelize_module(model, mesh["tp"], {"col": ColwiseParallel(), "row": RowwiseParallel()})
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        fully_shard(model.replica, mesh=replica_mesh, mp_policy=policy)
        fully_shard(model.col, mesh=mesh["dp"], mp_policy=policy)
        fully_shard(model.row, mesh=mesh["dp"], mp_policy=policy)
        fully_shard(model, mesh=replica_mesh, mp_policy=policy, reshard_after_forward=False)
        to_empty_parameters_only(model, device=device)
        with torch.no_grad():
            for parameter in model.parameters():
                nn.init.uniform_(parameter, -0.25, 0.25)

    reference = _TinyModel(device="cpu", recompute=False)
    with torch.no_grad():
        for (_, parameter), (_, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            expected.copy_(parameter.full_tensor().cpu())
        if tp_size > 1:
            local = model.replica.weight.to_local()
            peers = [torch.empty_like(local) for _ in range(tp_size)]
            dist.all_gather(peers, local, group=mesh["tp"].get_group())
            for other in peers:
                torch.testing.assert_close(other, local, rtol=0, atol=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=False)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01, foreach=False)
    # The second update is the short final accumulation window.
    for microbatches in (window, 1):
        for microbatch in range(microbatches):
            model.set_requires_gradient_sync(microbatch == microbatches - 1)
            inputs = torch.arange(4, dtype=torch.float32).reshape(1, 4) * 0.1 + dp_rank + microbatch * 0.25
            model(inputs.to(device)).square().sum().div(microbatches).backward()
            for sample_rank in range(dp_size):
                reference_inputs = inputs - dp_rank + sample_rank
                reference(reference_inputs).square().sum().div(microbatches * dp_size).backward()

        for (name, parameter), (_, expected) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            if expected.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad.full_tensor().cpu(), expected.grad, rtol=1e-5, atol=1e-6)
        norm = clip_grad_norm(max_norm, [model], device_mesh=mesh)
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
        torch.testing.assert_close(norm.cpu().float(), reference_norm, rtol=1e-5, atol=1e-6)
        optimizer.step()
        reference_optimizer.step()
        for (name, parameter), (_, expected) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.full_tensor().cpu(), expected, rtol=1e-5, atol=1e-6)
            if expected not in reference_optimizer.state:
                assert parameter not in optimizer.state, name
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                actual_state = optimizer.state[parameter][key].full_tensor().cpu()
                torch.testing.assert_close(actual_state, reference_optimizer.state[expected][key], rtol=1e-5, atol=1e-6)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
    dist.barrier()


def _run_matrix() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    torch.manual_seed(100 + dist.get_rank())
    try:
        world_size = dist.get_world_size()
        for tp_size in (1, 2):
            if world_size % tp_size:
                continue
            mesh = init_device_mesh("cuda", (world_size // tp_size, tp_size), mesh_dim_names=("dp", "tp"))
            for window, max_norm, recompute in itertools.product((1, 3), (float("inf"), 0.1), (False, True)):
                _check_case(mesh, window=window, max_norm=max_norm, recompute=recompute)
                if dist.get_rank() == 0:
                    print(
                        f"PASS tp={tp_size} dp={world_size // tp_size} window={window} clip={max_norm} ac={recompute}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run_matrix()
