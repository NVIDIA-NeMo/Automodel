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
"""Two-rank FSDP weight-gradient regression, including empty supervision."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy


def _worker():
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp_cp",))
        for empty_rank in (False, True):
            torch.manual_seed(43)
            hidden = torch.randn(2, 11, 16, device=device)
            weight = torch.randn(32, 16, device=device)
            labels = torch.randint(32, (2, 11), device=device)
            labels[0, :3] = -100
            if empty_rank:
                labels[1] = -100
            normalizer = (labels != -100).sum().item()
            sharded_weight = torch.nn.Parameter(distribute_tensor(weight, mesh, [Shard(0)]))
            local_hidden = hidden[rank].clone().requires_grad_()
            loss = ChunkedCrossEntropy(4, compile=False)(
                local_hidden,
                labels[rank],
                sharded_weight,
                num_label_tokens=normalizer,
                grad_reduce_group=dist.group.WORLD,
            )
            # Match the recipe's compensation for FSDP's averaged gradients.
            (loss * 2).backward()
            ref_hidden = hidden.clone().requires_grad_()
            ref_weight = weight.clone().requires_grad_()
            reference = F.cross_entropy(F.linear(ref_hidden, ref_weight).flatten(0, 1), labels.flatten())
            reference.backward()
            torch.testing.assert_close(sharded_weight.grad.full_tensor(), ref_weight.grad, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(local_hidden.grad / 2, ref_hidden.grad[rank], rtol=2e-5, atol=2e-6)
            total_loss = loss.detach().clone()
            dist.all_reduce(total_loss)
            torch.testing.assert_close(total_loss, reference)
            optimizer = torch.optim.SGD([sharded_weight], lr=0.1)
            ref_optimizer = torch.optim.SGD([ref_weight], lr=0.1)
            optimizer.step()
            ref_optimizer.step()
            torch.testing.assert_close(sharded_weight.full_tensor(), ref_weight, rtol=2e-5, atol=2e-6)
        if rank == 0:
            print("CHUNKED_CE_DISTRIBUTED_PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for sharded gradients")
def test_sharded_weight_gradient_parity():
    repo_root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", __file__, "--worker"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert "CHUNKED_CE_DISTRIBUTED_PASS" in result.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
