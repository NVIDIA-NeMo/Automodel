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


def _check_fsdp_shared_projection(mesh, device, rank):
    """Compare real FSDP head policies and main/MTP gradients to dense projection."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from nemo_automodel.components.loss.mtp import calculate_mtp_loss
    from nemo_automodel.components.loss.utils import calculate_loss, prepare_lm_weight

    for compute_dtype in (torch.bfloat16, torch.float32):
        torch.manual_seed(71)
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(16, 32, bias=False, device=device)
        reference_weight = model.lm_head.weight.detach().clone().requires_grad_()
        fully_shard(model.lm_head, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=compute_dtype))
        loss_fn = ChunkedCrossEntropy(4, compile=False)
        weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=dist.group.WORLD)
        assert weight.dtype == compute_dtype
        torch.testing.assert_close(weight, reference_weight.to(compute_dtype), rtol=0, atol=0)
        states = [torch.randn(2, 9, 16, device=device, dtype=torch.bfloat16) for _ in range(3)]
        local_states = [h[rank : rank + 1].clone().requires_grad_() for h in states]
        ref_states = [h.clone().requires_grad_() for h in states]
        labels = torch.randint(32, (2, 9), device=device)
        labels[1] = -100  # the empty peer must still participate in the weight reduction
        saved = []

        def pack(tensor):
            """Record the projection tensors retained by autograd.

            Args:
                tensor: Saved tensor of arbitrary shape; [vocab, hidden] weights are recorded.

            Returns:
                The input tensor unchanged, preserving shape, dtype and storage.
            """
            if tensor.shape == weight.shape:
                saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = calculate_loss(
                loss_fn,
                model=model,
                hidden_states=local_states[0],
                labels=labels[rank : rank + 1],
                lm_weight=weight,
                logits_dtype=torch.bfloat16,
            )
            loss = loss + calculate_mtp_loss(
                loss_fn,
                model=model,
                mtp_per_depth_h=local_states[1:],
                labels=labels[rank : rank + 1],
                lm_weight=weight,
                logits_dtype=torch.bfloat16,
            )
        assert len(saved) == 3
        assert {tensor.data_ptr() for tensor in saved} == {weight.data_ptr()}
        (loss * 2).backward()
        reference = torch.zeros((), device=device)
        for depth, hidden in enumerate(ref_states):
            targets = labels.roll(-depth, -1)
            if depth:
                targets[..., -depth:] = -100
            logits = F.linear(hidden.to(compute_dtype), reference_weight.to(compute_dtype)).bfloat16().float()
            reference = reference + (1.0 if depth == 0 else 0.05) * F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
        reference.backward()
        dist.all_reduce(loss.detach())
        # BF16 GEMM rounding depends on the token tile size (chunked vs dense).
        torch.testing.assert_close(loss, reference, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(model.lm_head.weight.grad.full_tensor(), reference_weight.grad, rtol=2e-2, atol=2e-3)
        for local, ref in zip(local_states, ref_states):
            torch.testing.assert_close(local.grad / 2, ref.grad[rank : rank + 1], rtol=2e-2, atol=2e-3)


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
        _check_fsdp_shared_projection(mesh, device, rank)
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
