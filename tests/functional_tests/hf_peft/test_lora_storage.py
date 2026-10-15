# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Mixed frozen/trainable storage on real two-GPU FSDP, including PyTorch 2.6/2.7."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from nemo_automodel.components._peft.lora import LinearLoRA, patch_moe_module
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts


def _storage_worker(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for kind in ("linear", "dora", "experts"):
            torch.manual_seed(17)
            if kind == "experts":
                config = MoEConfig(
                    n_routed_experts=4,
                    n_shared_experts=0,
                    n_activated_experts=1,
                    n_expert_groups=1,
                    n_limited_groups=1,
                    train_gate=False,
                    gate_bias_update_factor=0.0,
                    aux_loss_coeff=0.0,
                    score_func="softmax",
                    route_scale=1.0,
                    dim=16,
                    inter_dim=32,
                    moe_inter_dim=32,
                    norm_topk_prob=False,
                    dtype=torch.bfloat16,
                )
                model = patch_moe_module(GroupedExperts(config).cuda(), dim=4)
            else:
                model = LinearLoRA(
                    nn.Linear(16, 16, bias=False, device="cuda", dtype=torch.bfloat16),
                    dim=4,
                    use_dora=kind == "dora",
                )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.normal_(std=0.05)
            reference = copy.deepcopy(model)
            frozen = {name: p.detach().clone() for name, p in reference.named_parameters() if not p.requires_grad}
            fully_shard_by_dtype(model, mesh, policy, None, reshard_after_forward=True)
            torch.manual_seed(29 + rank)
            for microbatch in range(2):
                model.set_requires_gradient_sync(microbatch == 1)
                x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
                args = (x,)
                if kind == "experts":
                    args = (
                        x,
                        torch.ones(8, device="cuda", dtype=torch.bool),
                        torch.ones(8, 1, device="cuda", dtype=torch.bfloat16),
                        torch.arange(8, device="cuda").remainder(4).view(-1, 1),
                    )
                actual, expected = model(*args), reference(*args)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                upstream = torch.randn_like(actual)
                (actual * upstream).float().sum().div(2).backward()
                (expected * upstream).float().sum().div(2).backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.dtype == parameter.grad.dtype == torch.float32
                    expected_grad = reference.get_parameter(name).grad
                    dist.all_reduce(expected_grad)
                    expected_grad.div_(2)
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(),
                        expected_grad,
                        rtol=0.03,
                        atol=0.002,
                        msg=lambda message: f"{kind}/{name}: {message}",
                    )
                else:
                    assert parameter.dtype == torch.bfloat16 and parameter.grad is None
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
            torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.1).step()
            for name, parameter in model.named_parameters():
                expected = reference.get_parameter(name) if parameter.requires_grad else frozen[name]
                torch.testing.assert_close(parameter.full_tensor(), expected, rtol=0.03, atol=0.002)
            if rank == 0:
                print(
                    f"{torch.__version__}: {kind} storage, accumulated gradients and optimizer step passed", flush=True
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.timeout(240)
def test_mixed_lora_storage_with_fsdp(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_storage_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
