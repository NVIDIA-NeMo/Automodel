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

"""Actual FSDP/HSDP two-update gate for frozen selector parameters.

This component uses real model-owned indexer modules and the production dtype
sharder; it is not checkpoint-loading validation. CI uses two GPUs. The same
test may run with four ranks to exercise a nontrivial HSDP replica/shard mesh.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.step5.sparse import Step5SparseIndexer
from nemo_automodel.components.optim.optimizer import _trainable_params_or_groups
from tests.unit_tests.models.step5.test_model import tiny_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4),
    reason="Requires a two-rank CUDA torchrun launch; four-rank HSDP is an additional local gate",
)


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = Step5SparseIndexer(tiny_config(), 3, dtype=torch.float32)
        self.output = nn.Linear(32, 32, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, empty: bool) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        padding = torch.zeros_like(positions, dtype=torch.bool)
        with torch.no_grad():
            projected = hidden[:, :0] if empty else hidden
            projected_positions = positions[:, :0] if empty else positions
            keys, logits = self.indexer.project_keys(projected, projected_positions)
            self.indexer(projected, projected_positions, keys, logits, positions, padding)
        return self.output(hidden)


@pytest.fixture(scope="module")
def world():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield dist.get_world_size()
    dist.destroy_process_group()


def _input(rank: int, step: int) -> torch.Tensor:
    return torch.arange(4 * 32, device="cuda").reshape(1, 4, 32).float() / 137 + rank / 7 + step / 11


@pytest.mark.parametrize("hsdp", [False, True])
@pytest.mark.parametrize("ac", [False, True])
def test_indexer_stays_immutable_across_updates_and_distributed_layouts(world, hsdp, ac):
    torch.manual_seed(1031)
    reference = Fixture().cuda()
    model = copy.deepcopy(reference)
    initial = {name: value.detach().clone() for name, value in model.indexer.named_parameters()}
    shape = (2, world // 2) if hsdp else (world,)
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=("replicate", "shard") if hsdp else ("shard",))
    # A real FP32 group contains both frozen selector and trainable output
    # parameters, which exercises the mixed-requires-grad storage contract.
    fully_shard_by_dtype(
        model,
        mesh,
        MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
        None,
        reshard_after_forward=True,
    )
    _apply_trainability_policy(model, peft_enabled=False, freeze_config=None, strict=False)
    optimizer = torch.optim.AdamW(_trainable_params_or_groups(model, []), lr=0.01, weight_decay=0.1)
    reference_optimizer = torch.optim.AdamW(_trainable_params_or_groups(reference, []), lr=0.01, weight_decay=0.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected = reference.output(torch.cat([_input(rank, step) for rank in range(world)]))
        expected.square().mean().backward()
        hidden = _input(dist.get_rank(), step).requires_grad_()
        empty = bool(dist.get_rank() % 2)
        output = checkpoint(model, hidden, empty, use_reentrant=False) if ac else model(hidden, empty)
        output.square().mean().backward()
        assert output.dtype == torch.float32
        gradient = model.output.weight.grad.full_tensor()
        torch.testing.assert_close(gradient, reference.output.weight.grad, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(gradient.norm(), reference.output.weight.grad.norm(), atol=2e-7, rtol=2e-5)
        for name, parameter in model.indexer.named_parameters():
            assert not parameter.requires_grad and parameter.grad is None, name
            assert parameter not in optimizer.state, name
            assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32, name
            torch.testing.assert_close(parameter.full_tensor(), initial[name], atol=0, rtol=0)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.output.weight.full_tensor(), reference.output.weight, atol=2e-7, rtol=2e-5)
        for name, parameter in model.indexer.named_parameters():
            torch.testing.assert_close(parameter.full_tensor(), initial[name], atol=0, rtol=0)
            assert parameter not in optimizer.state, name
