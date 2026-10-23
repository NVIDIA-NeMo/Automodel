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

"""Two-rank checkpoint and optimizer coverage for an exact FP32 update surface."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
from nemo_automodel.components.utils.model_utils import FreezeConfig, apply_parameter_freezing


def _worker(rank: int, rendezvous: str, checkpoint_dir: str, source: dict[str, torch.Tensor]) -> None:
    """Load and update one FSDP rank; compare against a replicated BF16-compute reference."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Tanh(),
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 2, bias=False),
        ).to(device=device, dtype=torch.float32)
        apply_parameter_freezing(model, FreezeConfig(freeze_parameters=("0.weight", "3.weight")))
        prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        reference = copy.deepcopy(model)
        reference.load_state_dict(source)
        mesh = init_device_mesh("cuda", (2,))
        fully_shard_by_dtype(
            model,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            offload_policy=None,
        )
        assert isinstance(model, FSDPModule)
        parameters = dict(model.named_parameters())
        assert parameters.keys() == source.keys()
        for name, parameter in parameters.items():
            assert isinstance(parameter, DTensor), name
            assert parameter.dtype == source[name].dtype, name
            assert parameter.requires_grad is (name == "2.weight"), name

        # Load into already sharded, mixed-storage destinations. The source FP32
        # values cannot survive an intermediate BF16 holder, even if later widened.
        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=checkpoint_dir)
        model.load_state_dict(state["model"])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.full_tensor().cpu(), source[name], rtol=0, atol=0, msg=name)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0625)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.0625)
        inputs = torch.full((2, 4), float(rank + 1), device=device, dtype=torch.bfloat16)
        actual = model(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference(inputs)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss = actual.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        # Match FSDP's average over two different rank-local batches. Autocast
        # retains the reference's FP32 master parameter and uses BF16 GEMMs.
        for source_rank in range(2):
            reference_inputs = torch.full_like(inputs, float(source_rank + 1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reference_output = reference(reference_inputs)
            (reference_output.float().square().mean() / 2).backward()
        for name, parameter in model.named_parameters():
            if name == "2.weight":
                assert parameter.grad is not None
                gradient = parameter.grad.full_tensor()
                assert gradient.dtype == torch.float32
                assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
                torch.testing.assert_close(gradient, reference[2].weight.grad, rtol=0, atol=0)
            else:
                assert parameter.grad is None, name
        optimizer.step()
        reference_optimizer.step()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            full_parameter = parameter.full_tensor()
            assert parameter.dtype == source[name].dtype, name
            assert parameter.requires_grad is (name == "2.weight"), name
            torch.testing.assert_close(full_parameter, reference_parameters[name], rtol=0, atol=0, msg=name)
            if name == "2.weight":
                assert not torch.equal(full_parameter.cpu(), source[name])
            else:
                torch.testing.assert_close(full_parameter.cpu(), source[name], rtol=0, atol=0, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices for actual mixed-storage FSDP")
def test_parameter_freezes_survive_fsdp_checkpoint_load_and_optimizer_step(tmp_path: Path) -> None:
    """FP32 selected values load exactly, update as expected, and leave frozen BF16 values unchanged."""
    source = {
        "0.weight": torch.arange(1, 17, dtype=torch.bfloat16).reshape(4, 4) / 32,
        "2.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) / 100 + 0.1234567,
        "3.weight": torch.arange(1, 9, dtype=torch.bfloat16).reshape(2, 4) / 16,
    }
    assert not torch.equal(source["2.weight"], source["2.weight"].bfloat16().float())
    checkpoint_dir = str(tmp_path / "checkpoint")
    dcp.save({"model": source}, checkpoint_id=checkpoint_dir, no_dist=True)
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), checkpoint_dir, source), nprocs=2, join=True)
