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

import copy
from datetime import timedelta

import pytest
import torch
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.training.utils import clip_grad_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _fully_shard_tiny_model(*, cpu_offload: bool) -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).cuda()
    kwargs = {"offload_policy": CPUOffloadPolicy()} if cpu_offload else {}
    for module in model:
        fully_shard(module, **kwargs)
    fully_shard(model, **kwargs)
    return model


def _init_process_group(init_file: str, *, rank: int = 0, world_size: int = 1) -> bool:
    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == rank
        assert torch.distributed.get_world_size() == world_size
        torch.cuda.set_device(rank)
        return False

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    torch.cuda.set_device(rank)
    return True


def _run_two_rank_clip_grad_norm(rank: int, world_size: int, init_file: str, cpu_offload: bool) -> None:
    """Exercise the real multi-rank NCCL collective for either gradient residency mode."""
    owns_process_group = _init_process_group(init_file, rank=rank, world_size=world_size)
    try:
        torch.manual_seed(1234)
        device = torch.device("cuda", rank)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).to(device)
        kwargs = {"offload_policy": CPUOffloadPolicy()} if cpu_offload else {}
        for module in model:
            fully_shard(module, **kwargs)
        fully_shard(model, **kwargs)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(3, 8, device=device)).sum().backward()
        grad_norm = clip_grad_norm(1.0, [model])

        assert isinstance(grad_norm, torch.Tensor)
        expected_device = "cpu" if cpu_offload else "cuda"
        assert grad_norm.device.type == expected_device
        gathered_norms = [torch.empty((), dtype=grad_norm.dtype, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_norms, grad_norm.to(device))
        for gathered_norm in gathered_norms[1:]:
            torch.testing.assert_close(gathered_norm, gathered_norms[0])
        assert torch.isfinite(gathered_norms[0])

        for parameter in model.parameters():
            assert isinstance(parameter.grad, DTensor)
            assert torch.isfinite(parameter.grad.to_local()).all()
        optimizer.step()
        torch.distributed.barrier()
    finally:
        if owns_process_group:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp2_clip_grad_norm_matches_reference_with_and_without_cpu_offload(tmp_path, cpu_offload: bool) -> None:
    """FSDP2 clipping matches an unsharded reference in both gradient residency modes."""
    owns_process_group = _init_process_group(str(tmp_path / f"clip-grad-norm-{cpu_offload}"))
    try:
        torch.manual_seed(1234)
        reference = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).cuda()
        model = copy.deepcopy(reference)
        kwargs = {"offload_policy": CPUOffloadPolicy()} if cpu_offload else {}
        for module in model:
            fully_shard(module, **kwargs)
        fully_shard(model, **kwargs)

        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.randn(3, 8, device="cuda")
        reference(inputs).sum().backward()
        model(inputs).sum().backward()

        first_grad = next(model.parameters()).grad
        assert isinstance(first_grad, DTensor)
        expected_device = "cpu" if cpu_offload else "cuda"
        assert first_grad.to_local().device.type == expected_device

        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1.0)
        actual_norm = clip_grad_norm(1.0, [model])
        assert isinstance(actual_norm, torch.Tensor)
        assert actual_norm.device.type == expected_device
        torch.testing.assert_close(actual_norm.cpu(), expected_norm.cpu(), rtol=1e-6, atol=1e-8, check_dtype=False)

        for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            assert isinstance(actual_parameter.grad, DTensor)
            torch.testing.assert_close(
                actual_parameter.grad.to_local().cpu(), expected_parameter.grad.cpu(), rtol=1e-6, atol=1e-7
            )

        optimizer.step()
        reference_optimizer.step()
        for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            assert isinstance(actual_parameter, DTensor)
            torch.testing.assert_close(
                actual_parameter.to_local().cpu(), expected_parameter.cpu(), rtol=1e-6, atol=1e-7
            )
    finally:
        if owns_process_group:
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_two_rank_fsdp2_clip_grad_norm_with_and_without_cpu_offload(tmp_path, cpu_offload: bool) -> None:
    """Both ranks complete clipping with identical finite norms in both residency modes."""
    torch.multiprocessing.spawn(
        _run_two_rank_clip_grad_norm,
        args=(2, str(tmp_path / f"two-rank-clip-grad-norm-{cpu_offload}"), cpu_offload),
        nprocs=2,
        join=True,
    )


def test_fsdp2_gpu_clip_grad_norm_has_no_host_scalar_sync(tmp_path) -> None:
    """The standard GPU-gradient path keeps its reported norm on the device."""
    owns_process_group = _init_process_group(str(tmp_path / "clip-grad-norm-profiler"))
    try:
        model = _fully_shard_tiny_model(cpu_offload=False)
        model(torch.randn(3, 8, device="cuda")).sum().backward()
        clip_grad_norm(1.0, [model])
        model.zero_grad(set_to_none=True)
        model(torch.randn(3, 8, device="cuda")).sum().backward()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            with torch.profiler.record_function("clip_grad_norm_under_test"):
                grad_norm = clip_grad_norm(1.0, [model])

        assert isinstance(grad_norm, torch.Tensor)
        assert grad_norm.device.type == "cuda"
        host_scalar_events = []
        for event in profiler.events():
            parent = event.cpu_parent
            while parent is not None and parent.key != "clip_grad_norm_under_test":
                parent = parent.cpu_parent
            if parent is not None and event.key in {
                "aten::item",
                "aten::_local_scalar_dense",
                "aten::is_nonzero",
            }:
                host_scalar_events.append(event.key)
        assert host_scalar_events == []
    finally:
        if owns_process_group:
            torch.distributed.destroy_process_group()
