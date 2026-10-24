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


"""Actual NCCL interleaved schedule gradient/update parity with pooled receives."""

import datetime
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.distributed.pipelining.recv_buffer_pool import install_interleaved_recv_pool

# The implementation keeps native buffers on the older prepare-infra API.
# Remove this gate when interleaved pooling supports that allocation layout.
pytestmark = pytest.mark.skipif(
    not all(hasattr(PipelineStage, name) for name in ("_setup_forward_recv_info", "_setup_backward_recv_info")),
    reason="Interleaved pooling requires PyTorch per-direction stage setup",
)


class _Block(nn.Module):
    def __init__(self, activation_checkpointing: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 32)
        self.activation_checkpointing = activation_checkpointing

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Stage input of shape [microbatch, hidden].

        Returns:
            Stage output with the same shape and dtype as x.
        """
        return x + self.linear2(torch.nn.functional.silu(self.linear1(x))) * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block with optional activation recomputation.

        Args:
            x: Stage input of shape [microbatch, hidden].

        Returns:
            Stage output with the same shape and dtype as x.
        """
        return checkpoint(self._core, x, use_reentrant=False) if self.activation_checkpointing else self._core(x)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_pipeline(
    rank: int,
    world_size: int,
    virtual: int,
    dtype: torch.dtype,
    activation_checkpointing: bool,
    pooled: bool,
    unsafe_forward_alias: bool = False,
    pool_slack: int = 2,
    num_microbatches: int = 16,
) -> list[list[torch.Tensor]]:
    device = torch.device("cuda", rank)
    torch.manual_seed(911)
    all_blocks = [_Block(activation_checkpointing).to(device=device, dtype=dtype) for _ in range(world_size * virtual)]
    local_indices = list(range(rank, world_size * virtual, world_size))
    blocks = [all_blocks[index] for index in local_indices]
    stages = [PipelineStage(block, index, world_size * virtual, device) for index, block in zip(local_indices, blocks)]

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the microbatch mean squared error.

        Args:
            output: Predicted values of shape [microbatch, hidden].
            target: Reference values of shape [microbatch, hidden].

        Returns:
            Scalar FP32 loss.
        """
        return (output.float() - target.float()).square().mean()

    schedule = ScheduleInterleaved1F1B(stages, n_microbatches=num_microbatches, loss_fn=loss_fn, scale_grads=False)
    if pooled:
        install_interleaved_recv_pool(schedule, slack=pool_slack)
        if unsafe_forward_alias:
            # Test-only defect: leave receives pooled but remove their owned input copy.
            for stage in stages:
                if not stage.is_first:
                    stage._retrieve_recv_activations = PipelineStage._retrieve_recv_activations.__get__(
                        stage, PipelineStage
                    )
    parameters = [parameter for block in blocks for parameter in block.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.001)
    rng = torch.Generator().manual_seed(912)
    observations = []
    for _ in range(3):
        x = torch.randn(2 * num_microbatches, 32, generator=rng).to(device=device, dtype=dtype).requires_grad_()
        target = torch.randn(2 * num_microbatches, 32, generator=rng).to(device=device, dtype=dtype)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        if rank == 0:
            schedule.step(x)
        elif rank == world_size - 1:
            schedule.step(target=target, losses=losses)
        else:
            schedule.step()
        assert all(parameter.grad is not None for parameter in parameters)
        snapshot = [parameter.grad.detach().cpu().clone() for parameter in parameters]
        if rank == 0:
            assert x.grad is not None
            snapshot.append(x.grad.detach().cpu().clone())
        snapshot.extend(loss.detach().cpu().clone() for loss in losses)
        optimizer.step()
        snapshot.extend(parameter.detach().cpu().clone() for parameter in parameters)
        assert all(torch.isfinite(tensor).all() for tensor in snapshot)
        observations.append(snapshot)
        for stage in stages:
            assert stage.chunks == num_microbatches
            if not stage.is_first:
                distinct = len({id(info) for info in stage.args_recv_info.values()})
                assert distinct < num_microbatches if pooled else distinct == num_microbatches
            if not stage.is_last:
                distinct = len({id(info) for info in stage.grad_recv_info.values()})
                assert distinct < num_microbatches if pooled else distinct == num_microbatches
    return observations


def _worker(
    rank: int,
    world_size: int,
    port: int,
    virtual: int,
    dtype_name: str,
    activation_checkpointing: bool,
    unsafe_forward_alias: bool = False,
    pool_slack: int = 2,
    num_microbatches: int = 16,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=180))
    try:
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
        reference = _run_pipeline(
            rank, world_size, virtual, dtype, activation_checkpointing, False, num_microbatches=num_microbatches
        )
        actual = _run_pipeline(
            rank,
            world_size,
            virtual,
            dtype,
            activation_checkpointing,
            True,
            unsafe_forward_alias,
            pool_slack,
            num_microbatches,
        )
        assert len(reference) == len(actual) == 3
        for reference_step, actual_step in zip(reference, actual):
            assert len(reference_step) == len(actual_step)
            for expected, observed in zip(reference_step, actual_step):
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize(
    "world_size,virtual,dtype_name,activation_checkpointing,pool_slack,num_microbatches",
    [
        (2, 2, "float32", False, 2, 16),
        (2, 2, "bfloat16", True, 2, 16),
        (2, 4, "bfloat16", False, 2, 16),
        (2, 4, "bfloat16", True, 2, 16),
        (4, 4, "bfloat16", True, 2, 16),
        (4, 4, "bfloat16", True, 0, 16),
        (4, 2, "float32", False, 2, 32),
        (4, 2, "bfloat16", True, 2, 32),
        (4, 2, "bfloat16", True, 2, 64),
    ],
)
@pytest.mark.timeout(240)
def test_interleaved_pooled_receives_preserve_every_gradient_and_update(
    world_size: int,
    virtual: int,
    dtype_name: str,
    activation_checkpointing: bool,
    pool_slack: int,
    num_microbatches: int,
) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices required")
    mp.spawn(
        _worker,
        args=(
            world_size,
            _free_port(),
            virtual,
            dtype_name,
            activation_checkpointing,
            False,
            pool_slack,
            num_microbatches,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Four CUDA devices required")
@pytest.mark.timeout(240)
@pytest.mark.parametrize("virtual,num_microbatches", [(4, 16), (2, 32), (2, 64)])
def test_omitting_owned_forward_inputs_fails_the_checkpointed_parity_check(virtual: int, num_microbatches: int) -> None:
    # A version-counter error or numerical mismatch is evidence of the storage
    # defect; unrelated initialization/communication failures must not pass.
    with pytest.raises(
        mp.ProcessRaisedException,
        match="not equal|not close|modified by an inplace operation|backward through the graph a second time",
    ):
        mp.spawn(
            _worker, args=(4, _free_port(), virtual, "bfloat16", True, True, 0, num_microbatches), nprocs=4, join=True
        )
