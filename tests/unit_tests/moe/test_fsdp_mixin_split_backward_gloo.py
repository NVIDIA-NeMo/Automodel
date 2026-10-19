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

"""Real two-rank FSDP coverage for ``patched_backward_maybe_with_nosync`` on MoE stages.

The mocked tests in ``test_fsdp_mixin.py`` check which FSDP flags the patch sets; they
cannot see the parameter lifecycle that FSDP drives from those flags. These tests run
real FSDP2 on two gloo ranks and replay pipeline schedules action by action, exactly as
the runtime dispatches them: ``F`` forward, ``B`` full backward, ``I``/``W`` the split
input/weight backward of zero-bubble schedules, including mixed full/split orders and
interleaved split orders under activation checkpointing, with
``distributed.defer_fsdp_grad_sync`` both ``false`` (reduce per micro-batch) and ``true``
(deferred). Gradients are compared with an unsharded single-process reference.
"""

from __future__ import annotations

import copy
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe import fsdp_mixin
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin, patched_backward_maybe_with_nosync

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
pytestmark = pytest.mark.timeout(120)

_HIDDEN = 8
_MICROBATCH = 4

# Each schedule is replayed with every "last" flag derived from the position of the final B/W action.
_SCHEDULES: dict[str, list[str]] = {
    "full": ["F0", "B0", "F1", "B1", "F2", "B2"],
    "split_queued": ["F0", "I0", "F1", "I1", "F2", "I2", "W0", "W1", "W2"],
    "split_interleaved": ["F0", "I0", "F1", "I1", "W0", "F2", "I2", "W1", "W2"],
    "mixed_full_split": ["F0", "I0", "F1", "B1", "W0"],
}


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(_HIDDEN, _HIDDEN)
        # ``_iter_fsdp_modules`` finds experts at ``block.mlp.experts``.
        self.mlp = nn.Module()
        self.mlp.experts = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply one dense + one expert-proxy linear layer.

        Args:
            hidden: Activations of shape [tokens, hidden].

        Returns:
            Activations of shape [tokens, hidden].
        """
        return self.mlp.experts(torch.tanh(self.attn(hidden)))


class _ToyBackbone(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_ToyBlock() for _ in range(num_layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the decoder blocks.

        Args:
            hidden: Activations of shape [tokens, hidden].

        Returns:
            Activations of shape [tokens, hidden].
        """
        for block in self.layers:
            hidden = block(hidden)
        return hidden


class _ToyMoEStage(MoEFSDPSyncMixin, nn.Module):
    """A pipeline-stage model chunk with the module layout the mixin walks (``model.layers[*].mlp.experts``, ``lm_head``)."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = _ToyBackbone(num_layers)
        self.lm_head = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the stage.

        Args:
            hidden: Stage input of shape [tokens, hidden].

        Returns:
            Stage output of shape [tokens, hidden].
        """
        return self.lm_head(self.model(hidden))


def _shard(model: _ToyMoEStage, mesh: DeviceMesh, *, activation_checkpointing: bool) -> None:
    """Wrap the stage the way ``moe/parallelizer`` does: experts per layer, then the backbone and the head.

    Pipeline stages keep their parameters unsharded between forward and backward
    (``reshard_after_forward=False``): the schedules capture ``submod.parameters()`` before the
    backward starts, so a split backward can only find the parameters that took part in the forward
    if they are still the unsharded objects. With ``activation_checkpointing`` every block is wrapped in
    PyTorch's non-reentrant ``checkpoint_wrapper`` before sharding, so backwards recompute the block
    forward and FSDP's forward hooks run again inside ``stage_backward_weight``.
    """
    for index, block in enumerate(model.model.layers):
        fully_shard(block.mlp.experts, mesh=mesh, reshard_after_forward=False)
        if activation_checkpointing:
            model.model.layers[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    fully_shard(model.model, mesh=mesh, reshard_after_forward=False)
    fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=False)


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a possibly sharded tensor for comparison.

    Args:
        tensor: Tensor of arbitrary shape, optionally a DTensor.

    Returns:
        Replicated tensor with the global shape and values of ``tensor``.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _microbatches(rank: int, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic (input, target) pairs; every rank sees a different slice, like data parallelism."""
    generator = torch.Generator().manual_seed(1000 + rank)
    return [
        (
            torch.randn(_MICROBATCH, _HIDDEN, generator=generator),
            torch.randn(_MICROBATCH, _HIDDEN, generator=generator),
        )
        for _ in range(count)
    ]


def _reference_grads(reference: _ToyMoEStage, world_size: int, count: int) -> dict[str, torch.Tensor]:
    """Gradients of the data-parallel loss on an unsharded copy: sum over micro-batches, mean over ranks."""
    reference.zero_grad(set_to_none=True)
    for rank in range(world_size):
        for inputs, target in _microbatches(rank, count):
            loss = F.mse_loss(reference(inputs), target) / world_size
            loss.backward()
    return {name: param.grad.detach().clone() for name, param in reference.named_parameters()}


def _replay(model: _ToyMoEStage, *, reduce_grad_per_microbatch: bool, actions: list[str], rank: int) -> None:
    """Replay a pipeline schedule through ``patched_backward_maybe_with_nosync`` as the runtime dispatches it.

    ``F<i>`` runs the forward of micro-batch ``i``; ``B<i>`` its full backward; ``I<i>``/``W<i>`` the split
    input and weight backwards. ``last_backward`` is set on the final ``B``/``W`` action, as the schedule does.
    """
    stage = SimpleNamespace(submod=model, _reduce_grad_per_microbatch=reduce_grad_per_microbatch)
    last_action = max(index for index, action in enumerate(actions) if action[0] in "BW")
    count = sum(action[0] == "F" for action in actions)
    data = _microbatches(rank, count)
    state: dict[int, dict] = {}
    for index, action in enumerate(actions):
        kind, microbatch = action[0], int(action[1:])
        last = index == last_action
        if kind == "F":
            inputs, target = data[microbatch]
            inputs = inputs.clone().requires_grad_(True)  # a non-first stage receives activations that need grads
            output = model(inputs)
            detached = output.detach().requires_grad_(True)
            (output_grads,) = torch.autograd.grad(F.mse_loss(detached, target), detached)
            # The schedules hand the stage helpers lists of tensors (``stage_backward_input`` detaches each in place).
            state[microbatch] = {"stage_output": [output], "output_grads": [output_grads], "input_values": [inputs]}
        elif kind == "B":
            patched_backward_maybe_with_nosync(stage, "full", state[microbatch], last_backward=last)
        elif kind == "I":
            ids_before = [id(param) for param in model.parameters()]
            _, param_groups = patched_backward_maybe_with_nosync(stage, "input", state[microbatch], last_backward=False)
            assert [id(param) for param in model.parameters()] == ids_before, (
                "the input backward must leave the unsharded parameter objects in place for the weight backward"
            )
            assert state[microbatch]["input_values"][0].grad is not None, "the input backward must produce dInput"
            state[microbatch]["param_groups"] = param_groups
        else:
            patched_backward_maybe_with_nosync(stage, "weight", state[microbatch], last_backward=last)


def _assert_matches_reference(model: _ToyMoEStage, expected: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        name = name.replace("_checkpoint_wrapped_module.", "")  # checkpoint_wrapper adds a prefix to parameter names
        assert isinstance(param, DTensor), f"{name} must be resharded after the last backward"
        assert param.grad is not None, f"{name} has no gradient after the last backward"
        torch.testing.assert_close(_full(param.grad), expected[name], rtol=1e-5, atol=1e-6)


def _run_case(
    mesh: DeviceMesh,
    rank: int,
    world_size: int,
    *,
    reduce_grad_per_microbatch: bool,
    schedule: str,
    activation_checkpointing: bool,
) -> None:
    torch.manual_seed(2026)
    model = _ToyMoEStage()
    reference = copy.deepcopy(model)
    _shard(model, mesh, activation_checkpointing=activation_checkpointing)
    actions = _SCHEDULES[schedule]
    expected = _reference_grads(reference, world_size, sum(action[0] == "F" for action in actions))
    try:
        _replay(model, reduce_grad_per_microbatch=reduce_grad_per_microbatch, actions=actions, rank=rank)
        _assert_matches_reference(model, expected)
    except Exception as exc:  # noqa: BLE001 - re-raise with the case so the spawn error names it
        raise AssertionError(
            f"case reduce_grad_per_microbatch={reduce_grad_per_microbatch} schedule={schedule} "
            f"activation_checkpointing={activation_checkpointing}: {type(exc).__name__}: {exc}"
        ) from exc


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # The deferred branch reduces only on the optimizer step; make every "last backward" one.
    fsdp_mixin.get_is_optim_step = lambda: True
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        for reduce_grad_per_microbatch in (True, False):
            for schedule in _SCHEDULES:
                for activation_checkpointing in (False, True):
                    _run_case(
                        mesh,
                        rank,
                        world_size,
                        reduce_grad_per_microbatch=reduce_grad_per_microbatch,
                        schedule=schedule,
                        activation_checkpointing=activation_checkpointing,
                    )
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("world_size", [2])
def test_moe_stage_backward_matches_unsharded_reference(world_size: int) -> None:
    """Full, split and mixed pipeline backwards, deferred and per-micro-batch, with and without AC, on real two-rank FSDP."""
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
