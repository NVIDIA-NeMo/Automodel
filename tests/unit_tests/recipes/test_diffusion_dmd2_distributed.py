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

"""Real two-rank CPU coverage for DMD2 collectives and checkpoint state."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from nemo_automodel.recipes.diffusion.dmd2 import DMD2DiffusionRecipe, _DMD2CheckpointState


class _TinyFakeScore(nn.Module):
    """Small module sharded like the DMD2 fake-score transformer."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.SiLU())
        self.output = nn.Linear(4, 1)
        self.register_buffer("gain", torch.tensor([1.25]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the fake-score network.

        Args:
            inputs: Tensor of shape [batch, hidden], where hidden is 4.

        Returns:
            Tensor of shape [batch, output], where output is 1. The output
            parameter's global dim 0 is intentionally smaller than the two-rank
            mesh, producing an empty ``Shard(0)`` on rank 1.
        """
        return self.output(self.block(inputs)) * self.gain


class _TinyDiscriminator(nn.Module):
    """Small replicated discriminator with both parameters and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.output = nn.Linear(3, 1)
        self.register_buffer("running_scale", torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the discriminator.

        Args:
            inputs: Tensor of shape [batch, hidden], where hidden is 3.

        Returns:
            Tensor of shape [batch, 1].
        """
        return self.output(torch.tanh(self.projection(inputs))) * self.running_scale


class _FlatEMA:
    """Minimal preallocated flat state matching ModelOpt's EMA checkpoint contract."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        """Preallocate EMA shadows.

        Args:
            state: Mapping from model state names to tensors with the corresponding
                full, unsharded parameter or buffer shapes.
        """
        self._state = {name: tensor.detach().clone() for name, tensor in state.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the preallocated shadow tensors used by DCP.

        Returns:
            Mapping from model state names to tensors with the corresponding full,
            unsharded parameter or buffer shapes.
        """
        return dict(self._state)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore each preallocated shadow tensor from DCP-loaded values.

        Args:
            state: Mapping from model state names to tensors with the corresponding
                full, unsharded parameter or buffer shapes.
        """
        if self._state.keys() != state.keys():
            raise RuntimeError(
                f"EMA state keys changed during round trip: expected {sorted(self._state)}, got {sorted(state)}."
            )
        for name, tensor in state.items():
            self._state[name].copy_(tensor)


class _RecipeCollectiveHarness:
    """Expose the DP methods used by the recipe's discriminator collectives."""

    def __init__(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator

    @staticmethod
    def _get_dp_group() -> dist.ProcessGroup:
        return dist.group.WORLD

    @staticmethod
    def _get_dp_group_size() -> int:
        return dist.get_world_size()


def _build_sharded_fake_score(seed: int) -> tuple[nn.Module, torch.optim.AdamW]:
    torch.manual_seed(seed)
    model = _TinyFakeScore()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    fully_shard(model.block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    output_weight = dict(model.named_parameters())["output.weight"]
    local_output_weight = output_weight.to_local()
    expected_local_shape = (1, 4) if dist.get_rank() == 0 else (0, 4)
    if tuple(local_output_weight.shape) != expected_local_shape:
        raise AssertionError(
            f"expected output.weight local shard {expected_local_shape} on rank {dist.get_rank()}, "
            f"got {tuple(local_output_weight.shape)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.01, foreach=False)
    return model, optimizer


def _build_replicated_discriminator(seed: int) -> tuple[nn.Module, torch.optim.AdamW]:
    torch.manual_seed(seed)
    model = _TinyDiscriminator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=0.02, foreach=False)
    return model, optimizer


def _fake_score_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    rank: int,
    step: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7.0
    inputs = inputs + rank * 0.125 + step * 0.0625
    target = torch.tensor([[0.5], [0.0], [-0.5]], dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    if not torch.isfinite(loss):
        raise AssertionError(f"rank {rank} produced a non-finite fake-score loss at step {step}: {loss}")
    loss.backward()
    optimizer.step()


def _ema_for_model(model: nn.Module, fill_base: float) -> _FlatEMA:
    state: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        state[name] = torch.full(parameter.shape, fill_base + index, dtype=torch.float32)
    for index, (name, buffer) in enumerate(model.named_buffers(), start=len(state)):
        state[name] = torch.full(buffer.shape, fill_base + index, dtype=torch.float32)
    return _FlatEMA(state)


def _as_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize one model state tensor for comparison.

    Args:
        tensor: Tensor with arbitrary shape, or a DTensor with a globally sharded
            shape and ``Shard(0)`` placement on the CPU device mesh.

    Returns:
        CPU tensor with the full global shape and no storage alias to ``tensor``.
    """
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu().clone()


def _assert_models_close(actual: nn.Module, expected: nn.Module) -> None:
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    if actual_parameters.keys() != expected_parameters.keys():
        raise AssertionError(
            f"parameter keys differ: actual={sorted(actual_parameters)}, expected={sorted(expected_parameters)}"
        )
    for name in actual_parameters:
        torch.testing.assert_close(
            _as_full_tensor(actual_parameters[name]),
            _as_full_tensor(expected_parameters[name]),
            msg=f"parameter mismatch for {name}",
        )

    actual_buffers = dict(actual.named_buffers())
    expected_buffers = dict(expected.named_buffers())
    if actual_buffers.keys() != expected_buffers.keys():
        raise AssertionError(
            f"buffer keys differ: actual={sorted(actual_buffers)}, expected={sorted(expected_buffers)}"
        )
    for name in actual_buffers:
        torch.testing.assert_close(
            _as_full_tensor(actual_buffers[name]),
            _as_full_tensor(expected_buffers[name]),
            msg=f"buffer mismatch for {name}",
        )


def _assert_ema_close(actual: _FlatEMA, expected: dict[str, torch.Tensor]) -> None:
    """Compare preallocated EMA shadows.

    Args:
        actual: EMA object whose shadows are full, unsharded tensors.
        expected: Mapping from state names to full, unsharded tensors with the
            corresponding model parameter or buffer shapes.
    """
    actual_state = actual.state_dict()
    if actual_state.keys() != expected.keys():
        raise AssertionError(f"EMA keys differ: actual={sorted(actual_state)}, expected={sorted(expected)}")
    for name in actual_state:
        torch.testing.assert_close(actual_state[name], expected[name], msg=f"EMA mismatch for {name}")


def _exercise_discriminator_collectives(
    *,
    rank: int,
) -> tuple[nn.Module, torch.optim.AdamW]:
    discriminator, optimizer = _build_replicated_discriminator(seed=4100 + rank)
    harness = _RecipeCollectiveHarness(discriminator)

    with torch.no_grad():
        for index, parameter in enumerate(discriminator.parameters()):
            parameter.fill_(rank * 10.0 + index + 1.0)
        discriminator.running_scale.fill_(rank * 10.0 + 99.0)

    DMD2DiffusionRecipe._broadcast_discriminator_state(harness, discriminator)

    for index, parameter in enumerate(discriminator.parameters()):
        torch.testing.assert_close(parameter, torch.full_like(parameter, index + 1.0))
    torch.testing.assert_close(discriminator.running_scale, torch.full_like(discriminator.running_scale, 99.0))

    for index, parameter in enumerate(discriminator.parameters()):
        parameter.grad = torch.full_like(parameter, rank * 2.0 + index + 1.0)

    DMD2DiffusionRecipe._synchronize_discriminator_gradients(harness)

    for index, parameter in enumerate(discriminator.parameters()):
        expected_mean = index + 2.0
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, expected_mean))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return discriminator, optimizer


def _continue_discriminator_step(
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(discriminator.parameters()):
        parameter.grad = torch.full_like(parameter, index + 0.375)
    optimizer.step()


def _run_dmd2_distributed_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        source_fake_score, source_fake_optimizer = _build_sharded_fake_score(seed=3100)
        _fake_score_step(source_fake_score, source_fake_optimizer, rank=rank, step=0)
        source_discriminator, source_discriminator_optimizer = _exercise_discriminator_collectives(rank=rank)
        source_ema = _ema_for_model(source_fake_score, fill_base=0.25)

        source_state = _DMD2CheckpointState(
            fake_score=source_fake_score,
            fake_score_optimizer=source_fake_optimizer,
            discriminator=source_discriminator,
            discriminator_optimizer=source_discriminator_optimizer,
            dmd_pipeline=SimpleNamespace(ema=source_ema),
            student_update_freq=5,
            cpu_offload=False,
        )
        source_state.student_update_count = 17
        expected_ema = {name: tensor.clone() for name, tensor in source_ema.state_dict().items()}

        dcp.save(
            {"dmd2": source_state},
            checkpoint_id=checkpoint_dir,
            process_group=dist.group.WORLD,
        )
        dist.barrier()

        target_fake_score, target_fake_optimizer = _build_sharded_fake_score(seed=9100)
        target_discriminator, target_discriminator_optimizer = _build_replicated_discriminator(seed=9200)
        target_ema = _ema_for_model(target_fake_score, fill_base=-4.0)
        target_state = _DMD2CheckpointState(
            fake_score=target_fake_score,
            fake_score_optimizer=target_fake_optimizer,
            discriminator=target_discriminator,
            discriminator_optimizer=target_discriminator_optimizer,
            dmd_pipeline=SimpleNamespace(ema=target_ema),
            student_update_freq=5,
            cpu_offload=False,
        )

        dcp.load(
            {"dmd2": target_state},
            checkpoint_id=checkpoint_dir,
            process_group=dist.group.WORLD,
        )

        _assert_models_close(target_fake_score, source_fake_score)
        _assert_models_close(target_discriminator, source_discriminator)
        _assert_ema_close(target_ema, expected_ema)
        if target_state.student_update_count != 17:
            raise AssertionError(
                f"student update counter was not restored: expected 17, got {target_state.student_update_count}"
            )
        if len(target_fake_optimizer.state) != len(source_fake_optimizer.state):
            raise AssertionError(
                "fake-score AdamW state was not restored into a fresh optimizer: "
                f"expected {len(source_fake_optimizer.state)} parameter states, "
                f"got {len(target_fake_optimizer.state)}"
            )
        if len(target_discriminator_optimizer.state) != len(source_discriminator_optimizer.state):
            raise AssertionError(
                "discriminator AdamW state was not restored into a fresh optimizer: "
                f"expected {len(source_discriminator_optimizer.state)} parameter states, "
                f"got {len(target_discriminator_optimizer.state)}"
            )

        _fake_score_step(source_fake_score, source_fake_optimizer, rank=rank, step=1)
        _fake_score_step(target_fake_score, target_fake_optimizer, rank=rank, step=1)
        _continue_discriminator_step(source_discriminator, source_discriminator_optimizer)
        _continue_discriminator_step(target_discriminator, target_discriminator_optimizer)

        _assert_models_close(target_fake_score, source_fake_score)
        _assert_models_close(target_discriminator, source_discriminator)

        mismatched_fake_score, mismatched_fake_optimizer = _build_sharded_fake_score(seed=9300)
        mismatched_state = _DMD2CheckpointState(
            fake_score=mismatched_fake_score,
            fake_score_optimizer=mismatched_fake_optimizer,
            discriminator=None,
            discriminator_optimizer=None,
            dmd_pipeline=SimpleNamespace(ema=None),
            student_update_freq=5,
            cpu_offload=False,
        )
        try:
            dcp.load(
                {"dmd2": mismatched_state},
                checkpoint_id=checkpoint_dir,
                process_group=dist.group.WORLD,
            )
        except ValueError as error:
            if "topology does not match" not in str(error):
                raise
        else:
            raise AssertionError("DMD2 checkpoint restore accepted a GAN/EMA topology mismatch.")

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_dmd2_two_rank_gloo_collectives_and_dcp_round_trip(tmp_path: Path) -> None:
    """Exercise DMD2's production distributed state on a real two-rank CPU mesh."""
    mp.spawn(
        _run_dmd2_distributed_worker,
        args=(
            2,
            str(tmp_path / "dmd2_gloo_init"),
            str(tmp_path / "dmd2_checkpoint"),
        ),
        nprocs=2,
        join=True,
    )
