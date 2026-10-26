# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank gradient parity for CP FLCE and rank-asymmetric FSDP graphs."""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.distributed.parallelizer_utils import configure_fsdp_unused_param_reduction
from nemo_automodel.components.loss.linear_ce import HAVE_CUT_CROSS_ENTROPY, FusedLinearCrossEntropy

_RESULT_PREFIX = "CP_FLCE_FSDP_CORRECTNESS_RESULT "


def _full_grad(parameter: nn.Parameter) -> torch.Tensor:
    """Return a full regular tensor for a sharded or regular parameter gradient."""
    grad = parameter.grad
    if grad is None:
        raise AssertionError("expected parameter gradient, got None")
    if isinstance(grad, DTensor):
        grad = grad.full_tensor()
    return grad.detach().float()


def _assert_flce_weight_grad_parity(device: torch.device) -> None:
    """Compare sharded rank-local FLCE against a full-batch reference."""
    if not HAVE_CUT_CROSS_ENTROPY:
        raise RuntimeError("cut-cross-entropy is required for the CP FLCE functional test")

    world = dist.get_world_size()
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_cp",))
    dtype = torch.bfloat16
    vocab_size, hidden_size, local_tokens = 64, 32, 8

    torch.manual_seed(1234)
    full_weight_data = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    weight = nn.Parameter(distribute_tensor(full_weight_data.clone(), mesh, [Shard(0)]))

    all_hidden = []
    all_labels = []
    for data_rank in range(world):
        generator = torch.Generator(device=device).manual_seed(9000 + data_rank)
        all_hidden.append(torch.randn(local_tokens, hidden_size, device=device, dtype=dtype, generator=generator))
        all_labels.append(
            torch.randint(0, vocab_size, (local_tokens,), device=device, generator=generator, dtype=torch.long)
        )

    loss_fn = FusedLinearCrossEntropy(reduction="sum")
    local_hidden = all_hidden[rank].clone().requires_grad_(True)
    local_loss = loss_fn(
        local_hidden,
        all_labels[rank],
        weight,
        num_label_tokens=world * local_tokens,
        grad_reduce_group=dist.group.WORLD,
    )
    (local_loss * world).backward()

    reference_weight = full_weight_data.clone().requires_grad_(True)
    reference_loss = loss_fn(
        torch.cat(all_hidden),
        torch.cat(all_labels),
        reference_weight,
        num_label_tokens=world * local_tokens,
    )
    reference_loss.backward()

    actual_local_grad = weight.grad.to_local().float()
    expected_local_grad = reference_weight.grad.chunk(world, dim=0)[rank].float()
    torch.testing.assert_close(actual_local_grad, expected_local_grad, rtol=2e-2, atol=2e-2)


class _ConditionalFSDPModel(nn.Module):
    """Tiny model with one rank-conditional parameter-owning branch."""

    def __init__(self) -> None:
        super().__init__()
        self.conditional = nn.Linear(8, 8, bias=False)
        self.output = nn.Linear(8, 1, bias=False)

    def forward(self, inputs: torch.Tensor, *, use_conditional: bool) -> torch.Tensor:
        """Run the conditional branch before the always-used output layer."""
        hidden = self.conditional(inputs) if use_conditional else inputs
        return self.output(hidden)


def _assert_empty_rank_fsdp_grad_parity(device: torch.device) -> None:
    """Compare a rank-skipped FSDP unit against zero-contribution reference math."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_cp",))

    torch.manual_seed(4321)
    model = _ConditionalFSDPModel().to(device)
    reference = _ConditionalFSDPModel().to(device)
    reference.load_state_dict(model.state_dict())

    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    configured_units = configure_fsdp_unused_param_reduction(model)
    assert configured_units >= 1

    inputs = []
    for data_rank in range(world):
        generator = torch.Generator(device=device).manual_seed(7000 + data_rank)
        inputs.append(torch.randn(2, 3, 8, device=device, generator=generator))

    use_conditional = rank == 0
    model(inputs[rank], use_conditional=use_conditional).sum().backward()

    reference_loss = (
        sum(reference(batch, use_conditional=data_rank == 0).sum() for data_rank, batch in enumerate(inputs)) / world
    )
    reference_loss.backward()

    torch.testing.assert_close(
        _full_grad(model.conditional.weight),
        reference.conditional.weight.grad.float(),
        rtol=1e-5,
        atol=1e-6,
    )


class _PartiallyUsedFSDPModel(nn.Module):
    """One FSDP unit whose second branch never runs, so it never gets a local gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(8, 8, bias=False)
        self.unused = nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run only the `used` branch.

        Args:
            inputs: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.used(inputs)


def _assert_accumulated_unused_param_grad_dtype(device: torch.device) -> None:
    """Reduce one dtype when CP zero-fill meets mixed-precision gradient accumulation.

    Regression for NVBug 6599894: `used` carries an fp32 accumulation from the
    first micro-batch while the CP unused-parameter fill contributes a bf16 zero
    for `unused`, and FSDP2's `foreach_reduce` rejects a mixed-dtype group with
    "FSDP reduce-scatter expects uniform gradient dtype".
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_cp",))

    torch.manual_seed(6599894)
    model = _PartiallyUsedFSDPModel().to(device)
    reference = _PartiallyUsedFSDPModel().to(device)
    reference.load_state_dict(model.state_dict())

    fully_shard(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        reshard_after_forward=True,
    )
    configured_units = configure_fsdp_unused_param_reduction(model)
    assert configured_units >= 1

    def _micro_batch(micro_index: int, data_rank: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(5000 + 17 * micro_index + data_rank)
        return torch.randn(2, 3, 8, device=device, generator=generator)

    micro_batches = 2
    for micro_index in range(micro_batches):
        # Defer reduction to the final micro-batch, as fsdp_mixin does.
        is_final = micro_index == micro_batches - 1
        model.set_is_last_backward(is_final)
        model.set_reshard_after_backward(is_final)
        model.set_requires_gradient_sync(is_final)
        model(_micro_batch(micro_index, rank)).float().sum().backward()

    # FSDP2 computes in param dtype and averages the reduction over the mesh, so
    # mirror both to keep the reference comparable.
    reference_bf16 = reference.to(torch.bfloat16)
    expected_used = torch.zeros_like(reference_bf16.used.weight, dtype=torch.float32)
    for micro_index in range(micro_batches):
        for data_rank in range(world):
            (grad,) = torch.autograd.grad(
                reference_bf16(_micro_batch(micro_index, data_rank).to(torch.bfloat16)).float().sum(),
                reference_bf16.used.weight,
            )
            expected_used += grad.float()
    expected_used /= world

    # No rank used `unused`, so its reduced zero is dropped and the optimizer skips it as without CP.
    assert model.unused.weight.grad is None
    torch.testing.assert_close(_full_grad(model.used.weight), expected_used, rtol=2e-2, atol=2e-2)


def _assert_globally_unused_param_keeps_none_grad(device: torch.device) -> None:
    """A parameter no rank used in a window keeps ``grad is None``, so AdamW leaves it and its state alone.

    Reduction runs after every micro-batch, the default without deferred gradient sync: a
    micro-batch in which no rank runs the conditional branch creates a reduced zero, which must
    not turn into an AdamW step (decoupled weight decay alone would move the weight), while a
    branch that one rank runs in another micro-batch of the same window keeps its gradient.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp_cp",))

    torch.manual_seed(2468)
    model = _ConditionalFSDPModel().to(device)
    reference = _ConditionalFSDPModel().to(device)
    reference.load_state_dict(model.state_dict())
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    assert configure_fsdp_unused_param_reduction(model) >= 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

    def _inputs(window: int, micro_index: int, data_rank: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(8000 + 100 * window + 10 * micro_index + data_rank)
        return torch.randn(2, 3, 8, device=device, generator=generator)

    def _run_window(window: int, users_per_micro_batch: list[tuple[int, ...]]) -> None:
        for micro_index, users in enumerate(users_per_micro_batch):
            model(_inputs(window, micro_index, rank), use_conditional=rank in users).sum().backward()

    # Only rank 0 runs the branch, in the second micro-batch: the restore after the first micro-batch
    # must not stop the second micro-batch's contribution from arriving intact.
    _run_window(0, [(), (0,)])
    reference(_inputs(0, 1, 0), use_conditional=True).sum().div(world).backward()
    torch.testing.assert_close(
        _full_grad(model.conditional.weight), reference.conditional.weight.grad.float(), rtol=1e-5, atol=1e-6
    )
    optimizer.step()
    optimizer.zero_grad()

    # A contribution from the first micro-batch is kept through a later micro-batch nobody used it in.
    _run_window(1, [(0,), ()])
    assert model.conditional.weight.grad is not None
    optimizer.step()
    optimizer.zero_grad()

    # Compare full tensors: ``output.weight`` is [1, 8], so on two ranks one local shard is empty and
    # a local ``torch.equal`` holds vacuously there, whatever the optimizer did.
    conditional_before = model.conditional.weight.full_tensor().clone()
    output_before = model.output.weight.full_tensor().clone()
    conditional_step = optimizer.state[model.conditional.weight]["step"].clone()
    _run_window(2, [(), ()])
    assert model.conditional.weight.grad is None
    optimizer.step()
    optimizer.zero_grad()
    assert torch.equal(model.conditional.weight.full_tensor(), conditional_before)
    assert torch.equal(optimizer.state[model.conditional.weight]["step"], conditional_step)
    assert not torch.equal(model.output.weight.full_tensor(), output_before)


def _run_worker() -> None:
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"This regression requires exactly 2 ranks, got {dist.get_world_size()}.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        _assert_flce_weight_grad_parity(device)
        dist.barrier()
        _assert_empty_rank_fsdp_grad_parity(device)
        dist.barrier()
        _assert_accumulated_unused_param_grad_dtype(device)
        dist.barrier()
        _assert_globally_unused_param_keeps_none_grad(device)
        dist.barrier()

        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + "PASS", flush=True)
    except Exception:
        # Surface the failure and exit at once: destroy_process_group would block behind the peer
        # rank's pending collective and the parent would only ever see a timeout.
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
def test_cp_flce_and_empty_rank_fsdp_two_rank_gradient_parity() -> None:
    """FLCE and rank-asymmetric FSDP gradients match full references on two GPUs."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible.split(",")[:2]) if visible else "0,1"
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,  # a cold cache compiles the fused-CE kernels first
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"").decode(errors="replace")
        pytest.fail(f"worker timed out after {error.timeout}s:\n{output}")
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    assert _RESULT_PREFIX + "PASS" in completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    if _parse_args().worker:
        _run_worker()
