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

"""Real FSDP output preservation using the native Qwen3-Next FP32 SSM gate.

This is a boundary-test fixture, not Qwen3-Next's full forward: the real model
feeds its FP32 decay gate into a recurrence. Here an explicit activation cast
feeds a BF16 projection while the original gate remains a separate FP32 output.
Both dtype-group orderings and activation-checkpoint recomputation are exercised.
"""

import copy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype


class _GateAndProjection(nn.Module):
    """Exercise a real FP32 gate with model-owned BF16 activation consumption."""

    def __init__(self, *, fp32_parent: bool) -> None:
        from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextSSMGate

        super().__init__()
        self._fp32_params = Qwen3NextSSMGate(4)
        with torch.no_grad():
            self._fp32_params.A_log.copy_(torch.linspace(-0.7, 0.2, 4))
            self._fp32_params.dt_bias.copy_(torch.linspace(-0.3, 0.4, 4))
        # Group selection counts parameter tensors: the gate has two, while the
        # BF16 group has one or three. This deliberately avoids a tie.
        self.projections = nn.ModuleList(
            nn.Linear(4, 4, bias=False, dtype=torch.bfloat16) for _ in range(1 if fp32_parent else 3)
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the gate while converting only the projection activation.

        Args:
            inputs: BF16 tensor of shape [batch, sequence, hidden], where hidden
                is 4 and also serves as the gate's SSM head axis in this fixture.

        Returns:
            BF16 activation and FP32 gate, each of shape [batch, sequence, hidden].
            Neither output aliases or mutates inputs.
        """
        assert inputs.dtype == torch.bfloat16
        assert self._fp32_params.A_log.dtype == self._fp32_params.dt_bias.dtype == torch.float32
        gate = self._fp32_params(inputs)
        assert gate.dtype == torch.float32
        activation = inputs + gate.to(inputs.dtype)
        for projection in self.projections:
            assert projection.weight.dtype == torch.bfloat16
            activation = projection(activation)
        return activation, gate


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    """Gather an FSDP tensor for comparison with its unsharded reference.

    Args:
        value: Tensor of a parameter's arbitrary global shape, or a DTensor with
            that global shape sharded on axis 0 across the two-rank FSDP mesh.

    Returns:
        Tensor with the global shape and original dtype, replicated on each rank.
        An ordinary input is returned unchanged.
    """
    return value.full_tensor() if isinstance(value, DTensor) else value


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        policy = FSDP2Config().mp_policy
        assert policy.param_dtype == torch.bfloat16
        assert policy.cast_forward_inputs is False
        assert policy.output_dtype is None
        for fp32_parent in (False, True):
            for checkpoint in (False, True):
                torch.manual_seed(4150)
                model = _GateAndProjection(fp32_parent=fp32_parent).to(device)
                reference = copy.deepcopy(model)
                legacy = copy.deepcopy(model) if not fp32_parent else None
                gate, projection = model._fp32_params, model.projections[0]
                if checkpoint:
                    model = checkpoint_wrapper(model)
                fully_shard_by_dtype(
                    model,
                    mesh=mesh,
                    mp_policy=policy,
                    offload_policy=None,
                    fp32_compute_module_names=("_fp32_params",),
                )
                assert isinstance(model, FSDPModule)
                assert isinstance(gate, FSDPModule) is (not fp32_parent)
                assert isinstance(projection, FSDPModule) is fp32_parent
                inputs = torch.linspace(-0.9, 0.8, 24, device=device, dtype=torch.bfloat16).reshape(2, 3, 4)
                inputs.requires_grad_()
                reference_inputs = inputs.detach().clone().requires_grad_()
                actual = model(inputs)
                expected = reference(reference_inputs)
                assert actual[0].dtype == torch.bfloat16
                assert actual[1].dtype == torch.float32
                for output, reference_output in zip(actual, expected):
                    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
                # Nonuniform cotangents exercise every projection and both gate
                # parameters, including the FP32 auxiliary's separate gradient.
                activation_probe = torch.randn_like(actual[0], dtype=torch.float32)
                gate_probe = torch.randn_like(actual[1])
                for output in (actual, expected):
                    loss = (output[0].float() * activation_probe).sum() + (output[1] * gate_probe).sum()
                    loss.backward()
                assert inputs.grad is not None and reference_inputs.grad is not None
                assert torch.isfinite(inputs.grad).all() and inputs.grad.count_nonzero() > 0
                torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                actual_parameters = {
                    name.replace("_checkpoint_wrapped_module.", ""): parameter
                    for name, parameter in model.named_parameters()
                }
                reference_parameters = dict(reference.named_parameters())
                assert actual_parameters.keys() == reference_parameters.keys()
                for name, parameter in actual_parameters.items():
                    gradient = parameter.grad
                    reference_gradient = reference_parameters[name].grad
                    assert gradient is not None and reference_gradient is not None, name
                    gradient = _full_tensor(gradient)
                    assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0, name
                    torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0, msg=name)
                # Use the same optimizer implementation on both layouts rather
                # than a manually expanded update with different rounding.
                torch.optim.SGD(model.parameters(), lr=0.03125).step()
                torch.optim.SGD(reference.parameters(), lr=0.03125).step()
                for name, parameter in actual_parameters.items():
                    torch.testing.assert_close(
                        _full_tensor(parameter), reference_parameters[name], rtol=0, atol=0, msg=name
                    )
                if legacy is not None:
                    if checkpoint:
                        legacy = checkpoint_wrapper(legacy)
                    fully_shard_by_dtype(
                        legacy,
                        mesh=mesh,
                        mp_policy=replace(policy, output_dtype=torch.bfloat16),
                        offload_policy=None,
                        fp32_compute_module_names=("_fp32_params",),
                    )
                    with torch.no_grad():
                        legacy_output = legacy(inputs.detach())
                    assert legacy_output[1].dtype == torch.bfloat16
                    torch.testing.assert_close(legacy_output[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(legacy_output[1], expected[1].bfloat16(), rtol=0, atol=0)
                    assert not torch.equal(legacy_output[1].float(), expected[1])
                if rank == 0:
                    print(
                        f"fp32_parent={fp32_parent}, checkpoint={checkpoint}: "
                        "mixed output dtypes, outputs, input/parameter gradients and SGD step match reference",
                        flush=True,
                    )
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices for actual FSDP dtype groups")
def test_fsdp_preserves_model_owned_output_dtypes(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "output_dtypes"),), nprocs=2, join=True)
