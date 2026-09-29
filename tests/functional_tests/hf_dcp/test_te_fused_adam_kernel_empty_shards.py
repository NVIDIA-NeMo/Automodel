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

"""Real TE kernel regression; requires CUDA and two GPUs for DTensor cases.

Run on an affected TE release (e.g. 2.15). TE 2.19+ filters empty tensors
upstream, so passing only on a newer release does not validate the backport.
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.shared.import_utils import safe_import, safe_import_te
from nemo_automodel.shared.te_patches import apply_te_patches


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the underlying local tensor without changing its storage.

    Args:
        tensor: Tensor of shape [elements], or DTensor of global shape [elements]
            sharded on axis 0; a rank's local shape may be [0].

    Returns:
        Tensor of shape [local_elements], aliasing the input's local storage.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _optimizer_parity_worker(
    rank: int, world_size: int, rendezvous: str, master_weights: bool, use_dtensor: bool
) -> None:
    """Compare real TE updates with and without an empty local parameter."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    have_te, _ = safe_import_te()
    assert have_te, "Real Transformer Engine PyTorch extension is required"
    available, fused_adam = safe_import("transformer_engine.pytorch.optimizers.fused_adam")
    assert available
    original_applier = fused_adam.multi_tensor_applier
    try:
        mesh = None
        if use_dtensor:
            dist.init_process_group(
                "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
            )
            mesh = init_device_mesh("cuda", (world_size,))

        depth = 5 if master_weights else 4
        # TE <= 2.18 metadata capacities for four/five aligned tensor lists.
        # Empty at the final slot, followed by nonempty tensors, triggers the
        # overflow. Ending the list at the empty slot does not reproduce it.
        capacity = 30 if master_weights else 36
        empty_index = capacity - 1
        dtype = torch.bfloat16 if master_weights else torch.float32
        parameters = []
        for index in range(capacity + 4):
            size = (1 if use_dtensor else 0) if index == empty_index else 4
            value = torch.arange(size, device=device, dtype=torch.float32).add_(index / 16).to(dtype)
            if use_dtensor:
                value = distribute_tensor(value, mesh, [Shard(0)])
            parameters.append(torch.nn.Parameter(value))
        nonempty = [index for index, parameter in enumerate(parameters) if _local(parameter).numel()]
        reference_parameters = [torch.nn.Parameter(parameters[index].detach().clone()) for index in nonempty]
        empty_parameters = [parameter for parameter in parameters if not _local(parameter).numel()]
        if not use_dtensor or rank == 1:
            assert len(empty_parameters) == 1
            assert _local(parameters[empty_index]).numel() == 0
            assert all(_local(p).numel() for p in parameters[empty_index + 1 :])
        if use_dtensor:
            assert isinstance(parameters[empty_index], DTensor)
            assert parameters[empty_index].numel() == 1

        # Observe the real native boundary, never replace it with a mock.
        native_depths = []

        def record_native_call(op, noop_flag_buffer, tensor_lists, *args):
            """Record list depth while forwarding actual tensor storage to TE.

            Args:
                op: Native optimizer kernel.
                noop_flag_buffer: Device integer tensor of shape [1].
                tensor_lists: Role-major [role][parameter] lists of tensors with
                    shape [local_elements], or axis-0-sharded DTensors with global
                    shape [elements]. Native op updates the existing storage.
                *args: Scalar native-kernel arguments forwarded unchanged.

            Returns:
                The native applier's return value, without altering its updates.
            """
            native_depths.append(len(tensor_lists))
            return original_applier(op, noop_flag_buffer, tensor_lists, *args)

        fused_adam.multi_tensor_applier = record_native_call
        apply_te_patches()
        patched_applier = fused_adam.multi_tensor_applier
        options = dict(lr=1e-2, weight_decay=0.01, master_weights=master_weights)
        # Construct directly: FusedAdamConfig would drop empties before this test.
        optimizer = fused_adam.FusedAdam(parameters, **options)
        reference = fused_adam.FusedAdam(reference_parameters, **options)
        initial_values = [_local(parameter).detach().clone() for parameter in reference_parameters]
        for step in range(3):
            for index, parameter in enumerate(parameters):
                parameter.grad = torch.full_like(parameter, (index + step + 1) / 16)
            for index, parameter in zip(nonempty, reference_parameters, strict=True):
                parameter.grad = parameters[index].grad.detach().clone()
            fused_adam.multi_tensor_applier = patched_applier
            optimizer.step()
            torch.cuda.synchronize()
            # The empty-free reference uses the original unwrapped TE path.
            fused_adam.multi_tensor_applier = original_applier
            reference.step()
            torch.cuda.synchronize()
            for index, expected in zip(nonempty, reference_parameters, strict=True):
                actual = parameters[index]
                torch.testing.assert_close(_local(actual), _local(expected), rtol=0, atol=0)
                assert optimizer.state[actual].keys() == reference.state[expected].keys()
                for key in reference.state[expected]:
                    torch.testing.assert_close(
                        _local(optimizer.state[actual][key]), _local(reference.state[expected][key]), rtol=0, atol=0
                    )
            assert optimizer.param_groups[0]["step"] == reference.param_groups[0]["step"] == step + 1
            assert len(optimizer.param_groups[0]["params"]) == len(parameters)
            assert len(optimizer.state) == len(parameters)
            for empty in empty_parameters:
                assert any(p is empty for p in optimizer.param_groups[0]["params"])
                assert {"exp_avg", "exp_avg_sq"} <= optimizer.state[empty].keys()
                if master_weights:
                    assert "master_param" in optimizer.state[empty]
                assert all(_local(value).numel() == 0 for value in optimizer.state[empty].values())
            state_dict = optimizer.state_dict()
            assert len(state_dict["state"]) == len(parameters)
            assert len(state_dict["param_groups"][0]["params"]) == len(parameters)
        assert native_depths == [depth] * 3
        assert any(
            not torch.equal(_local(parameter), initial)
            for parameter, initial in zip(reference_parameters, initial_values, strict=True)
        )
    finally:
        fused_adam.multi_tensor_applier = original_applier
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("use_dtensor", [False, True], ids=["plain", "dtensor"])
@pytest.mark.parametrize("master_weights", [False, True], ids=["four-list", "five-list"])
def test_real_te_fused_adam_empty_shard_update_parity(tmp_path: Path, master_weights: bool, use_dtensor: bool) -> None:
    """Keep native optimizer updates and state intact with boundary-slot empties."""
    world_size = 2 if use_dtensor else 1
    # This functional test needs a working TE CUDA extension and at most two GPUs.
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA device(s)")
    have_te, _ = safe_import_te()
    if not have_te:
        pytest.skip("Requires the real Transformer Engine PyTorch extension")
    mp.spawn(
        _optimizer_parity_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), master_weights, use_dtensor),
        nprocs=world_size,
        join=True,
    )
