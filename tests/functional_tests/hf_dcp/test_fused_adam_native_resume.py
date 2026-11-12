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

"""Native optimizer checkpoint path: resume real TE FusedAdam state into a never-stepped optimizer."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.shared.import_utils import safe_import, safe_import_te
from nemo_automodel.shared.te_patches import apply_te_patches

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_CASES = ["fp32-moments", "bf16-moments", "capturable"]


def _optimizer_options(case: str) -> dict[str, float | bool | torch.dtype]:
    """FusedAdam options per case: fp32 master weights throughout.

    ``fp32-moments`` and ``bf16-moments`` keep int16 master remainders for the bf16 parameters;
    ``capturable`` keeps the step counter as an int32 tensor of shape [1].
    """
    if case == "capturable":
        return dict(lr=0.003, master_weights=True, master_weight_dtype=torch.float32, capturable=True)
    moment_dtype = torch.float32 if case == "fp32-moments" else torch.bfloat16
    return dict(
        lr=0.003,
        master_weights=True,
        master_weight_dtype=torch.float32,
        store_param_remainders=True,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
    )


class _MixedPrecisionModel(nn.Module):
    """A bf16 linear layer (int16 master remainders when enabled) and an fp32 linear layer (fp32 master copy)."""

    def __init__(self) -> None:
        super().__init__()
        self.bf16_layer = nn.Linear(8, 4, device="cuda", dtype=torch.bfloat16)
        self.fp32_layer = nn.Linear(4, 2, device="cuda", dtype=torch.float32)


def _require_te_fused_adam() -> ModuleType:
    """Skip unless the real Transformer Engine PyTorch extension and its FusedAdam module import."""
    have_te, _ = safe_import_te()
    if not have_te:
        pytest.skip("Requires the real Transformer Engine PyTorch extension")
    available, fused_adam = safe_import("transformer_engine.pytorch.optimizers.fused_adam")
    if not available:
        pytest.skip("Requires transformer_engine.pytorch.optimizers.fused_adam")
    return fused_adam


def _make_optimizer(fused_adam: ModuleType, model: _MixedPrecisionModel, case: str) -> torch.optim.Optimizer:
    """Build FusedAdam directly, with one parameter group per layer so every group carries its own step."""
    return fused_adam.FusedAdam(
        [
            {"params": list(model.bf16_layer.parameters())},
            {"params": list(model.fp32_layer.parameters()), "weight_decay": 0.0},
        ],
        weight_decay=0.1,
        **_optimizer_options(case),
    )


def _set_grads(model: nn.Module, value: float) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)


def _steps(optimizer: torch.optim.Optimizer) -> list[int]:
    """Per-group step counters; TE stores an int, or an int32 tensor of shape [1] when capturable."""
    return [int(group["step"]) for group in optimizer.param_groups]


def _assert_same_parameters_and_state(
    expected_model: nn.Module,
    expected_optimizer: torch.optim.Optimizer,
    actual_model: nn.Module,
    actual_optimizer: torch.optim.Optimizer,
) -> None:
    """Bitwise-compare parameters and every optimizer state tensor, parameter by parameter."""
    for expected, actual in zip(expected_model.parameters(), actual_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert expected_optimizer.state[expected].keys() == actual_optimizer.state[actual].keys()
        for name, value in expected_optimizer.state[expected].items():
            torch.testing.assert_close(actual_optimizer.state[actual][name], value, rtol=0, atol=0)


@pytest.fixture
def expert_mesh() -> Iterator[DeviceMesh]:
    """One-rank CPU mesh; the checkpointer reads it only as the expert-parallel topology signal."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
    finally:
        dist.destroy_process_group()


def _make_checkpointer(checkpoint_dir: Path, expert_mesh: DeviceMesh) -> Checkpointer:
    """PEFT plus expert parallelism selects the native optimizer checkpoint path."""
    return Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=str(checkpoint_dir),
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=expert_mesh,
    )


@pytest.mark.parametrize("case", _CASES)
def test_native_resume_restores_fused_adam_state_into_fresh_optimizer(
    tmp_path: Path, expert_mesh: DeviceMesh, case: str
) -> None:
    """A never-stepped FusedAdam resumes moments, master weights and group steps through Checkpointer."""
    fused_adam = _require_te_fused_adam()
    apply_te_patches()
    torch.manual_seed(19)
    source = _MixedPrecisionModel()
    reference = _make_optimizer(fused_adam, source, case)
    for step in range(2):
        _set_grads(source, 0.125 * (step + 1))
        reference.step()
    torch.cuda.synchronize()
    assert _steps(reference) == [2, 2]

    target = _MixedPrecisionModel()
    target.load_state_dict(source.state_dict())
    resumed = _make_optimizer(fused_adam, target, case)
    assert not resumed.state
    assert all("step" not in group for group in resumed.param_groups)

    checkpointer = _make_checkpointer(tmp_path, expert_mesh)
    checkpointer.save_optimizer(reference, source, str(tmp_path))
    checkpointer.load_optimizer(resumed, target, str(tmp_path))

    assert _steps(resumed) == [2, 2]
    assert all(parameter.grad is None for parameter in target.parameters())
    _assert_same_parameters_and_state(source, reference, target, resumed)

    # Serializing the resumed optimizer again must reuse its buffers instead of allocating new state.
    pointers = [value.data_ptr() for state in resumed.state.values() for value in state.values()]
    OptimizerState(target, resumed, is_peft=True, has_expert_parallelism=True).state_dict()
    assert pointers == [value.data_ptr() for state in resumed.state.values() for value in state.values()]

    for model, optimizer in ((source, reference), (target, resumed)):
        _set_grads(model, 0.375)
        optimizer.step()
    torch.cuda.synchronize()
    assert _steps(resumed) == [3, 3]
    _assert_same_parameters_and_state(source, reference, target, resumed)


@pytest.mark.parametrize("case", _CASES)
def test_native_skeleton_materialization_keeps_first_update(case: str) -> None:
    """Building the native skeleton for a fresh FusedAdam allocates zero state and leaves its first update unchanged."""
    fused_adam = _require_te_fused_adam()
    apply_te_patches()
    torch.manual_seed(23)
    control_model = _MixedPrecisionModel()
    probe_model = _MixedPrecisionModel()
    probe_model.load_state_dict(control_model.state_dict())
    control = _make_optimizer(fused_adam, control_model, case)
    probe = _make_optimizer(fused_adam, probe_model, case)

    skeleton = OptimizerState(probe_model, probe, is_peft=True, has_expert_parallelism=True).state_dict()["optim"]

    assert skeleton["state"].keys() == {0, 1, 2, 3}
    assert _steps(probe) == [0, 0]
    assert [int(group["step"]) for group in skeleton["param_groups"]] == [0, 0]
    for parameter in probe_model.parameters():
        state = probe.state[parameter]
        assert state.keys() == {"exp_avg", "exp_avg_sq", "master_param"}
        assert torch.count_nonzero(state["exp_avg"]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"]).item() == 0
        if probe.store_param_remainders and parameter.dtype == torch.bfloat16:
            assert state["master_param"].dtype == torch.int16
            assert torch.count_nonzero(state["master_param"]).item() == 0
        else:
            torch.testing.assert_close(state["master_param"], parameter.detach().float(), rtol=0, atol=0)
    assert not control.state

    for model, optimizer in ((control_model, control), (probe_model, probe)):
        _set_grads(model, 0.25)
        optimizer.step()
    torch.cuda.synchronize()
    assert _steps(control) == _steps(probe) == [1, 1]
    _assert_same_parameters_and_state(control_model, control, probe_model, probe)
