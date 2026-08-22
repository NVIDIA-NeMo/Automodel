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

import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint import stateful_wrappers
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.shared.owner_sharding import ModelOwnedDTensorSpec, OwnerShardedParameterSpec

_TEST_OWNER_STATE_NAMESPACE = "__test_owner_sharded_v1"


class _FakeTEFusedAdam(torch.optim.Optimizer):
    """CPU stand-in for TE's lazy optimizer-state initialization contract."""

    def __init__(self, parameter: nn.Parameter, *, capturable: bool = False) -> None:
        super().__init__([parameter], {"lr": 0.01})
        self.master_weights = True
        self.store_param_remainders = True
        self.capturable = capturable
        self.initialize_calls = 0

    def initialize_state(self, parameter: nn.Parameter, store_param_remainders: bool) -> None:
        """Materialize the three numerical slots without updating the parameter."""
        self.initialize_calls += 1
        self.state[parameter]["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
        self.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        master_dtype = torch.int16 if store_param_remainders else torch.float32
        self.state[parameter]["master_param"] = torch.zeros_like(parameter, dtype=master_dtype)


class _ModelOwnedDTensorOptimizerModel(nn.Module):
    """Tiny globally shaped row DTensor consumed through its local shard."""

    def __init__(self, process_group: torch.distributed.ProcessGroup, *, global_rows: int = 8) -> None:
        super().__init__()
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if global_rows % world_size:
            raise ValueError("global_rows must divide evenly over the test process group")
        local_rows = global_rows // world_size
        local_weight = torch.arange(rank * local_rows, (rank + 1) * local_rows, dtype=torch.float32).unsqueeze(1)
        local_weight = local_weight.expand(-1, 2).contiguous()
        mesh = DeviceMesh.from_group(
            process_group,
            device_type="cpu",
            mesh_dim_names=("ple_owner",),
        )
        weight = DTensor.from_local(
            local_weight,
            device_mesh=mesh,
            placements=(Shard(0),),
            run_check=False,
            shape=(global_rows, 2),
            stride=(2, 1),
        )
        self.weight = nn.Parameter(weight)
        self.weight._nemo_model_owned_dtensor_spec = ModelOwnedDTensorSpec(
            process_group=process_group,
            gradient_divisor=float(world_size),
            legacy_optimizer_state_namespace=_TEST_OWNER_STATE_NAMESPACE,
        )


class _LegacyOwnerOptimizerModel(nn.Module):
    """Pre-DTensor local owner parameter used to create bridge checkpoints."""

    def __init__(self, process_group: torch.distributed.ProcessGroup, *, global_rows: int = 8) -> None:
        super().__init__()
        world_size = torch.distributed.get_world_size(process_group)
        if global_rows % world_size:
            raise ValueError("global_rows must divide evenly over the test process group")
        self.weight = nn.Parameter(torch.zeros(global_rows // world_size, 2))
        self.weight._nemo_owner_sharded_spec = OwnerShardedParameterSpec(
            process_group=process_group,
            gradient_divisor=float(world_size),
            optimizer_state_namespace=_TEST_OWNER_STATE_NAMESPACE,
        )


def _make_distributed_checkpointer(rank: int, checkpoint_dir: str) -> Checkpointer:
    """Build the ordinary full-training DCP wrapper for one test rank."""
    return Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=False,
        ),
        dp_rank=rank,
        tp_rank=0,
        pp_rank=0,
        process_group=torch.distributed.group.WORLD,
    )


def _state_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a writable optimizer-state shard."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _fill_row_coded_adam_state(model: nn.Module, optimizer: torch.optim.AdamW, rank: int) -> None:
    """Materialize Adam state and overwrite it with deterministic owner rows."""
    parameter = model.weight
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[parameter]
    local_rows = _state_local(state["exp_avg"]).shape[0]
    row_start = rank * local_rows
    rows = torch.arange(row_start, row_start + local_rows, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(2, dtype=torch.float32).unsqueeze(0)
    _state_local(state["exp_avg"]).copy_(rows * 10 + columns + 0.25)
    _state_local(state["exp_avg_sq"]).copy_(rows * 10 + columns + 0.75)
    state["step"].fill_(5)


def _assert_row_coded_adam_state(model: nn.Module, optimizer: torch.optim.AdamW, rank: int) -> None:
    """Check the deterministic state expected for this rank's row interval."""
    state = optimizer.state[model.weight]
    exp_avg = _state_local(state["exp_avg"])
    exp_avg_sq = _state_local(state["exp_avg_sq"])
    local_rows = exp_avg.shape[0]
    rows = torch.arange(rank * local_rows, (rank + 1) * local_rows, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(2, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(exp_avg, rows * 10 + columns + 0.25, rtol=0, atol=0)
    torch.testing.assert_close(exp_avg_sq, rows * 10 + columns + 0.75, rtol=0, atol=0)
    torch.testing.assert_close(state["step"], torch.tensor(5.0), rtol=0, atol=0)


def _make_stepped_adam_parts(
    count: int = 2,
) -> tuple[list[nn.Module], list[torch.optim.AdamW]]:
    model_parts: list[nn.Module] = []
    optimizers: list[torch.optim.AdamW] = []
    for index in range(count):
        model_part = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model_part.parameters(), lr=0.01 * (index + 1))
        for parameter in model_part.parameters():
            parameter.grad = torch.full_like(parameter, float(index + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model_parts.append(model_part)
        optimizers.append(optimizer)
    return model_parts, optimizers


class _OwnerShardedOptimizerModel(nn.Module):
    """Tiny model with one replicated and one model-owned parameter shard."""

    def __init__(self, process_group: torch.distributed.ProcessGroup) -> None:
        super().__init__()
        self.regular = nn.Parameter(torch.ones(2))
        self.owner = nn.Parameter(torch.ones(2))
        self.owner._nemo_owner_sharded_spec = OwnerShardedParameterSpec(
            process_group=process_group,
            gradient_divisor=2.0,
            optimizer_state_namespace=_TEST_OWNER_STATE_NAMESPACE,
        )


class _NamedOwnerShardModel(nn.Module):
    """Owner-sharded parameter with a caller-selected FQN for multi-part tests."""

    def __init__(self, process_group: torch.distributed.ProcessGroup, name: str) -> None:
        super().__init__()
        self.register_parameter(name, nn.Parameter(torch.ones(2)))
        parameter = self.get_parameter(name)
        parameter._nemo_owner_sharded_spec = OwnerShardedParameterSpec(
            process_group=process_group,
            gradient_divisor=2.0,
            optimizer_state_namespace=_TEST_OWNER_STATE_NAMESPACE,
        )


def _run_owner_sharded_optimizer_dcp_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        source_model = _OwnerShardedOptimizerModel(torch.distributed.group.WORLD)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
        source_model.regular.grad = torch.ones_like(source_model.regular)
        source_model.owner.grad = torch.full_like(source_model.owner, float(rank + 1))
        source_optimizer.step()

        regular_state = source_optimizer.state[source_model.regular]
        owner_state = source_optimizer.state[source_model.owner]
        regular_state["exp_avg"].fill_(7.0)
        regular_state["exp_avg_sq"].fill_(8.0)
        owner_state["step"].fill_(rank + 10)
        owner_state["exp_avg"].fill_(rank + 1)
        owner_state["exp_avg_sq"].fill_(rank + 2)
        expected_owner_state = {name: value.clone() for name, value in owner_state.items()}

        local_state = OptimizerState(source_model, source_optimizer).state_dict()["optim"]
        assert "state.regular.exp_avg" in local_state
        assert "state.owner.exp_avg" not in local_state
        owner_key = next(
            key
            for key in local_state
            if key.startswith(f"state.{_TEST_OWNER_STATE_NAMESPACE}") and key.endswith(".exp_avg")
        )
        assert owner_key == (f"state.{_TEST_OWNER_STATE_NAMESPACE}.owner_rank_{rank}.global_rank_{rank}.owner.exp_avg")
        topology_key = f"{_TEST_OWNER_STATE_NAMESPACE}.metadata.global_rank_{rank}.owner.owner_world_size"
        assert local_state[topology_key] == world_size
        assert "param_groups.owner.lr" in local_state

        mismatched_topology_state = {"optim": dict(local_state)}
        mismatched_topology_state["optim"][topology_key] += 1
        with pytest.raises(RuntimeError, match="Optimizer-state resharding is unsupported"):
            OptimizerState(source_model, source_optimizer).load_state_dict(mismatched_topology_state)

        legacy_state = {"optim": {}}
        owner_prefix = f"state.{_TEST_OWNER_STATE_NAMESPACE}.owner_rank_{rank}.global_rank_{rank}.owner."
        for key, value in local_state.items():
            if key == topology_key:
                continue
            legacy_key = f"state.owner.{key.removeprefix(owner_prefix)}" if key.startswith(owner_prefix) else key
            legacy_state["optim"][legacy_key] = value
        with pytest.raises(RuntimeError, match="Legacy owner-sharded optimizer state cannot be resumed safely"):
            OptimizerState(source_model, source_optimizer).load_state_dict(legacy_state)

        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=checkpoint_dir,
                model_save_format="safetensors",
                save_consolidated=False,
                is_peft=False,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=torch.distributed.group.WORLD,
        )
        checkpointer.save_optimizer(source_optimizer, source_model, checkpoint_dir)
        checkpoint_keys = FileSystemReader(os.path.join(checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        assert "optim.state.regular.exp_avg" in checkpoint_keys
        for owner_rank in range(world_size):
            assert (
                f"optim.state.{_TEST_OWNER_STATE_NAMESPACE}.owner_rank_{owner_rank}."
                f"global_rank_{owner_rank}.owner.exp_avg"
            ) in checkpoint_keys

        target_model = _OwnerShardedOptimizerModel(torch.distributed.group.WORLD)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.5)
        assert target_model.owner is not source_model.owner
        checkpointer.load_optimizer(target_optimizer, target_model, checkpoint_dir)

        target_regular_state = target_optimizer.state[target_model.regular]
        target_owner_state = target_optimizer.state[target_model.owner]
        torch.testing.assert_close(target_regular_state["exp_avg"], regular_state["exp_avg"])
        torch.testing.assert_close(target_regular_state["exp_avg_sq"], regular_state["exp_avg_sq"])
        for name, expected in expected_owner_state.items():
            torch.testing.assert_close(target_owner_state[name], expected)

        source_parts = [
            _NamedOwnerShardModel(torch.distributed.group.WORLD, "owner_a"),
            _NamedOwnerShardModel(torch.distributed.group.WORLD, "owner_b"),
        ]
        source_optimizers = [torch.optim.AdamW(part.parameters()) for part in source_parts]
        expected_part_moments = []
        for part_index, (part, optimizer) in enumerate(zip(source_parts, source_optimizers, strict=True)):
            parameter = next(part.parameters())
            parameter.grad = torch.full_like(parameter, float(rank + part_index + 1))
            optimizer.step()
            optimizer.state[parameter]["exp_avg"].fill_(10 * (part_index + 1) + rank)
            expected_part_moments.append(optimizer.state[parameter]["exp_avg"].clone())
        combined_state = OptimizerState(source_parts, source_optimizers).state_dict()

        target_parts = [
            _NamedOwnerShardModel(torch.distributed.group.WORLD, "owner_a"),
            _NamedOwnerShardModel(torch.distributed.group.WORLD, "owner_b"),
        ]
        target_optimizers = [torch.optim.AdamW(part.parameters()) for part in target_parts]
        OptimizerState(target_parts, target_optimizers).load_state_dict(combined_state)
        for part, optimizer, expected in zip(
            target_parts,
            target_optimizers,
            expected_part_moments,
            strict=True,
        ):
            torch.testing.assert_close(optimizer.state[next(part.parameters())]["exp_avg"], expected)
    finally:
        torch.distributed.destroy_process_group()


def _run_native_dtensor_optimizer_save(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _ModelOwnedDTensorOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.031, foreach=False)
        _fill_row_coded_adam_state(model, optimizer, rank)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        checkpointer.save_optimizer(optimizer, model, checkpoint_dir)
        torch.distributed.barrier()
        metadata = FileSystemReader(os.path.join(checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        assert not any(_TEST_OWNER_STATE_NAMESPACE in key for key in metadata)
        exp_avg_metadata = metadata["optim.state.weight.exp_avg"]
        assert tuple(exp_avg_metadata.size) == (8, 2)
        assert len(exp_avg_metadata.chunks) == world_size
    finally:
        torch.distributed.destroy_process_group()


def _run_native_dtensor_optimizer_load(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    resaved_checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _ModelOwnedDTensorOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.9, foreach=False)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        checkpointer.load_optimizer(optimizer, model, checkpoint_dir)
        _assert_row_coded_adam_state(model, optimizer, rank)
        assert optimizer.param_groups[0]["lr"] == 0.031

        resave_checkpointer = _make_distributed_checkpointer(rank, resaved_checkpoint_dir)
        resave_checkpointer.save_optimizer(optimizer, model, resaved_checkpoint_dir)
        torch.distributed.barrier()
        metadata = FileSystemReader(os.path.join(resaved_checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        assert tuple(metadata["optim.state.weight.exp_avg"].size) == (8, 2)
        assert len(metadata["optim.state.weight.exp_avg"].chunks) == world_size
        assert not any(_TEST_OWNER_STATE_NAMESPACE in key for key in metadata)
    finally:
        torch.distributed.destroy_process_group()


def _run_legacy_owner_optimizer_save(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _LegacyOwnerOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.031, foreach=False)
        _fill_row_coded_adam_state(model, optimizer, rank)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        checkpointer.save_optimizer(optimizer, model, checkpoint_dir)
        torch.distributed.barrier()
        metadata = FileSystemReader(os.path.join(checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        assert any(_TEST_OWNER_STATE_NAMESPACE in key for key in metadata)
        assert "optim.state.weight.exp_avg" not in metadata
    finally:
        torch.distributed.destroy_process_group()


def _run_legacy_owner_optimizer_load(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    expect_topology_rejection: bool,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _ModelOwnedDTensorOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.9, foreach=False)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        if expect_topology_rejection:
            with pytest.raises(RuntimeError, match="Cross-topology legacy optimizer resharding is unsupported"):
                checkpointer.load_optimizer(optimizer, model, checkpoint_dir)
            return
        checkpointer.load_optimizer(optimizer, model, checkpoint_dir)
        _assert_row_coded_adam_state(model, optimizer, rank)
        assert optimizer.param_groups[0]["lr"] == 0.031
    finally:
        torch.distributed.destroy_process_group()


def _assert_adam_states_equal(
    actual: torch.optim.AdamW,
    expected: torch.optim.AdamW,
) -> None:
    actual_parameter = actual.param_groups[0]["params"][0]
    expected_parameter = expected.param_groups[0]["params"][0]
    actual_state = actual.state[actual_parameter]
    expected_state = expected.state[expected_parameter]

    torch.testing.assert_close(actual_state["step"], expected_state["step"])
    torch.testing.assert_close(actual_state["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(actual_state["exp_avg_sq"], expected_state["exp_avg_sq"])
    assert actual.param_groups[0]["lr"] == expected.param_groups[0]["lr"]


def _assert_native_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for state_name, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][state_name]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _make_peft_ep_checkpointer(tmp_path, *, pp_rank: int = 0) -> Checkpointer:
    config = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_save_format="safetensors",
        save_consolidated=False,
        is_peft=True,
    )
    return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=pp_rank, moe_mesh=object())


def _run_native_pp_optimizer_dcp_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        source_models, source_optimizers = _make_stepped_adam_parts(count=1)
        source_optimizers[0].param_groups[0]["lr"] = 0.01 * (rank + 1)
        source_parameter = source_optimizers[0].param_groups[0]["params"][0]
        source_optimizers[0].state[source_parameter]["exp_avg"].fill_(rank + 1)
        source_optimizers[0].state[source_parameter]["exp_avg_sq"].fill_(rank + 2)
        checkpointer = _make_peft_ep_checkpointer(checkpoint_dir, pp_rank=rank)
        checkpointer.save_optimizer(
            source_optimizers,
            source_models,
            checkpoint_dir,
            optimizer_part_ids=[rank],
        )

        target_models = [nn.Linear(2, 1, bias=False)]
        target_optimizers = [torch.optim.AdamW(target_models[0].parameters(), lr=0.5)]
        checkpointer.load_optimizer(
            target_optimizers,
            target_models,
            checkpoint_dir,
            optimizer_part_ids=[rank],
        )
        _assert_adam_states_equal(target_optimizers[0], source_optimizers[0])
    finally:
        torch.distributed.destroy_process_group()


def test_checkpointer_uses_native_state_for_all_optimizer_parts_with_ep_topology(tmp_path):
    model_parts, optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)

    with patch.object(checkpointer, "_do_save") as do_save:
        checkpointer.save_optimizer(
            optimizers,
            model_parts,
            str(tmp_path),
            optimizer_part_ids=[0, 8],
        )

    state_dict = do_save.call_args.args[0]
    saved_parts = state_dict["optim"]["optimizer_parts"]
    assert saved_parts.keys() == {"stage_0", "stage_8"}
    for part_key, optimizer in zip(("stage_0", "stage_8"), optimizers, strict=True):
        _assert_native_state_dicts_equal(saved_parts[part_key], optimizer.state_dict())


def test_native_single_optimizer_state_preserves_existing_checkpoint_schema():
    model_parts, optimizers = _make_stepped_adam_parts(count=1)
    state_dict = OptimizerState(
        model_parts[0],
        optimizers[0],
        is_peft=True,
        has_expert_parallelism=True,
    ).state_dict()

    assert state_dict["optim"].keys() == {"state", "param_groups"}


def test_flattened_untagged_optimizer_state_preserves_existing_checkpoint_schema():
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    expected_state = {
        "state.weight.exp_avg": torch.ones_like(model.weight),
        "param_groups.weight.lr": 0.01,
    }

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.get_optimizer_state_dict",
        return_value=expected_state,
    ):
        actual_state = OptimizerState(model, optimizer).state_dict()["optim"]

    assert actual_state.keys() == expected_state.keys()
    assert actual_state["state.weight.exp_avg"] is expected_state["state.weight.exp_avg"]
    assert actual_state["param_groups.weight.lr"] == expected_state["param_groups.weight.lr"]


@pytest.mark.parametrize("capturable", [False, True])
def test_optimizer_state_materializes_te_fused_adam_load_destinations(monkeypatch, capturable: bool) -> None:
    """Fresh TE FusedAdam state and its group step exist before DCP planning."""
    model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _FakeTEFusedAdam(model.weight, capturable=capturable)
    monkeypatch.setattr(stateful_wrappers, "_te_fused_adam_type", lambda: _FakeTEFusedAdam)

    def fake_get_optimizer_state_dict(model_arg, optimizer_arg, *, options):
        del model_arg, options
        state = optimizer_arg.state[model.weight]
        assert set(state) == {"exp_avg", "exp_avg_sq", "master_param"}
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        assert state["master_param"].dtype == torch.int16
        step = optimizer_arg.param_groups[0]["step"]
        if capturable:
            torch.testing.assert_close(step, torch.tensor([0], dtype=torch.int), rtol=0, atol=0)
        else:
            assert step == 0
        return {"state.weight.exp_avg": state["exp_avg"]}

    monkeypatch.setattr(stateful_wrappers, "get_optimizer_state_dict", fake_get_optimizer_state_dict)
    state_dict = OptimizerState(model, optimizer).state_dict()
    assert state_dict["optim"].keys() == {"state.weight.exp_avg"}
    assert optimizer.initialize_calls == 1

    OptimizerState(model, optimizer).state_dict()
    assert optimizer.initialize_calls == 1


def test_optimizer_state_rejects_partially_initialized_te_fused_adam(monkeypatch) -> None:
    """A partial TE state is corruption, not permission to overwrite slots."""
    model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _FakeTEFusedAdam(model.weight)
    optimizer.state[model.weight]["exp_avg"] = torch.zeros_like(model.weight, dtype=torch.float32)
    monkeypatch.setattr(stateful_wrappers, "_te_fused_adam_type", lambda: _FakeTEFusedAdam)

    with pytest.raises(RuntimeError, match="partially initialized state.*exp_avg_sq.*master_param"):
        OptimizerState(model, optimizer).state_dict()


def test_native_multi_optimizer_state_dcp_round_trip(tmp_path):
    source_models, source_optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)
    checkpointer.save_optimizer(
        source_optimizers,
        source_models,
        str(tmp_path),
        optimizer_part_ids=[0, 2],
    )

    target_models = [nn.Linear(2, 1, bias=False) for _ in source_models]
    target_optimizers = [torch.optim.AdamW(model.parameters(), lr=0.5) for model in target_models]
    checkpointer.load_optimizer(
        target_optimizers,
        target_models,
        str(tmp_path),
        optimizer_part_ids=[0, 2],
    )

    for target, source in zip(target_optimizers, source_optimizers, strict=True):
        _assert_adam_states_equal(target, source)


def test_native_pipeline_optimizer_load_rejects_stage_mismatch():
    model_parts, optimizers = _make_stepped_adam_parts()
    optimizer_state = OptimizerState(
        model_parts,
        optimizers,
        is_peft=True,
        has_expert_parallelism=True,
        optimizer_part_ids=[0, 8],
    )
    state_dict = optimizer_state.state_dict()
    state_dict["optim"]["optimizer_parts"]["stage_7"] = state_dict["optim"]["optimizer_parts"].pop("stage_8")

    with pytest.raises(ValueError, match=r"checkpoint has \['stage_0', 'stage_7'\].*expects \['stage_0', 'stage_8'\]"):
        optimizer_state.load_state_dict(state_dict)


def test_native_multi_optimizer_state_requires_global_part_ids():
    model_parts, optimizers = _make_stepped_adam_parts()
    optimizer_state = OptimizerState(
        model_parts,
        optimizers,
        is_peft=True,
        has_expert_parallelism=True,
    )

    with pytest.raises(ValueError, match="requires global optimizer part IDs"):
        optimizer_state.state_dict()


def test_native_pipeline_optimizer_state_dcp_round_trip_across_ranks(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_native_pp_optimizer_dcp_round_trip,
        args=(world_size, str(tmp_path / "dist_init"), str(tmp_path / "checkpoint")),
        nprocs=world_size,
        join=True,
    )


def test_owner_sharded_optimizer_state_rejects_untyped_marker():
    model = nn.Linear(2, 1, bias=False)
    model.weight._nemo_owner_sharded_spec = object()
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(TypeError, match="must be an OwnerShardedParameterSpec"):
        OptimizerState(model, optimizer).state_dict()


def test_owner_sharded_optimizer_state_rejects_ambiguous_distributed_owner():
    model = nn.Linear(2, 1, bias=False)
    model.weight._nemo_owner_sharded_spec = OwnerShardedParameterSpec(
        process_group=None,
        gradient_divisor=2.0,
        optimizer_state_namespace=_TEST_OWNER_STATE_NAMESPACE,
    )
    optimizer = torch.optim.AdamW(model.parameters())

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        pytest.raises(RuntimeError, match="unique owner identity cannot be derived"),
    ):
        OptimizerState(model, optimizer).state_dict()


def test_owner_sharded_optimizer_state_dcp_round_trip_across_ranks(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_owner_sharded_optimizer_dcp_round_trip,
        args=(world_size, str(tmp_path / "dist_init"), str(tmp_path / "checkpoint")),
        nprocs=world_size,
        join=True,
    )


def test_model_owned_dtensor_optimizer_state_reshards_world_two_to_four(tmp_path):
    checkpoint_dir = str(tmp_path / "native-dtensor-checkpoint")
    torch.multiprocessing.spawn(
        _run_native_dtensor_optimizer_save,
        args=(2, str(tmp_path / "native-save-init"), checkpoint_dir),
        nprocs=2,
        join=True,
    )
    torch.multiprocessing.spawn(
        _run_native_dtensor_optimizer_load,
        args=(
            4,
            str(tmp_path / "native-load-init"),
            checkpoint_dir,
            str(tmp_path / "native-dtensor-resaved"),
        ),
        nprocs=4,
        join=True,
    )


def test_legacy_owner_optimizer_loads_same_topology_and_rejects_reshard(tmp_path):
    checkpoint_dir = str(tmp_path / "legacy-owner-checkpoint")
    torch.multiprocessing.spawn(
        _run_legacy_owner_optimizer_save,
        args=(2, str(tmp_path / "legacy-save-init"), checkpoint_dir),
        nprocs=2,
        join=True,
    )
    torch.multiprocessing.spawn(
        _run_legacy_owner_optimizer_load,
        args=(2, str(tmp_path / "legacy-load-same-init"), checkpoint_dir, False),
        nprocs=2,
        join=True,
    )
    torch.multiprocessing.spawn(
        _run_legacy_owner_optimizer_load,
        args=(4, str(tmp_path / "legacy-load-reshard-init"), checkpoint_dir, True),
        nprocs=4,
        join=True,
    )
