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

"""Mesh timeout overrides keep CPU collectives and create no extra process groups.

Spawning ranks dominates the cost of these tests, so a single two-rank spawn runs
every case, and the parametrized tests assert on the outcomes it records.
"""

import datetime
import functools
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs"),
    # The first test to run also pays for the shared two-rank spawn.
    pytest.mark.timeout(240),
]

_WORLD_SIZE = 2
_DEFAULT_TIMEOUT = datetime.timedelta(minutes=7)
_MESH_TIMEOUT_MINUTES = 45
_LAYOUTS = ("dp", "cp", "tp", "ep")
_BIND_MODES = {"lazy": False, "bound": True}
# PyTorch 2.10 segfaults in split_group when DeviceMesh splits a device-bound
# composite default group, which it does for every axis without a backend override.
# Bound runs therefore build only the mesh with a timeout, whose per-axis overrides
# create groups through new_group instead.
_COUNTED_TIMEOUTS = {"lazy": (None, _MESH_TIMEOUT_MINUTES), "bound": (_MESH_TIMEOUT_MINUTES,)}


def _parallelism(layout: str) -> ParallelismSizes:
    """Return two-rank sizes in which the named axis spans both ranks."""
    if layout == "dp":
        return ParallelismSizes()
    if layout == "cp":
        return ParallelismSizes(cp_size=2)
    if layout == "tp":
        return ParallelismSizes(tp_size=2)
    if layout == "ep":
        return ParallelismSizes(ep_size=2)
    raise ValueError(f"Unknown layout {layout!r}")


def _process_groups() -> set[dist.ProcessGroup]:
    """Return every process group this rank belongs to, including the default group."""
    return set(dist.distributed_c10d._world.pg_map)


def _count_created_groups(layout: str, timeouts: tuple[int | None, ...]) -> dict[str, int]:
    """Build the mesh once per timeout, and count the process groups each build creates.

    Counts are keyed by the timeout's string form. ``world_sized`` counts new groups
    that span every rank across all builds.
    """
    counts = {"world_sized": 0}
    contexts = []  # Keep every mesh alive while counting.
    for timeout_minutes in timeouts:
        before = _process_groups()
        contexts.append(
            MeshContext.build(
                FSDP2Config(), _parallelism(layout), world_size=_WORLD_SIZE, timeout_minutes=timeout_minutes
            )
        )
        created = _process_groups() - before
        counts[str(timeout_minutes)] = len(created)
        counts["world_sized"] += sum(group.size() == _WORLD_SIZE for group in created)
    return counts


def _check_backends(device: torch.device, layout: str) -> None:
    """Check CPU and CUDA collectives, timeouts, and CPU-offloaded FSDP gathers on every mesh axis."""
    context = MeshContext.build(
        FSDP2Config(), _parallelism(layout), world_size=_WORLD_SIZE, timeout_minutes=_MESH_TIMEOUT_MINUTES
    )
    meshes = [context.device_mesh]
    if context.moe_mesh is not None:
        meshes.append(context.moe_mesh)
    for mesh in meshes:
        for axis in mesh.mesh_dim_names:
            group = mesh[axis].get_group()
            # World-sized axes reuse the default group, which already owns its timeout,
            # instead of allocating another communicator. Every new group uses the mesh timeout.
            assert (group is dist.group.WORLD) == (mesh[axis].size() == _WORLD_SIZE), axis
            expected_timeout = (
                _DEFAULT_TIMEOUT if group is dist.group.WORLD else datetime.timedelta(minutes=_MESH_TIMEOUT_MINUTES)
            )
            for device_type in ("cpu", "cuda"):
                tensor = torch.ones(1, device=device_type)
                dist.all_reduce(tensor, group=group)
                torch.testing.assert_close(tensor, torch.full_like(tensor, dist.get_world_size(group)))
                assert group._get_backend(torch.device(device_type)).options._timeout == expected_timeout, axis

    # Exercise the failure from #4601: reconstructing a CPU-offloaded FSDP parameter
    # all-gathers on the CPU through the selected DP mesh.
    torch.manual_seed(17)
    model = torch.nn.Linear(4, 4, bias=False, device=device)
    expected = model.weight.detach().cpu().clone()
    fully_shard(model, mesh=get_fsdp_dp_mesh(context.device_mesh), offload_policy=CPUOffloadPolicy(pin_memory=False))
    model.to("cpu")
    torch.testing.assert_close(model.state_dict()["weight"].full_tensor(), expected)


def _capture(check: Callable[[], object]) -> dict[str, object]:
    """Run one check and record its value, or its traceback when it raises."""
    try:
        return {"value": check(), "error": None}
    except Exception:  # Reported by the test case that owns this check.
        return {"value": None, "error": traceback.format_exc()}


def _run_cases(rank: int, results_dir: str) -> None:
    """Run every layout under lazy and device-bound initialization, and write this rank's outcomes."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    results = {}
    for bind_name, bind_device in _BIND_MODES.items():
        dist.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"file://{results_dir}/store-{bind_name}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=_DEFAULT_TIMEOUT,
            device_id=device if bind_device else None,
        )
        try:
            for layout in _LAYOUTS:
                results[f"{layout}-{bind_name}"] = {
                    "groups": _capture(functools.partial(_count_created_groups, layout, _COUNTED_TIMEOUTS[bind_name])),
                    "backends": _capture(functools.partial(_check_backends, device, layout)),
                }
        finally:
            dist.destroy_process_group()
    Path(results_dir, f"rank{rank}.json").write_text(json.dumps(results))


@pytest.fixture(scope="module")
def mesh_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict]]:
    """Spawn two ranks once, and return each case's outcomes from both ranks."""
    results_dir = tmp_path_factory.mktemp("mesh_backends")
    mp.spawn(_run_cases, args=(str(results_dir),), nprocs=_WORLD_SIZE, join=True)
    per_rank = [json.loads((results_dir / f"rank{rank}.json").read_text()) for rank in range(_WORLD_SIZE)]
    return {case: [results[case] for results in per_rank] for case in per_rank[0]}


def _values(mesh_results: dict[str, list[dict]], case: str, check: str) -> list[object]:
    """Return a check's value from every rank, failing with the first recorded traceback."""
    values = []
    for rank, outcomes in enumerate(mesh_results[case]):
        outcome = outcomes[check]
        assert outcome["error"] is None, f"rank {rank}:\n{outcome['error']}"
        values.append(outcome["value"])
    return values


@pytest.mark.parametrize("bind", list(_BIND_MODES))
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_mesh_timeout_preserves_cpu_collectives(mesh_results: dict[str, list[dict]], layout: str, bind: str) -> None:
    _values(mesh_results, f"{layout}-{bind}", "backends")


@pytest.mark.parametrize("bind", list(_BIND_MODES))
@pytest.mark.parametrize(
    ("layout", "expected_groups"),
    [
        # pp, dp_replicate, cp, and tp have size 1. dp_shard and every flattened axis span both ranks.
        ("dp", 4),
        # Four size-1 root axes, plus the flattened dp axis (dp_replicate x dp_shard), which has size 1.
        ("cp", 5),
        # Four size-1 root axes, plus the flattened dp, dp_shard_cp, and dp_cp axes, which have size 1.
        ("tp", 7),
        # The dp layout's groups plus the size-1 ep_shard axis. ep spans both ranks.
        ("ep", 5),
    ],
)
def test_mesh_process_group_count(
    mesh_results: dict[str, list[dict]], layout: str, bind: str, expected_groups: int
) -> None:
    """Pin how many process groups each mesh build creates on a rank.

    Every process group owns its own communicators, so an unintended extra group
    costs device memory. A per-axis timeout must not add groups, and no new group may
    span every rank, because that duplicates the default group. When a change adds a
    mesh axis on purpose, update the expected count here.
    """
    expected = {str(timeout): expected_groups for timeout in _COUNTED_TIMEOUTS[bind]}
    for counts in _values(mesh_results, f"{layout}-{bind}", "groups"):
        assert counts == {**expected, "world_sized": 0}
