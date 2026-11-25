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

"""Admission guards and actual two-rank CPU FSDP executor coverage."""

from datetime import timedelta
from pathlib import Path
from typing import Unpack
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed import FSDP2Config, MeshContext, MoEParallelizerConfig
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.ernie5_5.parallelization import Ernie5_5ModelParallelizer, _FullyShardKwargs
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import model
from tests.unit_tests.models.ernie5_5.test_parallelization import (
    _context,
    _Mesh,
    _record_executors,
    _set_world_size,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("outer,cp,error", [(False, 1, "wrap_outer_model=True"), (True, 2, "requires CP1")])
def test_mtp_invalid_ownership_or_cp_is_rejected_before_executor(
    monkeypatch: pytest.MonkeyPatch, outer: bool, cp: int, error: str
) -> None:
    candidate = model()
    identities = {name: id(value) for name, value in candidate.named_parameters()}
    executors = _record_executors(monkeypatch)
    _set_world_size(monkeypatch, 2)
    context = _context(
        strategy_config=FSDP2Config(),
        moe_parallel_config=MoEParallelizerConfig(wrap_outer_model=outer),
        device_mesh=_Mesh(ranks=2),
        ep_size=2,
        cp_size=cp,
    )
    with pytest.raises(ValueError, match=error):
        parallelize_model(candidate, context)
    assert not any(executor.called for executor in executors.values())
    assert identities == {name: id(value) for name, value in candidate.named_parameters()}


def test_mtp_executor_without_outer_fsdp_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = model()
    executor = Mock(return_value=candidate)
    monkeypatch.setattr(Ernie5_5ModelParallelizer, "_apply_moe", executor)
    with pytest.raises(RuntimeError, match="outer model.*FSDP unit"):
        candidate.parallelizer._apply_moe(candidate, _context())
    executor.assert_called_once()


def _check_actual_mtp_fsdp_coverage(rank: int, rendezvous: str) -> None:
    """Run the real executor, then inject one ignored outer parameter."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        mesh = MeshContext.build(
            FSDP2Config(activation_checkpointing=False),
            ParallelismSizes(dp_size=2, ep_size=2),
            activation_checkpointing=False,
            world_size=2,
            timeout_minutes=2,
            moe_parallel_config=MoEParallelizerConfig(),
        )
        candidate = parallelize_model(model(), mesh)
        assert isinstance(candidate, FSDPModule)
        owned = {name: p for name, p in candidate.named_parameters() if name.startswith("mtp.") and p.requires_grad}
        assert len(owned) == 8 and all(isinstance(p, DTensor) for p in owned.values())

        incomplete = model()
        original_shard = incomplete.parallelizer._fully_shard_module
        excluded = incomplete.mtp.e_proj.weight

        def leave_one_parameter_unowned(module: nn.Module, **kwargs: Unpack[_FullyShardKwargs]) -> nn.Module:
            if module is incomplete:
                kwargs["ignored_params"] = (kwargs.get("ignored_params") or set()) | {excluded}
            return original_shard(module, **kwargs)

        # Only the real FSDP ignore boundary is varied. The parent executor,
        # child units, EP mesh, DTensors and outer FSDP root remain real.
        with patch.object(incomplete.parallelizer, "_fully_shard_module", new=leave_one_parameter_unowned):
            with pytest.raises(RuntimeError, match=r"outside FSDP units.*mtp\.e_proj\.weight"):
                parallelize_model(incomplete, mesh)
        assert isinstance(incomplete, FSDPModule)
        assert incomplete.mtp.e_proj.weight is excluded and not isinstance(excluded, DTensor)
    finally:
        dist.destroy_process_group()


def test_actual_fsdp_executor_covers_mtp_and_rejects_an_unowned_parameter(tmp_path: Path) -> None:
    """The same ownership postcondition must pass and fail on real shards."""
    mp.spawn(_check_actual_mtp_fsdp_coverage, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
