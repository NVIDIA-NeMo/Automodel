# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Behavioral tests for ``PipelineConfig.build``.

``build`` is exercised against the real :class:`AutoPipeline` rather than a mock,
so a forwarding mistake shows up as a wrong pipeline setting instead of an
assertion about a call signature.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig


class _FakePpMesh:
    """Minimal pipeline submesh exposing the size AutoPipeline reads."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        """Return the number of pipeline ranks."""
        return self._size


class _FakeDeviceMesh:
    """Device mesh stand-in whose ``["pp"]`` lookup yields a pipeline submesh."""

    def __init__(self, pp_size: int = 2) -> None:
        self._pp_size = pp_size

    def __getitem__(self, axis: str) -> _FakePpMesh:
        assert axis == "pp"
        return _FakePpMesh(self._pp_size)


def _mesh_context(pp_size: int = 2, *, has_device_mesh: bool = True) -> SimpleNamespace:
    """Build a ``MeshContext`` stand-in exposing what ``PipelineConfig.build`` reads."""
    return SimpleNamespace(
        device_mesh=_FakeDeviceMesh(pp_size) if has_device_mesh else None,
        moe_mesh=None,
        pp_size=pp_size,
        pipeline_axis_kwargs=lambda: {
            "pp_axis_name": "pp",
            "dp_axis_names": ("dp",),
            "cp_axis_name": None,
            "tp_axis_name": None,
            "ep_axis_name": None,
            "ep_shard_axis_names": None,
        },
    )


class TestPipelineConfigBuild:
    """Pipeline construction is driven by the mesh and the declared config."""

    def test_returns_none_when_the_mesh_has_no_device_mesh(self):
        """Without a device mesh there is nothing to pipeline over."""
        assert PipelineConfig().build(mesh=_mesh_context(pp_size=2, has_device_mesh=False)) is None

    def test_returns_none_when_the_mesh_disables_pipelining(self):
        """A pipeline degree of one means pipeline parallelism is off."""
        assert PipelineConfig().build(mesh=_mesh_context(pp_size=1)) is None

    def test_built_pipeline_carries_the_declared_settings(self):
        """Declarative config values and runtime arguments reach the pipeline."""
        config = PipelineConfig(
            pp_schedule="gpipe",
            pp_microbatch_size=2,
            pp_batch_size=8,
            layers_per_stage=4,
            round_virtual_stages_to_pp_multiple="up",
            module_fqns_per_model_part=[["model.layers.0"], ["model.layers.1"]],
            patch_inner_model=False,
            patch_causal_lm_model=False,
            patch_stage_backward_maybe_with_nosync=True,
            dtype=torch.bfloat16,
            scale_grads_in_schedule=True,
        )

        pipeline = config.build(mesh=_mesh_context(), device=torch.device("cpu"), defer_fsdp_grad_sync=False)

        assert isinstance(pipeline, AutoPipeline)
        assert pipeline.pp_schedule == "gpipe"
        assert pipeline.pp_microbatch_size == 2
        assert pipeline.pp_batch_size == 8
        assert pipeline.layers_per_stage == 4
        assert pipeline.round_virtual_stages_to_pp_multiple == "up"
        assert pipeline.module_fqns_per_model_part == [["model.layers.0"], ["model.layers.1"]]
        assert pipeline.patch_inner_model is False
        assert pipeline.patch_causal_lm_model is False
        assert pipeline.patch_stage_backward_maybe_with_nosync is True
        assert pipeline.dtype is torch.bfloat16
        assert pipeline.scale_grads_in_schedule is True
        assert pipeline.defer_fsdp_grad_sync is False
        assert pipeline.device == torch.device("cpu")
        assert pipeline.pp_axis_name == "pp"
        assert pipeline.dp_axis_names == ("dp",)

    def test_built_pipeline_uses_the_config_defaults(self):
        """An unconfigured PipelineConfig builds the default 1f1b pipeline."""
        pipeline = PipelineConfig().build(mesh=_mesh_context(), device=torch.device("cpu"))

        assert pipeline.pp_schedule == "1f1b"
        assert pipeline.pp_schedule_csv is None
        assert pipeline.pp_microbatch_size == 1
        assert pipeline.pp_batch_size == 1
        assert pipeline.layers_per_stage is None
        assert pipeline.dtype is None
        assert pipeline.scale_grads_in_schedule is False
        assert pipeline.defer_fsdp_grad_sync is True

    def test_invalid_microbatch_split_is_rejected_at_build_time(self):
        """A batch that cannot be split into microbatches fails when the pipeline is built."""
        config = PipelineConfig(pp_microbatch_size=3, pp_batch_size=8)

        with pytest.raises(ValueError, match="must be divisible by pp_microbatch_size"):
            config.build(mesh=_mesh_context(), device=torch.device("cpu"))
