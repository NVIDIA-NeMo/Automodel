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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nemo_automodel.components.distributed.pipelining.config import PipelineConfig


class TestPipelineConfigPpSeqLen:
    def test_default_pp_seq_len_is_none(self):
        config = PipelineConfig()
        assert config.pp_seq_len is None

    def test_pp_seq_len_set_via_constructor(self):
        config = PipelineConfig(pp_seq_len=2048)
        assert config.pp_seq_len == 2048

    def test_pp_seq_len_set_via_attribute(self):
        config = PipelineConfig()
        config.pp_seq_len = 4096
        assert config.pp_seq_len == 4096


class TestPipelineConfigBuild:
    def test_disabled_mesh_returns_none(self):
        mesh = SimpleNamespace(device_mesh=None, pp_size=1)

        assert PipelineConfig().build(mesh=mesh) is None

    def test_build_forwards_declarative_and_runtime_values(self):
        device_mesh = Mock()
        mesh = SimpleNamespace(
            device_mesh=device_mesh,
            moe_mesh=None,
            pp_size=2,
            pipeline_axis_kwargs=lambda: {"pp_axis_name": "pp", "dp_axis_names": ("dp",)},
        )
        config = PipelineConfig(
            pp_schedule="gpipe",
            pp_microbatch_size=2,
            pp_batch_size=8,
            pp_seq_len=1024,
        )

        with patch("nemo_automodel.components.distributed.pipelining.autopipeline.AutoPipeline") as auto_pipeline:
            result = config.build(
                mesh=mesh,
                device=torch.device("cpu"),
                defer_fsdp_grad_sync=False,
            )

        assert result is auto_pipeline.return_value
        auto_pipeline.assert_called_once_with(
            world_mesh=device_mesh,
            moe_mesh=None,
            device=torch.device("cpu"),
            defer_fsdp_grad_sync=False,
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            pp_schedule="gpipe",
            pp_schedule_csv=None,
            pp_microbatch_size=2,
            pp_batch_size=8,
            layers_per_stage=None,
            round_virtual_stages_to_pp_multiple=None,
            module_fqns_per_model_part=None,
            patch_inner_model=True,
            patch_causal_lm_model=True,
            patch_stage_backward_maybe_with_nosync=False,
            dtype=None,
            scale_grads_in_schedule=False,
            pp_seq_len=1024,
        )
