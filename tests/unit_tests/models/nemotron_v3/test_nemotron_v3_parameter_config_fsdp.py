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

"""Two-rank FSDP coverage for selected FP32 storage in hybrid Nemotron-H Mamba blocks."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from nemo_automodel.components.models.nemotron_v3.parallelization import PARALLELIZER
from nemo_automodel.components.utils.parameter_config import ParameterConfig, apply_parameter_config
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_model import MockNemotronV3Config


def _shard_with_fp32(mesh, selected_prefix: str) -> NemotronHForCausalLM:
    backend = BackendConfig(
        linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
    )
    model = NemotronHForCausalLM(MockNemotronV3Config(layers_block_type=["mamba", "attention"]), backend=backend)
    model.initialize_weights()
    selected = tuple(name for name, _ in model.named_parameters() if name.startswith(selected_prefix))
    apply_parameter_config(model, ParameterConfig(selected, torch.float32, torch.bfloat16))
    PARALLELIZER._apply(
        model,
        device_mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    )
    return model


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (1, 2, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"))
        block = _shard_with_fp32(mesh, "model.layers.0.").model.layers["0"]
        assert isinstance(block, FSDPModule)
        for name in ("conv1d", "norm", "out_proj"):
            module = getattr(block.mixer, name)
            assert not isinstance(module, FSDPModule), name
            assert module.weight.dtype == torch.float32, name

        # Three dtype groups would give each fused-kernel module its own unit.
        with pytest.raises(ValueError, match="Partial Mamba-block dtype selections are not supported"):
            _shard_with_fp32(mesh, "model.layers.0.norm.")
    finally:
        dist.destroy_process_group()


def test_mamba_fp32_selection_keeps_fused_kernel_weights_in_block_unit(tmp_path: Path) -> None:
    """Whole-block FP32 selection shards normally; a split of the fused-kernel weights is rejected."""
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
