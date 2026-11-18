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

"""DoRA magnitudes follow the base weights loaded on the meta-device init path."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.components._peft.lora import LinearLoRA, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.utils.model_utils import init_empty_weights
from nemo_automodel.shared.peft_config import PeftConfig

pytestmark = pytest.mark.cpu_only


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 8, bias=False)


def _dora_config() -> PeftConfig:
    return PeftConfig(target_modules=["q_proj"], dim=4, alpha=8, use_dora=True, use_triton=False)


def _peft_checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def _build_meta_dora_model() -> _Tiny:
    """Apply DoRA on meta and materialize, in the order the meta-device build path uses."""
    with init_empty_weights():
        model = _Tiny()
        apply_lora_to_linear_modules(model, _dora_config(), skip_freeze=True)
    Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier")
    return model


def test_meta_path_base_load_sets_dora_magnitude_from_loaded_weight(tmp_path: Path) -> None:
    torch.manual_seed(0)
    pretrained = {key: value.clone() for key, value in _Tiny().state_dict().items()}
    source = tmp_path / "source"
    source.mkdir()
    save_file(pretrained, source / "model.safetensors")

    model = _build_meta_dora_model()
    _peft_checkpointer(tmp_path).load_base_model(model, torch.device("cpu"), str(tmp_path), str(source))

    weight = pretrained["q_proj.weight"]
    torch.testing.assert_close(model.q_proj.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(model.q_proj.lora_magnitude, torch.linalg.norm(weight, dim=1))
    assert model.q_proj.lora_magnitude.requires_grad

    # With lora_B == 0 a DoRA layer must reproduce the base layer at step 0.
    assert bool((model.q_proj.lora_B.weight == 0).all())
    x = torch.randn(3, 16)
    with torch.no_grad():
        torch.testing.assert_close(model.q_proj(x), F.linear(x, weight))


def test_base_load_without_checkpoint_keeps_dora_magnitude(tmp_path: Path) -> None:
    model = _build_meta_dora_model()
    with torch.no_grad():
        model.q_proj.lora_magnitude.fill_(3.0)

    _peft_checkpointer(tmp_path).load_base_model(
        model, torch.device("cpu"), str(tmp_path), "source", load_base_model=False
    )

    torch.testing.assert_close(model.q_proj.lora_magnitude, torch.full((8,), 3.0))


@pytest.fixture
def cpu_mesh():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("weight_placement", "magnitude_placement"),
    [(Shard(0), Shard(0)), (Shard(0), Replicate()), (Shard(1), Replicate())],
    ids=["fsdp", "tp_colwise", "tp_rowwise"],
)
def test_init_dora_magnitude_keeps_dtensor_placement(cpu_mesh, weight_placement, magnitude_placement) -> None:
    torch.manual_seed(0)
    module = LinearLoRA(nn.Linear(16, 8, bias=False), dim=4, alpha=8, use_dora=True)
    weight = torch.randn(8, 16)
    module.weight = nn.Parameter(distribute_tensor(weight, cpu_mesh, [weight_placement]), requires_grad=False)
    module.lora_magnitude = nn.Parameter(
        distribute_tensor(torch.zeros(8), cpu_mesh, [magnitude_placement]), requires_grad=True
    )
    magnitude = module.lora_magnitude

    module.init_dora_magnitude()

    assert module.lora_magnitude is magnitude
    assert isinstance(magnitude, DTensor)
    assert magnitude.placements == (magnitude_placement,)
    torch.testing.assert_close(magnitude.full_tensor(), torch.linalg.norm(weight, dim=1))
