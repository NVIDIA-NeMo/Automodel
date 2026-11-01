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

"""Two-rank FSDP coverage for selected FP32 storage in Nemotron-H."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from transformers import PretrainedConfig

from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from nemo_automodel.components.models.nemotron_v3.parallelization import PARALLELIZER
from nemo_automodel.components.utils.model_utils import FreezeConfig, ModuleSelector, apply_parameter_freezing
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_model import MockNemotronV3Config

# Mamba's fused kernels read these weights directly, bypassing child FSDP unshard hooks.
_FUSED_KERNEL_MODULES = ("conv1d", "norm", "out_proj")


def _shard_mamba_block(mesh, freeze_mixer: bool) -> nn.Module:
    backend = BackendConfig(
        linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
    )
    model = NemotronHForCausalLM(
        MockNemotronV3Config(torch_dtype="float32", layers_block_type=["mamba", "attention"]), backend=backend
    )
    model.initialize_weights(dtype=torch.float32)
    apply_parameter_freezing(
        model, FreezeConfig(freeze_modules=[ModuleSelector(path="model.layers.0.mixer")] if freeze_mixer else [])
    )
    prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
    PARALLELIZER._apply(
        model,
        device_mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    )
    return model.model.layers["0"]


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (1, 2, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"))
        block = _shard_mamba_block(mesh, False)
        assert isinstance(block, FSDPModule)
        assert not any(isinstance(getattr(block.mixer, name), FSDPModule) for name in _FUSED_KERNEL_MODULES)
        assert all(getattr(block.mixer, name).weight.dtype == torch.float32 for name in _FUSED_KERNEL_MODULES)

        block = _shard_mamba_block(mesh, True)
        assert isinstance(block, FSDPModule)
        assert not any(isinstance(getattr(block.mixer, name), FSDPModule) for name in _FUSED_KERNEL_MODULES)
        assert block.norm.weight.dtype == torch.float32
        assert all(getattr(block.mixer, name).weight.dtype == torch.bfloat16 for name in _FUSED_KERNEL_MODULES)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_mamba_fp32_selection_keeps_fused_kernel_weights_in_block_unit(tmp_path: Path) -> None:
    """Whole-block and block-norm-only FP32 selections keep Mamba fused-kernel weights in the block's FSDP unit."""
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


# checkpoint format -> (layer types, selected parameter, use the HF state-dict adapter)
_FACTORY_CASES = {
    "safetensors": (["attention", "mlp"], "model.layers.0.norm.weight", False, "model.layers.1.norm.weight"),
    "bin": (
        ["mamba", "moe"],
        "model.layers.1.mixer.experts.gate_and_up_projs",
        True,
        "model.layers.1.mixer.experts.down_projs",
    ),
}


def _factory_config(checkpoint_dir: str, checkpoint_format: str) -> tuple[PretrainedConfig, BackendConfig]:
    layers, _, use_adapter, _ = _FACTORY_CASES[checkpoint_format]
    config = PretrainedConfig(**MockNemotronV3Config(layers_block_type=layers).to_dict())
    config.architectures = ["NemotronHForCausalLM"]
    config.name_or_path = checkpoint_dir
    backend = BackendConfig(
        linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch", enable_hf_state_dict_adapter=use_adapter
    )
    return config, backend


def _factory_worker(
    rank: int, rendezvous: str, checkpoint_dir: str, checkpoint_format: str, exact_allowlist: bool, expected
) -> None:
    from nemo_automodel import NeMoAutoModelForCausalLM

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        config, backend = _factory_config(checkpoint_dir, checkpoint_format)
        selected = _FACTORY_CASES[checkpoint_format][1]
        frozen = _FACTORY_CASES[checkpoint_format][3]
        freeze_config = (
            FreezeConfig(freeze_modules=[ModuleSelector(glob="*")], unfreeze_parameters=(selected,))
            if exact_allowlist
            else FreezeConfig(freeze_parameters=(frozen,))
        )
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            backend=backend,
            dtype=torch.float32,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            attn_implementation="sdpa",
            load_base_model=True,
            distributed_setup=DistributedSetup.build(
                FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
            ),
            freeze_config=freeze_config,
        )
        parameters = dict(model.named_parameters())
        assert parameters[selected].dtype == torch.float32
        assert parameters[selected].requires_grad
        torch.testing.assert_close(parameters[selected].full_tensor().cpu(), expected, rtol=0, atol=0)
        assert not parameters[frozen].requires_grad
        assert parameters[frozen].dtype == torch.bfloat16
        if exact_allowlist:
            assert {name for name, p in parameters.items() if p.requires_grad} == {selected}
            assert parameters["model.embed_tokens.weight"].dtype == torch.bfloat16
        else:
            assert all(p.requires_grad for n, p in parameters.items() if n != frozen)
            assert parameters["model.embed_tokens.weight"].dtype == torch.float32
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices for the FSDP2 factory path")
@pytest.mark.parametrize("checkpoint_format", ["safetensors", "bin"])
@pytest.mark.parametrize("exact_allowlist", [False, True])
def test_factory_loads_selected_fp32_values_exactly(
    tmp_path: Path, checkpoint_format: str, exact_allowlist: bool
) -> None:
    """Selected FP32 values, including grouped experts merged from a .bin checkpoint, load exactly."""
    from safetensors.torch import save_file

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    config, backend = _factory_config(str(checkpoint_dir), checkpoint_format)
    seed = NemotronHForCausalLM(config, backend=backend)
    seed.initialize_weights()
    selected = _FACTORY_CASES[checkpoint_format][1]
    weights = {name: value.detach().clone() for name, value in seed.state_dict().items()}
    expected = torch.linspace(0.01, 0.02, weights[selected].numel(), dtype=torch.float32).view(weights[selected].shape)
    assert not torch.equal(expected, expected.bfloat16().float())
    weights[selected] = expected.clone()
    if checkpoint_format == "bin":
        torch.save(seed.state_dict_adapter.to_hf(weights), checkpoint_dir / "pytorch_model.bin")
    else:
        save_file(weights, checkpoint_dir / "model.safetensors")
    mp.spawn(
        _factory_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(checkpoint_dir), checkpoint_format, exact_allowlist, expected),
        nprocs=2,
        join=True,
    )
