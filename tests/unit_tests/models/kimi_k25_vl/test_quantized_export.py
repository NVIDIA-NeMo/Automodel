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

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.kimi_k25_vl.model import KimiK25VLConfig
from nemo_automodel.components.models.kimi_k25_vl.state_dict_adapter import KimiK25VLStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _adapter():
    moe = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        norm_topk_prob=True,
    )
    return KimiK25VLStateDictAdapter(
        KimiK25VLConfig(),
        moe,
        BackendConfig(linear="torch", rms_norm="torch", attn="sdpa"),
        dtype=torch.bfloat16,
    )


class _TinyModel(torch.nn.Module, HFCheckpointingMixin):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(4078)
        shapes = {
            "model.language_model.model.layers.5.mlp.experts.gate_and_up_projs": (4, 64, 64),
            "model.language_model.model.layers.5.mlp.experts.down_projs": (4, 32, 64),
            "model.language_model.model.layers.5.mlp.shared_experts.gate_proj.weight": (32, 64),
            "model.language_model.model.layers.5.mlp.gate.weight": (4, 64),
            "model.language_model.model.layers.5.self_attn.q_proj.weight": (32, 64),
            "model.language_model.model.layers.0.mlp.gate_proj.weight": (32, 64),
            "model.language_model.lm_head.weight": (32, 64),
            "model.vision_tower.proj.weight": (32, 64),
            "model.multi_modal_projector.linear_1.weight": (32, 64),
        }
        for key, shape in shapes.items():
            parts = key.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            tensor = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
            parent.register_parameter(parts[-1], torch.nn.Parameter(tensor))
        self.config = KimiK25VLConfig(text_config={"hidden_size": 64, "quantization_config": {"stale": True}})
        self.state_dict_adapter = _adapter()
        self._pre_shard_hf_state_dict_keys = list(self.state_dict_adapter.to_hf(self.state_dict()))


@pytest.mark.parametrize("nested", [False, True])
def test_metadata_matches_int4_targets_without_mutation(nested):
    adapter = _adapter()
    config = {"text_config": {"hidden_size": 64}, "auto_map": {"AutoConfig": "custom.Config"}}
    (config["text_config"] if nested else config)["quantization_config"] = {"stale": True}
    original = deepcopy(config)
    exported = adapter.adapt_hf_config_for_save(config, quantization=True)
    scheme = exported["quantization_config"]
    assert scheme["format"] == "pack-quantized"
    assert scheme["config_groups"]["group_0"]["weights"] == {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "dynamic": False,
    }
    target = scheme["config_groups"]["group_0"]["targets"][0].removeprefix("re:")
    for name in adapter.to_hf(_TinyModel().state_dict()):
        assert bool(re.fullmatch(target, name.removesuffix(".weight"))) == adapter._is_quantized_expert_key(name)
    for name in (
        "language_model.model.layers.0.mlp.experts.0.gate_proj",
        "language_model.model.layers.5.mlp.experts.0.gate_proj.lora_A",
    ):
        assert re.fullmatch(target, name) is None
    assert "quantization_config" not in exported["text_config"]
    plain = adapter.adapt_hf_config_for_save(config)
    assert "quantization_config" not in plain
    assert "quantization_config" not in plain["text_config"]
    assert config == original
    assert exported["auto_map"] == original["auto_map"]


@pytest.mark.parametrize("quantization", [None, False, True], ids=["default", "bf16", "int4"])
@pytest.mark.parametrize("v4_compatible", [False, True])
def test_save_pretrained_weights_config_index_and_reload(tmp_path, quantization, v4_compatible):
    """Use the real save API and DCP loader, without an existing checkpoint or IO mocks."""
    model = _TinyModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    source = tmp_path / "source"
    source.mkdir()
    model.config.name_or_path = str(source)
    config_before = deepcopy(model.config.to_dict())
    # Exercise the original, nested HF v4 config as well as normalized local metadata.
    source_config = deepcopy(config_before)
    source_config["text_config"]["quantization_config"] = source_config.pop("quantization_config")
    (source / "config.json").write_text(json.dumps(source_config))
    packed_source = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True)
    save_file({key: value.contiguous() for key, value in packed_source.items()}, str(source / "model.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(packed_source, "model.safetensors")})
    )
    checkpointer = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        model_repo_id=str(source),
        model_cache_dir=str(source),
        model_save_format="safetensors",
        save_consolidated=True,
        v4_compatible=v4_compatible,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    kwargs = {} if quantization is None else {"quantization": quantization}
    try:
        model.save_pretrained(str(tmp_path / "export"), checkpointer=checkpointer, **kwargs)
    finally:
        checkpointer.close()
    output = tmp_path / "export/model/consolidated"
    expected = model.state_dict_adapter.to_hf(model.state_dict(), quantization=bool(quantization))
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(expected)
    actual = {}
    for filename in set(index.values()):
        actual.update(load_file(str(output / filename)))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if key.endswith("weight_packed"):
            assert actual[key].dtype == torch.int32
        elif key.endswith("weight_scale"):
            assert actual[key].dtype == torch.float16
        elif key.endswith("weight_shape"):
            assert actual[key].dtype == torch.int64
        else:
            assert actual[key].dtype == torch.bfloat16
    for name in ("config.json", "config.v5.json") if v4_compatible else ("config.json",):
        config = json.loads((output / name).read_text())
        assert ("quantization_config" in config) is bool(quantization)
        assert "quantization_config" not in config["text_config"]
        reloaded_config = KimiK25VLConfig.from_dict(config)
        assert (getattr(reloaded_config, "quantization_config", None) is not None) is bool(quantization)
    assert model.config.to_dict() == config_before
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    expected_native = _adapter().from_hf(actual)
    destination = _TinyModel()
    with torch.no_grad():
        for parameter in destination.parameters():
            parameter.zero_()
    loader = CheckpointingConfig(checkpoint_dir=str(tmp_path), dequantize_base_checkpoint=bool(quantization)).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    try:
        loader.load_model(destination, str(output), is_init_step=True)
    finally:
        loader.close()
    assert destination.state_dict().keys() == expected_native.keys()
    for key, value in destination.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_native[key].cpu(), rtol=0, atol=0)


def _distributed_export_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        model = _TinyModel()
        expected = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True)
        for key, value in list(model.state_dict().items()):
            if ".experts." in key:
                parent_path, name = key.rsplit(".", 1)
                model.get_submodule(parent_path).register_parameter(
                    name, torch.nn.Parameter(distribute_tensor(value, mesh, [Shard(0)]))
                )
        checkpointer = CheckpointingConfig(checkpoint_dir=output_dir, save_consolidated=True).build(
            dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=mesh
        )
        try:
            model.save_pretrained(output_dir, checkpointer=checkpointer, quantization=True)
        finally:
            checkpointer.close()
        output = Path(output_dir) / "model/consolidated"
        index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
        assert set(index) == set(expected)
        actual = {}
        for filename in set(index.values()):
            actual.update(load_file(str(output / filename)))
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(45, reason="Two real Gloo ranks exercise Kimi K2.5 expert-parallel INT4 saving.")
def test_expert_parallel_save_pretrained_includes_all_experts(tmp_path):
    mp.spawn(
        _distributed_export_worker, args=(str(tmp_path / "gloo_init"), str(tmp_path / "export")), nprocs=2, join=True
    )
