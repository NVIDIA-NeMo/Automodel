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

"""Loading a checkpoint whose experts are stored as NVFP4 / MXFP8 (released ``hf_quant_config.json`` layout).

A tiny checkpoint is written in the released key layout with layer 0's experts in NVFP4, layer 1's in MXFP8 and
the rest unquantized, then loaded through ``Checkpointer.load_model`` on both of its base-checkpoint routes.
"""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _should_dequantize_base_checkpoint,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.quantization import (
    checkpoint_expert_formats,
    dequantize_mxfp8_experts,
    dequantize_nvfp4_experts,
    quantization_config_from_manifest,
)

from .parity_check_inkling2 import TINY_TEXT_CONFIG, _randomize

pytestmark = pytest.mark.cpu_only

_BACKEND = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
_TEXT_CONFIG = {**TINY_TEXT_CONFIG, "num_hidden_layers": 3, "layer_types": ["hybrid_sliding", "hybrid", "hybrid"]}
_EXPERTS = "model.llm.layers.{}.mlp.experts.{}_weight"


def _manifest() -> dict:
    """Return a released-style manifest: layer 0 experts NVFP4, layer 1 MXFP8 with a [32, 32] block declaration."""
    nvfp4 = {
        "compute_dtype": "nvfp4",
        "output_dtype": "uint8",
        "required_activation_quantization": {"format": "nvfp4_e2m1", "block_size": 16},
        "weight_block_shape": [1, 16],
        "weight_scale_format": "amax_denominator",
        "weight_transform": "nvfp4_nearest",
        "tensors": [_EXPERTS.format(0, "w13"), _EXPERTS.format(0, "w2")],
    }
    mxfp8 = {
        "compute_dtype": "mxfp8",
        "output_dtype": "float8_e4m3fn",
        "required_activation_quantization": {"format": "mxfp8_e4m3", "block_size": 32},
        "weight_block_shape": [32, 32],
        "weight_transform": "mxfp8_nearest",
        "tensors": [_EXPERTS.format(1, "w13"), _EXPERTS.format(1, "w2")],
    }
    dense = {
        "compute_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "weight_block_shape": [],
        "weight_transform": "cast",
        "tensors": ["model.llm.embed.weight"],
    }
    return {"quant_method": "inkling2", "model": {"parameter_groups": [dense, nvfp4, mxfp8]}}


def _tiny_model(config: Inkling2Config) -> Inkling2ForConditionalGeneration:
    return Inkling2ForConditionalGeneration(config, backend=_BACKEND)


def _write_checkpoint(path: Path, *, with_manifest: bool = True) -> dict[str, torch.Tensor]:
    """Write a tiny released-layout checkpoint and return its tensors by released key.

    Layer 0's ``w13``/``w2`` hold random NVFP4 codes, E4M3 scales, per-expert amax and denominator 2688, and layer 1's
    random E4M3 values with UE8M0 exponents; every other tensor is the released-layout export of a random model.
    """
    config = Inkling2Config(text_config=_TEXT_CONFIG, vision_config=None, torch_dtype="bfloat16")
    model = _tiny_model(config)
    _randomize(model, seed=5)
    tensors = {
        key: value.detach().contiguous().clone()
        for key, value in model.state_dict_adapter.to_hf(model.state_dict()).items()
    }
    generator = torch.Generator().manual_seed(6)
    for leaf in ("w13", "w2"):
        experts, rows, columns = tensors[_EXPERTS.format(0, leaf)].shape
        key = _EXPERTS.format(0, leaf)
        tensors[key] = torch.randint(0, 256, (experts, rows, columns // 2), generator=generator, dtype=torch.uint8)
        scale = torch.randint(0x08, 0x7F, (experts, rows, columns // 16), generator=generator, dtype=torch.uint8)
        tensors[key + "_scale"] = scale.view(torch.float8_e4m3fn)
        tensors[key + "_amax"] = torch.rand(experts, generator=generator) + 0.05
        tensors[key + "_outer_scale_denominator"] = torch.tensor([2688.0])
        key = _EXPERTS.format(1, leaf)
        values = torch.randint(0, 256, (experts, rows, columns), generator=generator, dtype=torch.uint8)
        values[(values & 0x7F) == 0x7F] = 0x38  # no E4M3 NaN patterns
        tensors[key] = values.view(torch.float8_e4m3fn)
        tensors[key + "_scale"] = torch.randint(
            110, 130, (experts, rows, columns // 32), generator=generator, dtype=torch.uint8
        )
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    config_dict = config.to_dict()
    del config_dict["transformers_version"]  # as released
    (path / "config.json").write_text(json.dumps(config_dict))
    if with_manifest:
        (path / "hf_quant_config.json").write_text(json.dumps(_manifest()))
    return tensors


def _expected_experts(tensors: dict[str, torch.Tensor], layer: int, leaf: str) -> torch.Tensor:
    """Return the native ``[experts, in, out]`` BF16 expert tensor decoded straight from the stored tensors."""
    key = _EXPERTS.format(layer, leaf)
    if layer == 0:
        weight = dequantize_nvfp4_experts(
            tensors[key], tensors[key + "_scale"], tensors[key + "_amax"], tensors[key + "_outer_scale_denominator"]
        )
    else:
        weight = dequantize_mxfp8_experts(tensors[key], tensors[key + "_scale"])
    return weight.to(torch.bfloat16).transpose(-1, -2)


def _assert_loaded(model: Inkling2ForConditionalGeneration, tensors: dict[str, torch.Tensor]) -> None:
    """Every native parameter equals its stored tensor; quantized experts equal their direct decode bitwise."""
    native = model.state_dict()
    raw = model.state_dict_adapter.to_hf(native)
    for layer in (0, 1):
        for leaf, native_leaf in (("w13", "gate_and_up_projs"), ("w2", "down_projs")):
            loaded = native[f"model.language_model.layers.{layer}.mlp.experts.{native_leaf}"]
            assert torch.equal(loaded, _expected_experts(tensors, layer, leaf)), (layer, leaf)
    for key, value in raw.items():
        if key.startswith(("model.llm.layers.0.mlp.experts", "model.llm.layers.1.mlp.experts")):
            continue
        assert value.dtype == tensors[key].dtype and torch.equal(value, tensors[key]), key


def _checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="release",
            model_save_format="safetensors",
            save_consolidated=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def test_manifest_lists_the_stored_expert_formats():
    config = quantization_config_from_manifest(_manifest())
    assert config == {
        "quant_method": "inkling2",
        "expert_formats": {
            _EXPERTS.format(0, "w13"): "nvfp4",
            _EXPERTS.format(0, "w2"): "nvfp4",
            _EXPERTS.format(1, "w13"): "mxfp8",
            _EXPERTS.format(1, "w2"): "mxfp8",
        },
    }
    assert checkpoint_expert_formats(config) == config["expert_formats"]
    assert checkpoint_expert_formats(None) == {}


@pytest.mark.parametrize(
    ("group", "field", "value", "message"),
    [
        (1, "weight_block_shape", [1, 32], "nvfp4 weight precision"),
        (1, "weight_scale_format", "per_tensor", "nvfp4 weight precision"),
        (2, "weight_block_shape", [16, 16], "mxfp8 weight precision"),
        (2, "compute_dtype", "int4", "compute dtype"),
        (1, "tensors", ["model.llm.layers.0.attn.wq_du.weight"], "routed-expert"),
        (2, "tensors", [_EXPERTS.format(0, "w13")], "listed twice"),
    ],
)
def test_manifest_rejects_layouts_it_cannot_decode(group, field, value, message):
    """``group`` indexes ``_manifest()``'s parameter groups (0 dense, 1 NVFP4, 2 MXFP8); ``field`` is overwritten."""
    manifest = _manifest()
    manifest["model"]["parameter_groups"][group][field] = value
    with pytest.raises(ValueError, match=message):
        quantization_config_from_manifest(manifest)
    with pytest.raises(ValueError, match="inkling2"):
        quantization_config_from_manifest({**_manifest(), "quant_method": "fp8"})


def test_config_lookup_folds_in_the_manifest(tmp_path):
    _write_checkpoint(tmp_path / "release")
    _write_checkpoint(tmp_path / "bf16", with_manifest=False)
    expected = quantization_config_from_manifest(_manifest())

    for config in (
        get_hf_config(str(tmp_path / "release"), "eager"),  # AutoModel's registered-config lookup
        Inkling2Config.from_pretrained(tmp_path / "release"),
    ):
        assert isinstance(config, Inkling2Config) and config.quantization_config == expected
        model = _tiny_model(config)
        assert model.state_dict_adapter.expert_formats == expected["expert_formats"]
        assert _should_dequantize_base_checkpoint(model, None)

    plain = get_hf_config(str(tmp_path / "bf16"), "eager")
    assert not hasattr(plain, "quantization_config")
    assert not _should_dequantize_base_checkpoint(_tiny_model(plain), None)


def test_to_hf_emits_stored_layout_destinations_only_for_quantized_loads(tmp_path):
    _write_checkpoint(tmp_path / "release")
    model = _tiny_model(get_hf_config(str(tmp_path / "release"), "eager"))
    experts, intermediate, hidden = 8, TINY_TEXT_CONFIG["moe_intermediate_size"], TINY_TEXT_CONFIG["hidden_size"]
    raw = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True)

    expected = {
        _EXPERTS.format(0, "w13"): (torch.uint8, (experts, 2 * intermediate, hidden // 2)),
        _EXPERTS.format(0, "w13") + "_scale": (torch.float8_e4m3fn, (experts, 2 * intermediate, hidden // 16)),
        _EXPERTS.format(0, "w13") + "_amax": (torch.float32, (experts,)),
        _EXPERTS.format(0, "w13") + "_outer_scale_denominator": (torch.float32, (1,)),
        _EXPERTS.format(0, "w2"): (torch.uint8, (experts, hidden, intermediate // 2)),
        _EXPERTS.format(0, "w2") + "_scale": (torch.float8_e4m3fn, (experts, hidden, intermediate // 16)),
        _EXPERTS.format(1, "w13"): (torch.float8_e4m3fn, (experts, 2 * intermediate, hidden)),
        _EXPERTS.format(1, "w13") + "_scale": (torch.uint8, (experts, 2 * intermediate, hidden // 32)),
        _EXPERTS.format(1, "w2"): (torch.float8_e4m3fn, (experts, hidden, intermediate)),
        _EXPERTS.format(1, "w2") + "_scale": (torch.uint8, (experts, hidden, intermediate // 32)),
        _EXPERTS.format(2, "w13"): (torch.bfloat16, (experts, 2 * intermediate, hidden)),
        _EXPERTS.format(2, "w2"): (torch.bfloat16, (experts, hidden, intermediate)),
    }
    for key, (dtype, shape) in expected.items():
        assert raw[key].dtype == dtype and tuple(raw[key].shape) == shape, key
    assert _EXPERTS.format(1, "w13") + "_amax" not in raw

    saved = model.state_dict_adapter.to_hf(model.state_dict())  # saving: unquantized, no companions
    assert not any(key.endswith(("_scale", "_amax", "_outer_scale_denominator")) for key in saved)
    assert saved[_EXPERTS.format(0, "w13")].dtype == torch.bfloat16


@pytest.mark.parametrize("route", ["full_state_dict", "dcp"])
def test_checkpointer_loads_quantized_experts_bitwise(tmp_path, monkeypatch, route):
    """Both base-checkpoint routes of ``Checkpointer.load_model`` decode the stored experts exactly.

    A single-process custom model reads the full state dict; multi-rank jobs read DCP destinations from ``to_hf``.
    ``WORLD_SIZE=2`` without a process group selects the DCP route while DCP itself runs in one process.
    """
    tensors = _write_checkpoint(tmp_path / "release")
    model = _tiny_model(get_hf_config(str(tmp_path / "release"), "eager"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    if route == "dcp":
        monkeypatch.setenv("WORLD_SIZE", "2")
        with patch(
            "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
            side_effect=AssertionError("expected the DCP route"),
        ):
            _checkpointer(tmp_path).load_model(model, str(tmp_path / "release"), is_init_step=True)
    else:
        _checkpointer(tmp_path).load_model(model, str(tmp_path / "release"), is_init_step=True)
    _assert_loaded(model, tensors)


def test_bf16_export_round_trip(tmp_path):
    tensors = _write_checkpoint(tmp_path / "release")
    model = _tiny_model(get_hf_config(str(tmp_path / "release"), "eager"))
    checkpointer = _checkpointer(tmp_path)
    checkpointer.load_model(model, str(tmp_path / "release"), is_init_step=True)
    checkpointer.save_model(model, str(tmp_path / "saved"))

    exported = tmp_path / "saved/model/consolidated"
    saved_config = json.loads((exported / "config.json").read_text())
    assert "quantization_config" not in saved_config and not (exported / "hf_quant_config.json").exists()
    reloaded = _tiny_model(get_hf_config(str(exported), "eager"))
    assert reloaded.state_dict_adapter.expert_formats == {}
    _checkpointer(tmp_path).load_model(reloaded, str(exported), is_init_step=True)
    for name, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], value), name
    _assert_loaded(reloaded, tensors)


@pytest.fixture
def cpu_mesh():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("native_placement", [Shard(0), Shard(1)])
def test_sharded_destinations_keep_placements_and_decode(tmp_path, cpu_mesh, native_placement):
    """Expert-parallel (``Shard(0)``) and FSDP-style (``Shard(1)``, the input axis) experts load through DTensors."""
    tensors = _write_checkpoint(tmp_path / "release")
    model = _tiny_model(get_hf_config(str(tmp_path / "release"), "eager"))
    adapter = model.state_dict_adapter
    native = {
        key: distribute_tensor(value, cpu_mesh, [native_placement]) if ".mlp.experts." in key else value
        for key, value in model.state_dict().items()
    }
    raw = adapter.to_hf(native, quantization=True)
    stored_placement = Shard(2) if native_placement == Shard(1) else Shard(0)
    for layer, companions in ((0, ("", "_scale")), (1, ("", "_scale"))):
        for suffix in companions:
            destination = raw[_EXPERTS.format(layer, "w13") + suffix]
            assert isinstance(destination, DTensor) and destination.placements == (stored_placement,)
    amax = raw[_EXPERTS.format(0, "w13") + "_amax"]
    assert amax.placements == ((Shard(0),) if native_placement == Shard(0) else (Replicate(),))
    assert raw[_EXPERTS.format(0, "w13") + "_outer_scale_denominator"].placements == (Replicate(),)

    filled = copy.copy(raw)
    for key, destination in raw.items():
        if isinstance(destination, DTensor) and key in tensors and destination.dtype != torch.bfloat16:
            destination.to_local().copy_(tensors[key])
    loaded = adapter.from_hf(filled)
    for layer in (0, 1):
        for leaf, native_leaf in (("w13", "gate_and_up_projs"), ("w2", "down_projs")):
            value = loaded[f"model.language_model.layers.{layer}.mlp.experts.{native_leaf}"]
            assert isinstance(value, DTensor) and value.placements == (native_placement,)
            assert torch.equal(value.full_tensor(), _expected_experts(tensors, layer, leaf)), (layer, leaf)


class _RankMesh:
    """Stand-in for an expert-parallel mesh: the adapter only reads its dim names and rank here."""

    mesh_dim_names = ("ep",)

    def get_rank(self) -> int:
        return 1


def test_from_hf_under_a_mesh_slices_quantized_companions_per_expert(tmp_path, monkeypatch):
    """With plain full checkpoint tensors and a mesh, each rank decodes only its experts' bytes and amax."""
    from nemo_automodel.components.moe import state_dict_utils

    tensors = _write_checkpoint(tmp_path / "release")
    model = _tiny_model(get_hf_config(str(tmp_path / "release"), "eager"))
    start, stop = 2, 5
    monkeypatch.setattr(state_dict_utils, "get_expert_range_for_rank_from_mesh", lambda mesh, n: (start, stop))
    monkeypatch.setattr(state_dict_utils, "get_submesh", lambda mesh, dims: mesh)
    monkeypatch.setattr(state_dict_utils, "create_dtensor_from_local", lambda local, mesh, rank: local)
    loaded = model.state_dict_adapter.from_hf(dict(tensors), device_mesh=_RankMesh())
    for layer in (0, 1):
        for leaf, native_leaf in (("w13", "gate_and_up_projs"), ("w2", "down_projs")):
            value = loaded[f"model.language_model.layers.{layer}.mlp.experts.{native_leaf}"]
            assert torch.equal(value, _expected_experts(tensors, layer, leaf)[start:stop]), (layer, leaf)
