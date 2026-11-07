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

"""NVFP4 passthrough load path for Beam: routed experts load packed through the state-dict adapter, never bf16."""

import json
import re

import pytest
import torch
from torch.distributed.checkpoint.api import CheckpointException

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts_nvfp4 import GroupedExpertsLoRANVFP4
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, to_empty_parameters_only
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.beam.model import BeamForCausalLM, beam_moe_config
from nemo_automodel.components.models.beam.state_dict_adapter import BeamStateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
from nemo_automodel.components.moe.quantized_experts import (
    GroupedExpertsNVFP4,
    apply_nvfp4_to_moe_experts,
    packed_expert_parameters,
)
from nemo_automodel.components.quantization.nvfp4 import NVFP4_BLOCK_SIZE, dequantize_nvfp4, quantize_nvfp4_modelopt

from .utils import hf_model_and_checkpoint

MOE_LAYERS = (1, 2, 3)  # the tiny config's layer 0 is dense
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("weight", "weight_scale", "weight_scale_2")
_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _nvfp4_backend() -> BackendConfig:
    """The production expert backend of the NVFP4 recipe (the packed GEMM needs ``torch_mm``)."""
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch_mm",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
    )


@pytest.fixture
def adapter(beam_config):
    return BeamStateDictAdapter(beam_config, beam_moe_config(beam_config, torch.float32), _nvfp4_backend())


@pytest.fixture
def nvfp4_checkpoint(beam_config):
    """ModelOpt NVFP4 layout of the tiny HF Beam checkpoint: experts quantized per projection, the rest unchanged.

    Returns the floating-point HF state dict, the NVFP4 state dict and the per-expert quantized parts keyed by
    ``(layer, expert, projection)`` as ``(weight, weight_scale, weight_scale_2)``.
    """
    sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
    checkpoint, parts = {}, {}
    for key, value in sd.items():
        match = _EXPERT_KEY.match(key)
        if match is None:
            checkpoint[key] = value
            continue
        packed, scales, tensor_scale = quantize_nvfp4_modelopt(value.float())
        base = key[: -len(".weight")]
        checkpoint[base + ".weight"] = packed
        checkpoint[base + ".weight_scale"] = scales
        checkpoint[base + ".weight_scale_2"] = tensor_scale
        checkpoint[base + ".input_scale"] = torch.tensor(0.01)  # inference-only entry, dropped on load
        parts[(int(match.group(1)), int(match.group(2)), match.group(3))] = (packed, scales, tensor_scale)
    return sd, checkpoint, parts


def _expected_packed(parts, beam_config) -> dict[str, torch.Tensor]:
    """Grouped packed parameters the adapter must build: experts stacked, gate rows before up rows."""
    n_experts, inter, hidden = beam_config.num_experts, beam_config.moe_intermediate_size, beam_config.hidden_size
    expected = {}
    for layer in MOE_LAYERS:
        gate_up = {name: [] for name in ("_nvfp4", "_scales", "_global")}
        down = {name: [] for name in ("_nvfp4", "_scales", "_global")}
        for e in range(n_experts):
            gate, up, dn = (parts[(layer, e, projection)] for projection in PROJECTIONS)
            gate_up["_nvfp4"].append(torch.cat([gate[0], up[0]], dim=0))
            gate_up["_scales"].append(torch.cat([gate[1], up[1]], dim=0))
            gate_up["_global"].append(torch.cat([gate[2].expand(inter), up[2].expand(inter)], dim=0))
            down["_nvfp4"].append(dn[0])
            down["_scales"].append(dn[1])
            down["_global"].append(dn[2].expand(hidden))
        for name, pieces in gate_up.items():
            expected[f"model.layers.{layer}.mlp.experts.gate_and_up_projs{name}"] = torch.stack(pieces)
        for name, pieces in down.items():
            expected[f"model.layers.{layer}.mlp.experts.down_projs{name}"] = torch.stack(pieces)
    return expected


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, key: str) -> None:
    assert actual.dtype == expected.dtype, (key, actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
    assert torch.equal(
        actual.reshape(-1).contiguous().view(torch.uint8), expected.reshape(-1).contiguous().view(torch.uint8)
    ), key


def test_set_expert_storage_format(adapter):
    assert adapter.expert_storage_format == "bf16"
    adapter.set_expert_storage_format("nvfp4")
    assert adapter.expert_storage_format == "nvfp4"
    with pytest.raises(ValueError, match="Unsupported expert storage format"):
        adapter.set_expert_storage_format("mxfp4")
    assert (
        BeamStateDictAdapter(
            adapter.config, adapter.moe_config, adapter.backend, expert_storage_format="nvfp4"
        ).expert_storage_format
        == "nvfp4"
    )


def test_from_hf_emits_packed_keys_and_passes_everything_else(adapter, beam_config, nvfp4_checkpoint):
    sd, checkpoint, parts = nvfp4_checkpoint
    adapter.set_expert_storage_format("nvfp4")

    native = adapter.from_hf(dict(checkpoint))

    expected = _expected_packed(parts, beam_config)
    for key, value in expected.items():
        _assert_same_bytes(native[key], value, key)
    inter, hidden, n_experts = beam_config.moe_intermediate_size, beam_config.hidden_size, beam_config.num_experts
    assert native["model.layers.1.mlp.experts.gate_and_up_projs_nvfp4"].shape == (n_experts, 2 * inter, hidden // 2)
    assert native["model.layers.1.mlp.experts.gate_and_up_projs_scales"].shape == (
        n_experts,
        2 * inter,
        hidden // NVFP4_BLOCK_SIZE,
    )
    assert native["model.layers.1.mlp.experts.down_projs_global"].shape == (n_experts, hidden)
    # Never materialized in floating point, and no per-expert key (including input_scale) leaks through.
    assert not any(key.endswith(("gate_and_up_projs", "down_projs")) for key in native)
    assert not any(re.search(r"\.experts\.\d+\.", key) for key in native)
    # Every non-expert tensor passes through unchanged.
    for key, value in sd.items():
        if _EXPERT_KEY.match(key) is None:
            assert torch.equal(native[key], value), key
    assert set(native) == set(expected) | {key for key in sd if _EXPERT_KEY.match(key) is None}


def test_from_hf_decodes_like_the_per_expert_checkpoint_tensors(adapter, beam_config, nvfp4_checkpoint):
    """Decoding the grouped parameters reproduces each expert's own decode, so the fused row scales are exact."""
    _, checkpoint, parts = nvfp4_checkpoint
    adapter.set_expert_storage_format("nvfp4")
    native = adapter.from_hf(dict(checkpoint))

    inter = beam_config.moe_intermediate_size
    for layer in MOE_LAYERS:
        base = f"model.layers.{layer}.mlp.experts"
        gate_up = dequantize_nvfp4(
            native[f"{base}.gate_and_up_projs_nvfp4"],
            native[f"{base}.gate_and_up_projs_scales"],
            native[f"{base}.gate_and_up_projs_global"],
            torch.bfloat16,
        )
        down = dequantize_nvfp4(
            native[f"{base}.down_projs_nvfp4"],
            native[f"{base}.down_projs_scales"],
            native[f"{base}.down_projs_global"],
            torch.bfloat16,
        )
        for e in range(beam_config.num_experts):
            gate, up, dn = (parts[(layer, e, projection)] for projection in PROJECTIONS)
            assert torch.equal(
                gate_up[e, :inter], dequantize_nvfp4(gate[0], gate[1], gate[2].expand(inter), torch.bfloat16)
            )
            assert torch.equal(gate_up[e, inter:], dequantize_nvfp4(up[0], up[1], up[2].expand(inter), torch.bfloat16))
            assert torch.equal(
                down[e], dequantize_nvfp4(dn[0], dn[1], dn[2].expand(beam_config.hidden_size), torch.bfloat16)
            )


def test_round_trip_to_hf_recovers_the_checkpoint(adapter, nvfp4_checkpoint):
    """from_hf (aggregate) -> to_hf (split) reproduces the per-expert ModelOpt tensors bit-exactly."""
    _, checkpoint, _ = nvfp4_checkpoint
    adapter.set_expert_storage_format("nvfp4")

    back = adapter.to_hf(adapter.from_hf(dict(checkpoint)))

    for key, value in checkpoint.items():
        if key.endswith(".input_scale"):
            assert key not in back
            continue
        _assert_same_bytes(back[key], value, key)
    assert set(back) == {key for key in checkpoint if not key.endswith(".input_scale")}
    assert back["model.layers.1.mlp.experts.0.gate_proj.weight_scale_2"].shape == ()
    assert not adapter.to_hf(adapter.from_hf(dict(checkpoint)), exclude_key_regex=r".*\.experts\.\d+\..*|.*")


def test_to_hf_splits_packed_keys_regardless_of_storage_format(adapter, nvfp4_checkpoint):
    """Packed parameters only exist on NVFP4-resident modules, so their export does not depend on the load mode."""
    _, checkpoint, _ = nvfp4_checkpoint
    loader = BeamStateDictAdapter(adapter.config, adapter.moe_config, adapter.backend, expert_storage_format="nvfp4")
    native = loader.from_hf(dict(checkpoint))
    assert adapter.expert_storage_format == "bf16"
    back = adapter.to_hf(native)
    assert back["model.layers.2.mlp.experts.5.down_proj.weight"].dtype == torch.uint8
    assert torch.equal(
        back["model.layers.2.mlp.experts.5.down_proj.weight"],
        checkpoint["model.layers.2.mlp.experts.5.down_proj.weight"],
    )


def test_to_hf_refuses_differing_row_scales(adapter, nvfp4_checkpoint):
    _, checkpoint, _ = nvfp4_checkpoint
    adapter.set_expert_storage_format("nvfp4")
    native = adapter.from_hf(dict(checkpoint))
    native["model.layers.1.mlp.experts.down_projs_global"][0, 3] *= 2
    with pytest.raises(ValueError, match="per-row tensor scales differ"):
        adapter.to_hf(native)


def test_from_hf_reports_missing_and_malformed_expert_tensors(adapter, nvfp4_checkpoint):
    _, checkpoint, _ = nvfp4_checkpoint
    adapter.set_expert_storage_format("nvfp4")

    missing = dict(checkpoint)
    del missing["model.layers.2.mlp.experts.3.up_proj.weight_scale"]
    with pytest.raises(RuntimeError, match=r"missing from checkpoint.*experts\.3\.up_proj\.weight_scale"):
        adapter.from_hf(missing)

    floating = dict(checkpoint)
    floating["model.layers.1.mlp.experts.0.gate_proj.weight"] = torch.zeros(32, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected uint8 packed fp4 codes"):
        adapter.from_hf(floating)

    wrong_shape = dict(checkpoint)
    wrong_shape["model.layers.1.mlp.experts.0.down_proj.weight"] = torch.zeros(64, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="packed shape"):
        adapter.from_hf(wrong_shape)

    wrong_block = dict(checkpoint)
    wrong_block["model.layers.1.mlp.experts.0.down_proj.weight_scale"] = torch.zeros(64, 1, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="block-scale shape"):
        adapter.from_hf(wrong_block)

    bad_scale = dict(checkpoint)
    bad_scale["model.layers.1.mlp.experts.0.up_proj.weight_scale_2"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        adapter.from_hf(bad_scale)


def _nvfp4_model(beam_config, lora: bool) -> BeamForCausalLM:
    """Native Beam on meta with NVFP4-resident routed experts, as ``apply_model_infrastructure`` builds it."""
    with torch.device("meta"):
        model = BeamForCausalLM(beam_config, backend=_nvfp4_backend())
        if lora:
            for layer in model.model.layers.values():
                if hasattr(layer.mlp, "experts"):
                    layer.mlp.experts = patch_moe_module(
                        layer.mlp.experts, dim=8, alpha=16, expert_weight_format="nvfp4"
                    )
        apply_nvfp4_to_moe_experts(model)
    assert model.state_dict_adapter.expert_storage_format == "nvfp4"
    for layer in MOE_LAYERS:
        experts = model.model.layers[str(layer)].mlp.experts
        assert isinstance(experts, GroupedExpertsLoRANVFP4 if lora else GroupedExpertsNVFP4)
        assert experts.gate_and_up_projs_nvfp4.is_meta
    return model


def _materialize(model: BeamForCausalLM) -> None:
    """Materialize the meta model on CPU the way the infrastructure does before the base load.

    ``prepare_frozen_parameters_for_sharding`` runs before sharding in production and records the storage dtypes
    that the whole-model cast inside ``initialize_weights`` must keep (the packed codes and scales among them).
    """
    prepare_frozen_parameters_for_sharding(model, torch.float32, keep_storage=packed_expert_parameters(model))
    to_empty_parameters_only(model, device=torch.device("cpu"))
    for module in model.modules():
        for key, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[key] = torch.empty_like(buffer, device="cpu")
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)


def _assert_packed_dtypes(model: BeamForCausalLM) -> None:
    for layer in MOE_LAYERS:
        experts = model.model.layers[str(layer)].mlp.experts
        for base in ("gate_and_up_projs", "down_projs"):
            assert getattr(experts, base + "_nvfp4").dtype == torch.uint8
            assert getattr(experts, base + "_scales").dtype == torch.float8_e4m3fn
            assert getattr(experts, base + "_global").dtype == torch.float32


def test_hf_state_dict_keys_of_the_meta_model(beam_config):
    """Before sharding, the infrastructure lists the HF keys of the shape-only (meta) model: no value access allowed."""
    model = _nvfp4_model(beam_config, lora=False)
    keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    expert_keys = [key for key in keys if re.search(r"\.experts\.\d+\.", key)]
    assert len(expert_keys) == len(MOE_LAYERS) * beam_config.num_experts * len(PROJECTIONS) * len(SUFFIXES)
    assert "model.layers.3.mlp.experts.7.down_proj.weight_scale_2" in expert_keys
    assert not any(key.endswith(("_nvfp4", "_scales", "_global")) for key in keys)
    assert "model.layers.0.mlp.gate_proj.weight" in keys and "lm_head.weight" in keys


def test_initialize_weights_keeps_packed_dtypes_without_sharding_preparation(beam_config):
    """A run that never calls prepare_frozen_parameters_for_sharding (no FSDP2 strategy) still materialises the
    packed experts with their checkpoint dtypes: the mixin records them when the parameters are registered."""
    model = _nvfp4_model(beam_config, lora=False)
    to_empty_parameters_only(model, device=torch.device("cpu"))
    for module in model.modules():
        for key, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[key] = torch.empty_like(buffer, device="cpu")
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    _assert_packed_dtypes(model)
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16


def test_checkpoint_load_destinations_write_through_to_packed_experts(beam_config):
    """DCP destinations for the codes and block scales are views into the model's packed storage."""
    model = _nvfp4_model(beam_config, lora=False)
    _materialize(model)
    _assert_packed_dtypes(model)
    state = model.state_dict()

    destinations = model.state_dict_adapter.to_hf(state, for_checkpoint_load=True)

    packed_storage = {
        value.untyped_storage().data_ptr() for key, value in state.items() if key.endswith(("_nvfp4", "_scales"))
    }
    expert_destinations = [key for key in destinations if re.search(r"\.experts\.\d+\.", key)]
    assert len(expert_destinations) == len(MOE_LAYERS) * beam_config.num_experts * len(PROJECTIONS) * len(SUFFIXES)
    for key in expert_destinations:
        destination = destinations[key]
        if key.endswith(".weight_scale_2"):
            assert destination.shape == () and destination.dtype == torch.float32
        else:
            assert destination.untyped_storage().data_ptr() in packed_storage, key
            assert destination.dtype == (torch.uint8 if key.endswith(".weight") else torch.float8_e4m3fn)
    with torch.no_grad():
        destinations["model.layers.1.mlp.experts.2.up_proj.weight"].fill_(0xAB)
    inter = beam_config.moe_intermediate_size
    written = model.model.layers["1"].mlp.experts.gate_and_up_projs_nvfp4[2, inter:]
    assert torch.equal(written, torch.full_like(written, 0xAB))
    # CPU model storage is not registered as loaded in place (the loader rebuilds from the filled views).
    assert not getattr(model.state_dict_adapter, "_inplace_loaded_native_keys", None)


def test_from_hf_skips_rebuild_of_destinations_loaded_in_place(beam_config, nvfp4_checkpoint):
    """On GPU / EP the codes and block scales are registered as loaded in place (DCP wrote through the views):
    from_hf must then rebuild only the per-row scales and report the in-place keys as loaded, not as missing."""
    _, checkpoint, parts = nvfp4_checkpoint
    model = _nvfp4_model(beam_config, lora=False)
    _materialize(model)
    adapter = model.state_dict_adapter
    destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    # What DCP does: fill every destination from the checkpoint (the views write into the model's storage).
    with torch.no_grad():
        for key, destination in destinations.items():
            if re.search(r"\.experts\.\d+\.", key):
                destination.copy_(checkpoint[key])
    # What the CUDA / DTensor path does in to_hf: register the codes and block scales as loaded in place.
    in_place = {
        f"model.layers.{layer}.mlp.experts.{base}{suffix}"
        for layer in MOE_LAYERS
        for base in ("gate_and_up_projs", "down_projs")
        for suffix in ("_nvfp4", "_scales")
    }
    for fqn in in_place:
        adapter._register_inplace_loaded_key(fqn, None)

    native = adapter.from_hf(destinations)

    assert adapter.view_loaded_native_keys == in_place
    assert not getattr(adapter, "_inplace_loaded_native_keys", None)
    assert not any(key.endswith(("_nvfp4", "_scales")) for key in native)
    expected = _expected_packed(parts, beam_config)
    for key, value in expected.items():
        if key.endswith("_global"):
            _assert_same_bytes(native[key], value, key)
        else:
            assert key not in native
            _assert_same_bytes(model.state_dict()[key], value, key)  # written through the views


@pytest.mark.parametrize("route", ["dcp", "dcp_quantized_config", "full_cpu"])
@pytest.mark.parametrize("lora", [False, True])
def test_checkpointer_loads_nvfp4_experts_through_the_base_load(
    beam_config, nvfp4_checkpoint, tmp_path, monkeypatch, route, lora
):
    """The standard base load fills the packed experts (and everything else) from a ModelOpt NVFP4 checkpoint.

    ``dcp``: the low-memory DCP path. ``dcp_quantized_config``: the same path when the checkpoint's config
    declares a quantization method (several ranks). ``full_cpu``: the one-rank full-state conversion.
    """
    from safetensors.torch import save_file

    sd, checkpoint, parts = nvfp4_checkpoint
    checkpoint_dir = tmp_path / "nvfp4"
    checkpoint_dir.mkdir()
    save_file(checkpoint, str(checkpoint_dir / "model.safetensors"))
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": NVFP4_BLOCK_SIZE}})
    )
    model = _nvfp4_model(beam_config, lora=lora)
    if route != "dcp":
        model.config.quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    if route == "dcp_quantized_config":
        monkeypatch.setenv("WORLD_SIZE", "8")
    _materialize(model)
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "out"),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path),
            model_repo_id=str(checkpoint_dir),
            save_consolidated=False,
            is_peft=lora,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )

    checkpointer.load_model(model, model_path=str(checkpoint_dir), is_init_step=True)

    _assert_packed_dtypes(model)
    loaded = model.state_dict()
    for key, value in _expected_packed(parts, beam_config).items():
        _assert_same_bytes(loaded[key], value, key)
    for key in (
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.layers.2.self_attn.gate_bias",
        "model.layers.3.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        torch.testing.assert_close(loaded[key].float(), sd[key].float(), rtol=0, atol=0, msg=key)
    assert not any(key.endswith(("gate_and_up_projs", "down_projs")) for key in loaded)


def test_base_load_refuses_a_floating_point_checkpoint(beam_config, tmp_path):
    """A bf16 per-expert checkpoint neither matches the packed shapes nor carries block scales."""
    from safetensors.torch import save_file

    sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))[1]
    checkpoint_dir = tmp_path / "bf16"
    checkpoint_dir.mkdir()
    save_file(sd, str(checkpoint_dir / "model.safetensors"))
    model = _nvfp4_model(beam_config, lora=False)
    _materialize(model)
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "out"),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path),
            model_repo_id=str(checkpoint_dir),
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    # DCP wraps the planner's shape check in a BaseException subclass.
    with pytest.raises(CheckpointException, match=r"Size mismatch.*\.experts\.\d+\.(gate|up|down)_proj\.weight"):
        checkpointer.load_model(model, model_path=str(checkpoint_dir), is_init_step=True)
