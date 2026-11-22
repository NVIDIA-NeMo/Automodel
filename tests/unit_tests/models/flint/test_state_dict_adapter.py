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

"""Strict HF/native conversion, streamed export, checkpoint diagnostics and expert restore destinations."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorProperties, TensorStorageMetadata

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.model import FlintForCausalLM
from nemo_automodel.components.models.flint.state_dict_adapter import FlintStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from tests.functional_tests.models.flint import run_parallel_composition as composition
from tests.functional_tests.models.flint.test_checkpoint_resume import _compare


def _config() -> FlintConfig:
    return FlintConfig(
        vocab_size=17,
        hidden_size=6,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0],
        moe_intermediate_size=3,
        shared_expert_intermediate_size=5,
        num_experts=4,
        num_experts_per_tok=2,
        sliding_window=3,
        use_cache=False,
    )


def _adapter(dtype: torch.dtype = torch.float32) -> FlintStateDictAdapter:
    config = _config()
    moe_config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=6,
        inter_dim=10,
        moe_inter_dim=3,
        norm_topk_prob=True,
        shared_expert_inter_dim=5,
        dtype=dtype,
    )
    return FlintStateDictAdapter(
        config,
        moe_config,
        BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch", rope_fusion=False),
        dtype=dtype,
    )


def _source(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Create every tiny HF tensor with distinct finite source values.

    Args:
        dtype: BF16 base-checkpoint or FP32 resume-checkpoint storage dtype.

    Returns:
        Complete HF parameter mapping. Linear matrices use [out, in], norms
        use [hidden] or [head_dim], and sink logits use [query_heads].
    """
    generator = torch.Generator().manual_seed(724)
    tensors = {}

    def insert(key: str, shape: tuple[int, ...]) -> None:
        tensors[key] = torch.randn(shape, generator=generator).to(dtype)

    insert("model.embed_tokens.weight", (17, 6))
    insert("lm_head.weight", (17, 6))
    insert("model.norm.weight", (6,))
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        insert(prefix + "input_layernorm.weight", (6,))
        insert(prefix + "post_attention_layernorm.weight", (6,))
        for projection, shape in [("q", (8, 6)), ("k", (4, 6)), ("v", (4, 6)), ("o", (6, 8))]:
            insert(prefix + f"self_attn.{projection}_proj.weight", shape)
        for projection in ["q", "k"]:
            insert(prefix + f"self_attn.{projection}_norm.weight", (4,))
        insert(prefix + "self_attn.softmax_offset", (2,))
        modules = [("mlp", 10)] if layer == 0 else [("mlp.shared_expert", 5)]
        if layer != 0:
            insert(prefix + "mlp.gate.weight", (4, 6))
            modules += [(f"mlp.experts.{expert}", 3) for expert in range(4)]
        for module, width in modules:
            insert(prefix + module + ".gate_proj.weight", (width, 6))
            insert(prefix + module + ".up_proj.weight", (width, 6))
            insert(prefix + module + ".down_proj.weight", (6, width))
    return tensors


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_complete_split_expert_round_trip_preserves_names_values_and_precision(dtype: torch.dtype) -> None:
    adapter = _adapter(dtype)
    source = _source(dtype)
    native = adapter.from_hf(dict(source))
    exported = adapter.to_hf(native)
    assert exported.keys() == source.keys()
    for name, tensor in source.items():
        # Sinks/head intentionally acquire protected FP32 resident storage.
        # Source BF16 is restored at consolidated export, not in this hook.
        expected_dtype = torch.float32 if name.endswith(".softmax_offset") or name == "lm_head.weight" else dtype
        assert exported[name].dtype == expected_dtype
        torch.testing.assert_close(exported[name], tensor.to(expected_dtype), rtol=0, atol=0)
    assert native["model.layers.1.mlp.experts.gate_and_up_projs"].shape == (4, 6, 6)
    assert native["model.layers.1.mlp.experts.down_projs"].shape == (4, 3, 6)
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in native
    assert "model.layers.1.self_attn._fp32_params.softmax_offset" in native
    assert (
        native["lm_head.weight"].untyped_storage().data_ptr()
        != native["model.embed_tokens.weight"].untyped_storage().data_ptr()
    )


def test_grouped_axis_preserves_global_expert_ids_and_gate_before_up() -> None:
    adapter = _adapter()
    source = _source(torch.float32)
    for expert in range(4):
        prefix = f"model.layers.1.mlp.experts.{expert}."
        source[prefix + "gate_proj.weight"].fill_(10 + expert)
        source[prefix + "up_proj.weight"].fill_(20 + expert)
        source[prefix + "down_proj.weight"].fill_(30 + expert)
    native = adapter.from_hf(source)
    gate_up, down = (
        native["model.layers.1.mlp.experts.gate_and_up_projs"],
        native["model.layers.1.mlp.experts.down_projs"],
    )
    for expert in range(4):
        assert torch.all(gate_up[expert, :, :3] == 10 + expert)
        assert torch.all(gate_up[expert, :, 3:] == 20 + expert)
        assert torch.all(down[expert] == 30 + expert)


def test_fp32_training_resume_does_not_round_to_bf16() -> None:
    adapter = _adapter(torch.float32)
    source = _source(torch.float32)
    key = "model.layers.1.self_attn.softmax_offset"
    source[key] = torch.tensor([1.000001, -0.500001])
    source["lm_head.weight"][0, 0] = 1.000001
    source["model.layers.1.mlp.experts.0.gate_proj.weight"][0, 0] = 1.000001
    native = adapter.from_hf(dict(source))
    exported = adapter.to_hf(native)
    assert not torch.equal(source[key], source[key].bfloat16().float())
    for name in source:
        assert exported[name].dtype == torch.float32
        torch.testing.assert_close(exported[name], source[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.3.up_proj.weight",
        "model.layers.2.self_attn.k_norm.weight",
        "lm_head.weight",
    ],
)
def test_missing_ordinary_router_or_expert_tensor_is_rejected(key: str) -> None:
    state = _source(torch.bfloat16)
    del state[key]
    with pytest.raises(ValueError, match="Missing Flint checkpoint tensors"):
        _adapter().from_hf(state)


def test_missing_entire_moe_layer_is_rejected() -> None:
    state = {key: value for key, value in _source(torch.float32).items() if not key.startswith("model.layers.1.")}
    with pytest.raises(ValueError, match="Missing Flint checkpoint tensors"):
        _adapter().from_hf(state)


@pytest.mark.parametrize(
    "key",
    [
        "unexpected.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.layers.1.mlp.shared_expert_gate.weight",
        "model.layers.1.mlp.experts.4.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
    ],
)
def test_unknown_or_architecture_changing_keys_are_rejected(key: str) -> None:
    state = _source(torch.float32)
    state[key] = torch.zeros(1)
    with pytest.raises(ValueError, match="Unexpected Flint checkpoint tensor"):
        _adapter().from_hf(state)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("model.layers.0.self_attn.q_proj.weight", (6, 6)),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", (6, 3)),
        ("model.layers.1.mlp.shared_expert.up_proj.weight", (3, 6)),
        ("model.layers.2.self_attn.q_norm.weight", (8,)),
        ("lm_head.weight", (18, 6)),
    ],
)
def test_wrong_projection_shared_head_or_norm_shapes_are_rejected(key: str, shape: tuple[int, ...]) -> None:
    state = _source(torch.float32)
    state[key] = torch.zeros(shape)
    with pytest.raises(ValueError, match="has shape"):
        _adapter().from_hf(state)


def test_integer_checkpoint_tensors_are_rejected() -> None:
    state = _source(torch.float32)
    state["lm_head.weight"] = state["lm_head.weight"].to(torch.int32)
    with pytest.raises(TypeError, match="BF16 or FP32"):
        _adapter().from_hf(state)


def test_load_destinations_write_through_and_from_hf_does_not_rebuild_experts() -> None:
    adapter = _adapter()
    native = {key: torch.full_like(value, math.nan) for key, value in adapter.from_hf(_source(torch.float32)).items()}
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    assert adapter.supports_low_memory_dcp_load
    native_storages = {value.untyped_storage().data_ptr() for value in native.values()}
    for number, destination in enumerate(destinations.values(), start=1):
        assert destination.untyped_storage().data_ptr() in native_storages
        destination.fill_(number)
    assert all(torch.isfinite(value).all() for value in native.values())
    before = {key: value.clone() for key, value in native.items()}
    with patch.object(
        adapter, "_direct_fill_grouped_expert_tensor", side_effect=AssertionError("Unexpected expert rebuild")
    ):
        remaining = adapter.from_hf(destinations)
    expected_grouped = {key for key in native if key.endswith((".gate_and_up_projs", ".down_projs"))}
    assert adapter.view_loaded_native_keys == expected_grouped
    assert expected_grouped.isdisjoint(remaining)
    for key in native:
        torch.testing.assert_close(native[key], before[key], rtol=0, atol=0)
    assert all(
        remaining[key].untyped_storage().data_ptr() == native[key].untyped_storage().data_ptr() for key in remaining
    )


def test_failed_load_clears_view_bookkeeping_for_the_next_load() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    del destinations["lm_head.weight"]
    with pytest.raises(ValueError, match="Missing Flint checkpoint tensors"):
        adapter.from_hf(destinations)
    assert not adapter.view_loaded_native_keys
    restored = adapter.from_hf(_source(torch.float32))
    assert "model.layers.1.mlp.experts.gate_and_up_projs" in restored


def test_interrupted_read_does_not_skip_experts_from_a_fresh_source_mapping() -> None:
    adapter = _adapter()
    native = {key: torch.full_like(value, math.nan) for key, value in adapter.from_hf(_source(torch.float32)).items()}
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Interrupted storage read"):
        # Simulate a DCP read which writes only one destination, then aborts
        # without ever calling from_hf on its partially filled destination map.
        destinations["lm_head.weight"].fill_(-42)
        raise RuntimeError("Interrupted storage read")
    fresh_source = _source(torch.float32)
    restored = adapter.from_hf(dict(fresh_source))
    assert restored.keys() == native.keys()
    assert not adapter.view_loaded_native_keys
    assert torch.isnan(native["model.layers.1.mlp.experts.gate_and_up_projs"]).all()
    exported = adapter.to_hf(restored)
    for key, expected in fresh_source.items():
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("replacement", ["clone", "same_storage_new_stride"])
def test_replaced_destination_rebuilds_its_entire_grouped_projection(replacement: str) -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    key = "model.layers.1.mlp.experts.0.gate_proj.weight"
    original = destinations[key]
    if replacement == "clone":
        destinations[key] = original.clone().add_(10)
    else:
        destinations[key] = original.as_strided(original.shape, (original.shape[1], 1))
        assert destinations[key].untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        assert destinations[key].stride() != original.stride()
    expected = {name: value.clone() for name, value in destinations.items()}
    converted = adapter.from_hf(destinations)
    rebuilt = "model.layers.1.mlp.experts.gate_and_up_projs"
    assert rebuilt in converted
    assert rebuilt not in adapter.view_loaded_native_keys
    assert len(adapter.view_loaded_native_keys) == 3
    exported = adapter.to_hf(converted)
    for name, value in exported.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def _metadata(dtype: torch.dtype = torch.bfloat16) -> dict[str, TensorStorageMetadata | BytesStorageMetadata]:
    """Describe the independent tiny source contract without serializing payloads."""
    return {
        key: TensorStorageMetadata(properties=TensorProperties(dtype=dtype), size=tensor.shape, chunks=[])
        for key, tensor in _source(dtype).items()
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_actual_model_state_is_populated_without_rebuilding_grouped_experts(dtype: torch.dtype) -> None:
    config = _config()
    config.dtype = dtype
    model = FlintForCausalLM(config)
    adapter = model.state_dict_adapter
    assert adapter.supports_low_memory_dcp_load
    native = model.state_dict()
    for tensor in native.values():
        tensor.fill_(math.nan)
    source = _source(dtype)
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    native_storages = {tensor.untyped_storage().data_ptr() for tensor in native.values()}
    for name, destination in destinations.items():
        assert destination.untyped_storage().data_ptr() in native_storages
        destination.copy_(source[name])
    assert all(torch.isfinite(tensor).all() for tensor in native.values())
    with patch.object(
        adapter, "_direct_fill_grouped_expert_tensor", side_effect=AssertionError("Unexpected model expert rebuild")
    ):
        remaining = adapter.from_hf(destinations)
    missing, unexpected = model.load_state_dict(remaining, strict=False)
    assert set(missing) == adapter.view_loaded_native_keys
    assert not unexpected
    exported = adapter.to_hf(model.state_dict())
    assert exported.keys() == source.keys()
    for name in source:
        torch.testing.assert_close(exported[name], source[name].to(exported[name].dtype), rtol=0, atol=0)
    assert model.lm_head.weight.dtype == torch.float32
    assert model.model.layers[1].self_attn._fp32_params.softmax_offset.dtype == torch.float32


def test_load_destinations_cannot_silently_omit_a_native_layer_or_head() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    del native["lm_head.weight"]
    with pytest.raises(ValueError, match="Missing native Flint load destinations"):
        adapter.to_hf(native, for_checkpoint_load=True)


def test_shape_only_key_planning_and_partial_single_tensor_conversion() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    assert set(adapter.get_hf_state_dict_keys(native)) == set(_source(torch.float32))
    converted = adapter.convert_single_tensor_to_hf("model.norm.weight", native["model.norm.weight"])
    assert len(converted) == 1 and converted[0][0] == "model.norm.weight"
    assert converted[0][1] is native["model.norm.weight"]


def test_export_preserves_values_without_claiming_load_aliases_or_bf16_casts() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    before = {key: value.clone() for key, value in native.items()}
    exported = adapter.to_hf(native)
    assert all(value.dtype == torch.float32 for value in exported.values())
    assert all(value.is_contiguous() for key, value in exported.items() if ".mlp.experts." in key)
    for key, value in native.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not adapter.view_loaded_native_keys


def test_low_memory_capability_declines_untested_backends_and_multiple_ranks() -> None:
    adapter = _adapter()
    assert adapter.supports_low_memory_dcp_load
    baseline = adapter.backend
    adapter.backend = replace(baseline, experts="te", dispatcher="deepep")
    assert not adapter.supports_low_memory_dcp_load
    adapter.backend = replace(baseline, dispatcher="mok")
    assert not adapter.supports_low_memory_dcp_load
    adapter.backend = replace(baseline, experts="torch_mm")
    assert not adapter.supports_low_memory_dcp_load
    adapter.backend = baseline
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
    ):
        assert not adapter.supports_low_memory_dcp_load


def test_quantized_conversion_is_explicitly_unsupported() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    with pytest.raises(ValueError, match="does not support quantized"):
        adapter.to_hf(native, quantization=True)


def test_all_dense_configuration_does_not_require_shared_moe_configuration() -> None:
    config = _config()
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    config.no_rope_layer_indices = [0]
    config.mlp_only_layers = [0]
    adapter = FlintStateDictAdapter(config, None, _adapter().backend)
    source = {
        key: value
        for key, value in _source(torch.bfloat16).items()
        if not key.startswith(("model.layers.1.", "model.layers.2."))
    }
    consumed = dict(source)
    native = adapter.from_hf(consumed)
    assert not consumed
    assert not adapter.supports_low_memory_dcp_load
    assert all("experts" not in key for key in native)
    exported = adapter.to_hf(native)
    assert exported.keys() == source.keys()
    for key, tensor in source.items():
        torch.testing.assert_close(exported[key], tensor.to(exported[key].dtype), rtol=0, atol=0)
    assert set(adapter.to_hf(native, for_checkpoint_load=True)) == set(source)


def test_moe_configuration_cannot_be_omitted_for_a_moe_layer() -> None:
    with pytest.raises(ValueError, match="MoE layers require"):
        FlintStateDictAdapter(_config(), None, _adapter().backend)


def _prefix_adapter() -> FlintStateDictAdapter:
    """Construct two active layers with the original three-layer source schedule."""
    full = _adapter()
    prefix = FlintConfig(
        **{
            **full.config.to_dict(),
            "num_hidden_layers": 2,
            "schedule_num_hidden_layers": 3,
        }
    )
    return FlintStateDictAdapter(prefix, full.moe_config, full.backend)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_explicit_prefix_accepts_full_source_and_exact_prefix_only(dtype: torch.dtype) -> None:
    adapter = _prefix_adapter()
    source = _source(dtype)
    metadata = _metadata(dtype)
    prefix_metadata = {name: value for name, value in metadata.items() if not name.startswith("model.layers.2.")}
    converted = adapter.from_hf(source)
    assert not source
    exported = adapter.to_hf(converted)
    assert exported.keys() == prefix_metadata.keys()
    assert not any(name.startswith("model.layers.2.") for name in converted)
    expected = _source(dtype)
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, expected[name].to(tensor.dtype), rtol=0, atol=0)


def test_prefix_load_destinations_are_only_active_model_storage() -> None:
    """DCP fills only the constructed prefix views."""
    adapter = _prefix_adapter()
    source = _source(torch.bfloat16)
    native = adapter.from_hf(_source(torch.float32))
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    assert not any(name.startswith("model.layers.2.") for name in destinations)
    for name, destination in destinations.items():
        destination.copy_(source[name])
    result = adapter.from_hf(destinations)
    native.update(result)
    exported = adapter.to_hf(native)
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, source[name].to(tensor.dtype), rtol=0, atol=0)


def test_backend_extra_state_is_excluded_only_for_known_te_modules() -> None:
    adapter = _adapter()
    native = adapter.from_hf(_source(torch.float32))
    sink_extra = "model.layers.1.self_attn._fp32_params._extra_state"
    linear_extra = "model.layers.1.self_attn.q_proj._extra_state"
    for name in [sink_extra, linear_extra]:
        with pytest.raises(ValueError, match="Unexpected Flint"):
            adapter.to_hf({**native, name: torch.empty(0, dtype=torch.uint8)})
    adapter.backend = replace(adapter.backend, attn="te", linear="te")
    exported = adapter.to_hf({**native, sink_extra: torch.empty(0, dtype=torch.uint8), linear_extra: None})
    assert exported.keys() == _source(torch.float32).keys()
    assert adapter.convert_single_tensor_to_hf(sink_extra, torch.empty(0)) == []
    with pytest.raises(ValueError, match="Unexpected Flint"):
        adapter.to_hf({**native, "unknown._extra_state": torch.empty(0)})


@pytest.mark.parametrize("outer_prefix", ["", "base_model.model."])
def test_native_expert_and_linear_lora_roundtrip(outer_prefix: str) -> None:
    adapter = _adapter()
    generator = torch.Generator().manual_seed(900)
    shapes = {
        "model.layers.1.mlp.experts.lora_gate_and_up_A": (4, 6, 2),
        "model.layers.1.mlp.experts.lora_gate_and_up_B": (4, 2, 6),
        "model.layers.1.mlp.experts.lora_down_A": (4, 3, 2),
        "model.layers.1.mlp.experts.lora_down_B": (4, 2, 6),
        "model.layers.0.self_attn.q_proj.lora_A.weight": (2, 6),
        "model.layers.0.self_attn.q_proj.lora_B.weight": (8, 2),
    }
    native = {outer_prefix + name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}
    exported = adapter.to_hf(native)
    assert len(exported) == 26
    destinations = adapter.to_hf(native, for_checkpoint_load=True)
    restored = adapter.from_hf(destinations)
    assert restored.keys() == native.keys()
    for name, tensor in native.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
    assert not adapter.view_loaded_native_keys


@pytest.mark.parametrize("problem", ["missing", "bad_shape", "conflicting_a", "unknown", "bad_rank"])
def test_lora_state_rejects_incomplete_or_conflicting_weights(problem: str) -> None:
    adapter = _adapter()
    native = {
        "model.layers.1.mlp.experts.lora_gate_and_up_A": torch.randn(4, 6, 2),
        "model.layers.1.mlp.experts.lora_gate_and_up_B": torch.randn(4, 2, 6),
        "model.layers.1.mlp.experts.lora_down_A": torch.randn(4, 3, 2),
        "model.layers.1.mlp.experts.lora_down_B": torch.randn(4, 2, 6),
    }
    exported = adapter.to_hf(native)
    name = "model.layers.1.mlp.experts.0.gate_proj.lora_A.weight"
    if problem == "missing":
        del exported[name]
    elif problem == "bad_shape":
        exported[name] = torch.randn(2, 7)
    elif problem == "conflicting_a":
        exported[name].add_(1)
    elif problem == "bad_rank":
        exported["model.layers.1.mlp.experts.0.down_proj.lora_B.weight"] = torch.randn(6, 3)
    else:
        exported["unknown.lora_A.weight"] = torch.randn(2, 6)
    with pytest.raises((ValueError, RuntimeError), match="Flint|stack"):
        adapter.from_hf(exported)


def test_strict_checkpoint_mismatch_reports_path_and_absolute_error() -> None:
    """A FP32 sink mismatch must retain the exact gate and useful diagnostics."""
    path = "model.layers.0.self_attn._fp32_params.softmax_offset"
    expected = {path: torch.tensor([1.0, 2.0], dtype=torch.float32)}
    actual = {path: torch.tensor([1.0, 2.25], dtype=torch.float32)}
    with pytest.raises(AssertionError) as failure:
        _compare(actual, expected, atol=0, rtol=0)
    message = str(failure.value)
    assert path in message
    assert "Greatest absolute difference: 0.25" in message
    assert "Mismatched elements: 1 / 2" in message


def test_equal_checkpoint_state_still_reports_zero_error() -> None:
    """The diagnostic-only change leaves the passing exact comparator unchanged."""
    state = {"model": torch.tensor([1.0, 2.0]), "step": torch.tensor(2)}
    assert _compare(state, state, atol=0, rtol=0) == 0


@pytest.fixture(autouse=True)
def _single_rank_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real CPU archive I/O while replacing only world-size-one collectives."""
    monkeypatch.setattr(composition.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(composition.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(composition.dist, "all_gather_object", lambda output, value: output.__setitem__(0, value))
    monkeypatch.setattr(composition.dist, "broadcast_object_list", lambda payload, src: None)


def _baseline(directory: Path) -> nn.Linear:
    """Write real FP32 CPU weight [2,3] and bias [2] baseline archives."""
    model = nn.Linear(3, 2)
    composition._archive_parameters([model], directory, kind="initial", compare=False)
    return model


def test_matching_parameter_inventory_compares_real_cpu_archives(tmp_path: Path) -> None:
    """The positive path reads both archived tensors and reports a complete set."""
    baseline = _baseline(tmp_path)
    candidate = nn.Linear(3, 2)
    candidate.load_state_dict(baseline.state_dict())
    result = composition._archive_parameters([candidate], tmp_path, kind="initial", compare=True)
    assert result["tensor_count"] == result["parameter_inventory"]["parameter_count"] == 2
    assert set(result["tensors"]) == {"weight", "bias"}
    assert result["error_ss"] == 0
    assert json.loads((tmp_path / "parameter_inventory.json").read_text())["parameter_fqns"] == ["bias", "weight"]


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_missing_or_extra_fqn_fails_before_tensor_payload_read(tmp_path: Path, monkeypatch, change: str) -> None:
    """An incomplete candidate cannot pass by comparing only its remaining weights."""
    baseline = _baseline(tmp_path)
    candidate = nn.Linear(3, 2)
    candidate.load_state_dict(baseline.state_dict())
    if change == "missing":
        candidate.register_parameter("bias", None)
    else:
        candidate.register_parameter("extra", nn.Parameter(torch.zeros(1)))

    def forbidden_payload_read(*args, **kwargs):
        raise AssertionError("FQN mismatch must fail before reading any baseline tensor")

    monkeypatch.setattr(composition, "safe_open", forbidden_payload_read)
    expected = r"missing=\['bias'\]" if change == "missing" else r"extra=\['extra'\]"
    with pytest.raises(AssertionError, match=expected):
        composition._archive_parameters([candidate], tmp_path, kind="initial", compare=True)
