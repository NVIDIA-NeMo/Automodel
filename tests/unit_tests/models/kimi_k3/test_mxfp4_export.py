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
import math
import re
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PretrainedConfig

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import dequantize_mxfp4, quantize_mxfp4
from tests.unit_tests.models.kimi_k3.test_state_dict_adapter import _tiny_adapter


def _adapter():
    adapter = _tiny_adapter()
    adapter.moe_config.moe_inter_dim = 32
    return adapter


def _weights():
    generator = torch.Generator().manual_seed(4078)
    return {
        "model.layers.5.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64, generator=generator),
        "model.layers.5.mlp.experts.down_projs": torch.randn(4, 32, 64, generator=generator),
        "model.layers.5.mlp.shared_experts.gate_proj.weight": torch.randn(32, 64, generator=generator),
        "model.layers.5.mlp.routed_expert_down_proj.weight": torch.randn(32, 64, generator=generator),
        "model.layers.5.mlp.routed_expert_up_proj.weight": torch.randn(64, 32, generator=generator),
        "model.layers.5.mlp.gate.weight": torch.randn(4, 64, generator=generator),
        "model.layers.5.self_attn.q_proj.weight": torch.randn(32, 64, generator=generator),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(32, 64, generator=generator),
        "lm_head.weight": torch.randn(32, 64, generator=generator),
        "vision_tower.proj.weight": torch.randn(32, 64, generator=generator),
    }


def _reference(weight):
    """Scalar, double-precision oracle independent of the tensor implementation.

    Args:
        weight: Local floating-point CPU tensor of shape [out, in], in divisible by 32.

    Returns:
        CPU uint8 packed bytes [out, in / 2] and E8M0 scales [out, in / 32].
    """
    levels = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    packed, scales = [], []
    for row in weight.tolist():
        codes, row_scales = [], []
        for start in range(0, len(row), 32):
            block = row[start : start + 32]
            maximum = max(map(abs, block))
            exponent = max(-127, min(127, math.ceil(math.log2(maximum / 6)))) if maximum else 0
            row_scales.append(exponent + 127)
            for value in block:
                scaled = abs(math.ldexp(value, -exponent))
                code = min(range(8), key=lambda i: (abs(levels[i] - scaled), i % 2))
                codes.append(code | (8 if math.copysign(1, value) < 0 else 0))
        packed.append([lo | (hi << 4) for lo, hi in zip(codes[::2], codes[1::2])])
        scales.append(row_scales)
    return torch.tensor(packed, dtype=torch.uint8), torch.tensor(scales, dtype=torch.uint8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantize_matches_independent_oracle(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    weight = torch.randn(64, 129, generator=torch.Generator().manual_seed(4078)).to(dtype).t()
    assert not weight.is_contiguous()  # also crosses the 128-row scratch boundary
    expected = _reference(weight)
    weight = weight.to(device)
    before = weight.clone()
    packed, scales = quantize_mxfp4(weight)
    torch.testing.assert_close(packed.cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_mxfp4_golden_bytes_and_round_to_even():
    values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    weight = torch.tensor([values * 2])
    packed, scales = quantize_mxfp4(weight)
    assert packed.tolist() == [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2]
    assert scales.tolist() == [[127]]
    torch.testing.assert_close(dequantize_mxfp4(packed, scales, torch.float32), weight, rtol=0, atol=0)
    # Every E2M1 midpoint, plus 6 to fix the group scale to 1.
    midpoints = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]
    packed, scales = quantize_mxfp4(torch.tensor([midpoints * 4]))
    expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6] * 4], dtype=torch.float32)
    torch.testing.assert_close(dequantize_mxfp4(packed, scales, torch.float32), expected, rtol=0, atol=0)


def test_mxfp4_scale_boundaries_zeros_and_small_values():
    maxima = torch.tensor([0, 3, 6, 12, 1e-38, 1e-30, 1e30], dtype=torch.float32)
    six = torch.tensor(6.0)
    maxima = torch.cat([maxima, torch.nextafter(six, torch.tensor(float("inf"))).view(1)])
    weight = maxima[:, None].expand(-1, 32)
    packed, scales = quantize_mxfp4(weight)
    expected = _reference(weight)
    torch.testing.assert_close(packed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(scales, expected[1], rtol=0, atol=0)
    assert scales[:4, 0].tolist() == [127, 126, 127, 128]
    assert scales[-1, 0].item() == 128


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(32),
        torch.zeros(2, 31),
        torch.zeros(2, 0),
        torch.zeros(2, 32, dtype=torch.int32),
        torch.empty(2, 32, device="meta"),
        torch.full((2, 32), float("nan")),
        torch.full((2, 32), float("inf")),
    ],
)
def test_mxfp4_rejects_invalid_weights(weight):
    """Reject invalid ranks, input widths, dtypes, and values.

    Args:
        weight: Invalid tensor of shape [32], [2, 0], [2, 31], or [2, 32].
    """
    with pytest.raises(ValueError):
        quantize_mxfp4(weight)


def test_adapter_quantizes_only_routed_experts_and_roundtrips():
    adapter = _adapter()
    native = _weights()
    original = {key: value.clone() for key, value in native.items()}
    plain = adapter.to_hf(native)
    packed = adapter.to_hf(native, quantization=True)
    decoded_hf = {}
    for key, value in plain.items():
        if re.search(r"\.experts\.\d+\.w[123]\.weight$", key):
            expected_packed, expected_scales = _reference(value)
            torch.testing.assert_close(packed[key + "_packed"], expected_packed, rtol=0, atol=0)
            torch.testing.assert_close(packed[key + "_scale"], expected_scales, rtol=0, atol=0)
            decoded_hf[key] = dequantize_mxfp4(expected_packed, expected_scales, torch.float32)
        else:
            assert packed[key] is value
            decoded_hf[key] = value
    assert not hasattr(adapter, "_mxfp4_load_views")
    expected = _adapter().from_hf(decoded_hf)
    restored = adapter.from_hf(packed)
    assert restored.keys() == native.keys()
    for key in native:
        torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(native[key], original[key], rtol=0, atol=0)


def test_quantized_checkpoint_load_still_uses_empty_buffers_and_model_views(monkeypatch):
    adapter = _adapter()
    native = _weights()
    exported = adapter.to_hf(native, quantization=True)
    monkeypatch.setattr(
        "nemo_automodel.components.models.kimi_k3.state_dict_adapter.quantize_mxfp4",
        lambda _: pytest.fail("Loading must not quantize uninitialized model weights"),
    )
    destinations = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    assert len(adapter._mxfp4_load_views) == 12
    for key, value in destinations.items():
        assert value.shape == exported[key].shape
        assert value.dtype == exported[key].dtype
        value.copy_(exported[key])
    restored = adapter.from_hf(destinations)
    assert restored.keys() == native.keys()
    assert not hasattr(adapter, "_mxfp4_load_views")


@pytest.mark.parametrize("nested", [False, True])
def test_quantized_metadata_targets_and_no_mutation(nested):
    adapter = _adapter()
    text = {"hidden_size": 64, "quantization_config": {"stale": True}}
    config = {"text_config": text, "auto_map": {"AutoConfig": "custom.Config"}} if nested else text
    original = deepcopy(config)
    exported = adapter.adapt_hf_config_for_save(config, quantization=True)
    scheme = exported.get("text_config", exported)["quantization_config"]
    assert scheme["format"] == "mxfp4-pack-quantized"
    target = scheme["config_groups"]["group_0"]["targets"][0].removeprefix("re:")
    for name in _adapter().to_hf(_weights()):
        matched = re.fullmatch(target, name.removesuffix(".weight")) is not None
        assert matched == (".experts." in name)
    plain = adapter.adapt_hf_config_for_save(config)
    assert "quantization_config" not in plain
    assert "quantization_config" not in plain.get("text_config", {})
    assert config == original


def _model():
    model = torch.nn.Module()
    for key, tensor in _weights().items():
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parent.register_parameter(parts[-1], torch.nn.Parameter(tensor))
    model.config = PretrainedConfig(
        text_config={"hidden_size": 64, "quantization_config": {"stale": True}},
        quantization_config={"stale": True},
    )
    model.state_dict_adapter = _adapter()
    model._pre_shard_hf_state_dict_keys = list(model.state_dict_adapter.to_hf(model.state_dict()))
    return model


@pytest.mark.parametrize("quantization", [False, True])
@pytest.mark.parametrize("v4_compatible", [False, True])
def test_real_safetensors_export_weights_and_metadata(tmp_path, quantization, v4_compatible):
    """Exercise actual ModelState, DCP save, consolidation, and reload, without IO mocks."""
    model = _model()
    source = tmp_path / "source"
    source.mkdir()
    model.config.name_or_path = str(source)
    (source / "config.json").write_text(model.config.to_json_string(use_diff=False))
    # Real training models retain unquantized pre-shard HF keys, even when the
    # base checkpoint has packed names. The export index must use saved keys.
    source_weights = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True)
    save_file({key: value.contiguous() for key, value in source_weights.items()}, str(source / "model.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source_weights, "model.safetensors")})
    )
    checkpointer = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        model_repo_id=str(source),
        model_cache_dir=str(source),
        model_save_format="safetensors",
        save_consolidated=True,
        v4_compatible=v4_compatible,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    expected = model.state_dict_adapter.to_hf(model.state_dict(), quantization=quantization)
    try:
        checkpointer.save_model(model, str(tmp_path / "export"), quantization=quantization)
    finally:
        checkpointer.close()
    output = tmp_path / "export/model/consolidated"
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {}
    assert set(index) == set(expected)
    for filename in set(index.values()):
        actual.update(load_file(str(output / filename)))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for name in ("config.json", "config.v5.json") if v4_compatible else ("config.json",):
        config = json.loads((output / name).read_text())
        assert ("quantization_config" in config["text_config"]) is quantization
    reloaded = _adapter().from_hf(actual)
    assert reloaded.keys() == model.state_dict().keys()
    # Export does not overwrite trainable weights or the in-memory config.
    assert model.config.text_config["quantization_config"] == {"stale": True}
    assert model.config.quantization_config == {"stale": True}
    for key, value in _weights().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    destination = _model()
    with torch.no_grad():
        for parameter in destination.parameters():
            parameter.zero_()
    loader = CheckpointingConfig(checkpoint_dir=str(tmp_path), dequantize_base_checkpoint=quantization).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    try:
        loader.load_model(destination, str(output), is_init_step=True)
    finally:
        loader.close()
    for key, value in destination.state_dict().items():
        torch.testing.assert_close(value, reloaded[key], rtol=0, atol=0)


@pytest.mark.parametrize("unsupported", ["torch_save", "peft", "missing_adapter", "load_only_adapter"])
def test_checkpointer_rejects_unsupported_quantized_export(tmp_path, unsupported):
    from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

    model = _model()
    if unsupported == "missing_adapter":
        del model.state_dict_adapter
    elif unsupported == "load_only_adapter":
        # An adapter can allocate packed tensors for load without implementing export.
        model.state_dict_adapter.adapt_hf_config_for_save = (
            lambda config, **kwargs: StateDictAdapter.adapt_hf_config_for_save(
                model.state_dict_adapter, config, **kwargs
            )
        )
    checkpointer = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        model_save_format="torch_save" if unsupported == "torch_save" else "safetensors",
        is_peft=unsupported == "peft",
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    try:
        with pytest.raises((ValueError, NotImplementedError)):
            checkpointer.save_model(model, str(tmp_path / "export"), quantization=True)
        assert not (tmp_path / "export").exists()
    finally:
        checkpointer.close()


def _distributed_export_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        full = {key: value for key, value in _weights().items() if ".experts." in key}
        local = {key: distribute_tensor(value, mesh, [Shard(0)]) for key, value in full.items()}
        exported = _adapter().to_hf(local, quantization=True, device_mesh=mesh)
        reference = _adapter().to_hf(full, quantization=True)
        local_ids = {int(re.search(r"\.experts\.(\d+)\.", key).group(1)) for key in exported}
        assert local_ids == {rank * 2, rank * 2 + 1}
        for key, value in exported.items():
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
        all_keys = [None, None]
        dist.all_gather_object(all_keys, set(exported))
        assert all_keys[0].isdisjoint(all_keys[1])
        assert all_keys[0] | all_keys[1] == set(reference)
        model = _model()
        for key, value in local.items():
            parent_path, name = key.rsplit(".", 1)
            model.get_submodule(parent_path).register_parameter(name, torch.nn.Parameter(value))
        checkpointer = CheckpointingConfig(checkpoint_dir=output_dir, save_consolidated=True).build(
            dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=mesh
        )
        try:
            checkpointer.save_model(model, output_dir, quantization=True)
        finally:
            checkpointer.close()
        from pathlib import Path

        output = Path(output_dir) / "model/consolidated"
        index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
        expected = _adapter().to_hf(_weights(), quantization=True)
        assert set(index) == set(expected)
        actual = {}
        for filename in set(index.values()):
            actual.update(load_file(str(output / filename)))
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        within_expert = distribute_tensor(full["model.layers.5.mlp.experts.down_projs"], mesh, [Shard(1)])
        with pytest.raises(ValueError, match="within-expert"):
            _adapter().to_hf(
                {"model.layers.5.mlp.experts.down_projs": within_expert}, quantization=True, device_mesh=mesh
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(45, reason="Two real Gloo ranks import torch and test expert-parallel DCP export.")
def test_expert_parallel_export_has_complete_disjoint_expert_shards(tmp_path):
    mp.spawn(
        _distributed_export_worker, args=(str(tmp_path / "gloo_init"), str(tmp_path / "export")), nprocs=2, join=True
    )
