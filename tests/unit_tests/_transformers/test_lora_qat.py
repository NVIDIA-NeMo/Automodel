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

"""CPU integration tests using actual PEFT patching and packed numerical storage."""

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import (
    LinearLoRA,
    PeftConfig,
    apply_lora_to_linear_modules,
    patch_linear_module,
    patch_moe_module,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig


def _architecture(*, dtype=torch.float32, device="cpu", bias=True):
    return nn.Sequential(
        nn.Linear(32, 32, bias=bias, dtype=dtype, device=device),
        nn.Tanh(),
        nn.Linear(32, 32, bias=bias, dtype=dtype, device=device),
    )


def _patched(*, dtype=torch.float32, device="cpu", bias=True):
    model = _architecture(dtype=dtype, device=device, bias=bias)
    count = apply_lora_to_linear_modules(
        model, PeftConfig(match_all_linear=True, dim=4, alpha=8, use_memory_efficient_lora=False)
    )
    assert count == 2
    assert isinstance(model[0], LinearLoRA)
    return model


def _controller(*, scale_format="e8m0"):
    return QAT(QATConfig(rules=[
        QATRule(["0"], WeightQuantizationConfig("fp8", (32, 32), scale_format=scale_format)),
        QATRule(["2"], WeightQuantizationConfig("mxfp4", (1, 32))),
    ]))


def _one_rule(*patterns):
    return QAT(QATConfig(target_modules=patterns, weight=WeightQuantizationConfig("fp8", (32, 32))))


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def _write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _refresh_hash(directory, filename):
    manifest = _manifest(directory)
    key = "model_sha256" if filename == "model.safetensors" else "quantized_sha256"
    manifest[key] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    _write_manifest(directory, manifest)


def test_config_owned_build_normalizes_lists_and_preserves_frozen_settings():
    rule = QATRule(["0"], WeightQuantizationConfig("fp8", (32, 32)))
    config = QATConfig(rules=[rule])
    assert rule.target_modules == ("0",)
    assert config.rules == (rule,)
    assert config.build() is not config.build()
    with pytest.raises(FrozenInstanceError):
        rule.target_modules = ("2",)


@pytest.mark.parametrize("factory", [
    lambda: QATRule("0", WeightQuantizationConfig()),
    lambda: QATRule([], WeightQuantizationConfig()),
    lambda: QATRule([1], WeightQuantizationConfig()),
    lambda: QATRule(["0"], {"format": "fp8"}),
    lambda: QATConfig(rules=[{"target_modules": ["0"]}]),
    lambda: QATConfig(rules="0"),
    lambda: QAT({"rules": []}),
])
def test_configs_reject_uninstantiated_yaml_and_invalid_types(factory):
    with pytest.raises(TypeError):
        factory()


def test_prepare_meta_is_metadata_only_and_preserves_parameter_identity():
    model = _patched(device="meta")
    before = dict(model.named_parameters())
    keys = set(model.state_dict())
    controller = _controller()
    with patch.object(LinearLoRA, "materialize_effective_weight", side_effect=AssertionError("allocated weight")):
        with patch.object(WeightFakeQuantizer, "quantize", side_effect=AssertionError("read tensor contents")):
            assert controller.prepare(model) is model
    assert set(model.state_dict()) == keys
    assert all(parameter is dict(model.named_parameters())[name] for name, parameter in before.items())
    assert all(parameter.is_meta for parameter in model.parameters())
    assert model[0].weight_fake_quantizer.config.block_size == (32, 32)
    assert model[2].weight_fake_quantizer.config.block_size == (1, 32)
    assert not list(model[0].weight_fake_quantizer.buffers())


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_repeated_prepare_preserves_parameters_state_and_training_modes(device):
    model = _patched(device=device)
    model[2].eval()
    parameters = dict(model.named_parameters())
    keys = set(model.state_dict())
    controller = _controller()
    assert controller.prepare(model) is model
    assert controller.prepare(model) is model
    assert all(value is dict(model.named_parameters())[name] for name, value in parameters.items())
    assert set(model.state_dict()) == keys
    assert model[0].weight_fake_quantizer.training
    assert not model[2].weight_fake_quantizer.training
    assert not list(model[0].weight_fake_quantizer.buffers())
    assert not list(model[2].weight_fake_quantizer.buffers())
    if device == "cpu":
        inputs = torch.randn(2, 32)
        before = model(inputs).detach()
        controller.prepare(model)
        torch.testing.assert_close(model(inputs), before, rtol=0, atol=0)


# Exercise a genuinely unloaded checkpoint module without disturbing other tests'
# imports. Cold Torch/Transformers imports need more than the five-second default.
@pytest.mark.timeout(60)
def test_prepare_does_not_import_checkpoint_or_require_safetensors(tmp_path):
    code = """
import importlib
import sys
from unittest.mock import patch

import torch
from torch import nn
from nemo_automodel.components._peft.lora import patch_linear_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.quantization.qat import QATConfig
from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig

checkpoint_module = "nemo_automodel._transformers.qat_checkpoint"
assert checkpoint_module not in sys.modules
original_import = importlib.import_module
attempts = []

def without_safetensors(name, package=None):
    if name == "safetensors" or name.startswith("safetensors."):
        attempts.append(name)
        raise ImportError("safetensors deliberately unavailable")
    return original_import(name, package)

with patch("importlib.import_module", side_effect=without_safetensors):
    from nemo_automodel._transformers.qat import QAT

    model = nn.Linear(32, 32)
    patch_linear_module(model, use_triton=False)
    controller = QAT(QATConfig(target_modules=("",), weight=WeightQuantizationConfig("fp8", (32, 32))))
    assert controller.prepare(model) is model
    assert not attempts
    assert checkpoint_module not in sys.modules
    # Preserve dependency-error precedence over eval/non-LoRA validation.
    for operation, message in (
        (controller.export, "LoRA QAT export requires safetensors"),
        (QAT.load_quantized_checkpoint, "LoRA QAT loading requires safetensors"),
    ):
        try:
            operation(model, sys.argv[1])
        except ImportError as error:
            assert str(error) == message
        else:
            raise AssertionError("missing safetensors was not rejected")
    assert attempts == ["safetensors.torch"]
    assert controller.prepare(model) is model
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("patterns, error, message", [
    (("missing.*",), ValueError, "no modules match"),
    (("0.lora_A",), TypeError, "not a supported LoRA"),
    (("0.lora_B",), TypeError, "not a supported LoRA"),
    (("1",), TypeError, "not a supported LoRA"),
    (("0.*",), TypeError, "not a supported LoRA"),
])
def test_bad_targets_do_not_attach_any_quantizers(patterns, error, message):
    model = _patched()
    with pytest.raises(error, match=message):
        _one_rule(*patterns).prepare(model)
    assert model[0].weight_fake_quantizer is None
    assert model[2].weight_fake_quantizer is None


def test_full_name_case_sensitive_globs_and_overlap_atomicity():
    model = nn.ModuleDict({"Upper": _patched()})
    with pytest.raises(ValueError, match="no modules match"):
        _one_rule("upper.[02]").prepare(model)
    _one_rule("Upper.[02]").prepare(model)
    assert isinstance(model["Upper"][0].weight_fake_quantizer, WeightFakeQuantizer)
    model = _patched()
    config = QATConfig(rules=(
        QATRule(("[02]",), WeightQuantizationConfig("fp8", (32, 32))),
        QATRule(("2",), WeightQuantizationConfig("mxfp4", (1, 32))),
    ))
    with pytest.raises(ValueError, match="overlapping"):
        QAT(config).prepare(model)
    assert model[0].weight_fake_quantizer is model[2].weight_fake_quantizer is None


@pytest.mark.parametrize("invalid", ["dropout", "dora", "frozen", "dtype", "shape", "super_fwd", "quantized"])
def test_late_validation_failure_is_atomic(invalid):
    model = _patched()
    if invalid == "dropout":
        model[2].dropout_p = 0.1
    elif invalid == "dora":
        model[2].use_dora = True
    elif invalid == "frozen":
        model[2].weight.requires_grad_(True)
    elif invalid == "dtype":
        model[2].lora_A.to(dtype=torch.float64)
    elif invalid == "shape":
        model[2].lora_B.weight = nn.Parameter(torch.empty(31, 4))
    elif invalid == "super_fwd":
        model[2].super_fwd = nn.Identity()
    else:
        model[2].weight = nn.Parameter(torch.zeros(32, 32, dtype=torch.uint8), requires_grad=False)
    with pytest.raises((ValueError, TypeError)):
        _controller().prepare(model)
    assert model[0].weight_fake_quantizer is model[2].weight_fake_quantizer is None


def test_prepare_rejects_nondivisible_shape_without_mutation():
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 33))
    for layer in model:
        patch_linear_module(layer, use_triton=False)
    with pytest.raises(ValueError, match="not divisible"):
        _one_rule("[01]").prepare(model)
    assert model[0].weight_fake_quantizer is model[1].weight_fake_quantizer is None


def test_prepare_rejects_tied_embedding_head_before_attaching_any_quantizers():
    model = nn.ModuleDict(
        {
            "projection": nn.Linear(32, 32),
            "embedding": nn.Embedding(32, 32),
            "lm_head": nn.Linear(32, 32, bias=False),
        }
    )
    model["lm_head"].weight = model["embedding"].weight
    for name in ("projection", "lm_head"):
        patch_linear_module(model[name], use_triton=False)
    base = model["embedding"].weight
    before = base.detach().clone()
    with patch.object(LinearLoRA, "materialize_effective_weight", side_effect=AssertionError("merged weight")):
        with pytest.raises(ValueError, match="tied base parameter"):
            _one_rule("projection", "lm_head").prepare(model)
    assert model["projection"].weight_fake_quantizer is model["lm_head"].weight_fake_quantizer is None
    assert model["lm_head"].weight is model["embedding"].weight is base
    assert torch.equal(base, before)
    assert not base.requires_grad


def test_prepare_rejects_duplicate_lora_module_paths():
    model = _patched()
    model.add_module("alias", model[2])
    with pytest.raises(ValueError, match="tied base parameter"):
        _controller().prepare(model)
    assert model[0].weight_fake_quantizer is model[2].weight_fake_quantizer is None


@pytest.mark.parametrize("selected", [False, True])
def test_export_rejects_tied_embedding_head_even_when_untargeted(tmp_path, selected):
    model = nn.ModuleDict({
        "embedding": nn.Embedding(32, 32),
        "lm_head": nn.Linear(32, 32, bias=False),
        "projection": nn.Linear(32, 32),
    })
    for name in ("lm_head", "projection"):
        patch_linear_module(model[name], use_triton=False)
    controller = _one_rule("lm_head" if selected else "projection")
    controller.prepare(model)
    # Also revalidate ties introduced after preparation, before any export merge.
    model["lm_head"].weight = model["embedding"].weight.requires_grad_(False)
    with torch.no_grad():
        model["lm_head"].lora_B.weight.normal_()
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with patch.object(LinearLoRA, "materialize_effective_weight", side_effect=AssertionError("merged weight")):
        with pytest.raises(ValueError, match="tied base parameter"):
            controller.export(model, tmp_path)
    assert model["lm_head"].weight is model["embedding"].weight
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["activation", "base", "a", "b"])
def test_dense_qat_rejects_mixed_dtype_before_effective_matmul(target):
    model = _patched()
    _controller().prepare(model)
    layer = model[0]
    inputs = torch.randn(2, 32)
    if target == "activation":
        inputs = inputs.bfloat16()
    elif target == "base":
        layer.weight = nn.Parameter(layer.weight.bfloat16(), requires_grad=False)
    elif target == "a":
        layer.lora_A.bfloat16()
    else:
        layer.lora_B.bfloat16()
    with patch.object(LinearLoRA, "materialize_effective_weight", side_effect=AssertionError("merged weight")):
        with pytest.raises(ValueError, match="activation dtype"):
            layer(inputs)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_dense_qat_rejects_autocast_before_effective_matmul(device_type):
    model = _patched(dtype=torch.bfloat16)
    _controller().prepare(model)
    inputs = torch.randn(2, 32, dtype=torch.bfloat16)
    # Enable the real CUDA autocast flag on CPU CI; no CUDA execution is claimed.
    with patch("torch.cuda.is_available", return_value=True), torch.autocast(device_type):
        assert torch.is_autocast_enabled(device_type)
        with patch.object(LinearLoRA, "materialize_effective_weight", side_effect=AssertionError("merged weight")):
            with pytest.raises(ValueError, match="autocast"):
                model[0](inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_format", ["e8m0", "float32"])
def test_optimizer_export_reload_exact_eval_and_byte_roundtrip(tmp_path, dtype, scale_format):
    torch.manual_seed(91)
    model = _patched(dtype=dtype)
    controller = _controller(scale_format=scale_format)
    controller.prepare(model)
    with torch.no_grad():
        for layer in (model[0], model[2]):
            layer.lora_B.weight.normal_(std=0.04)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    before_step = {name: p.detach().clone() for name, p in model.named_parameters()}
    inputs = torch.randn(3, 32, dtype=dtype)
    model(inputs).float().square().mean().backward()
    for layer in (model[0], model[2]):
        assert layer.weight.grad is None
        assert layer.lora_A.weight.grad.abs().sum() > 0
        assert layer.lora_B.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(model[0].lora_B.weight, before_step["0.lora_B.weight"])
    assert torch.equal(model[0].weight, before_step["0.weight"])
    model.eval()
    reference = model(inputs).detach()
    snapshot = {key: value.clone() for key, value in model.state_dict().items()}
    identities = {key: id(value) for key, value in model.named_parameters()}
    controller.export(model, tmp_path)
    manifest = _manifest(tmp_path)
    assert set(manifest) == {
        "schema_version", "format", "model_file", "quantized_file", "model_sha256", "quantized_sha256", "weights"
    }
    assert manifest["format"] == "nemo_automodel.lora_qat"
    assert manifest["model_file"] == "model.safetensors"
    assert manifest["quantized_file"] == "quantized.safetensors"
    for entry in manifest["weights"]:
        assert set(entry) == {"weight_key", "dtype", "shape", "layout", "config", "payload", "scales"}
        assert set(entry["config"]) == {"format", "block_size", "scale_format"}
        assert entry["dtype"] == str(dtype).removeprefix("torch.")
        assert entry["layout"] == "out_in"
        assert entry["shape"] == [32, 32]
    assert type(manifest["schema_version"]) is int
    assert manifest["schema_version"] == 1
    assert [entry["config"]["format"] for entry in manifest["weights"]] == ["fp8", "mxfp4"]
    assert [entry["config"]["scale_format"] for entry in manifest["weights"]] == [scale_format, "e8m0"]
    ordinary = load_file(str(tmp_path / "model.safetensors"))
    assert set(ordinary) == {"0.bias", "2.bias"}
    packed = load_file(str(tmp_path / "quantized.safetensors"))
    for key, value in packed.items():
        expected_dtype = torch.float32 if key == "0.weight.scales" and scale_format == "float32" else torch.uint8
        assert value.dtype == expected_dtype
    assert packed["0.weight.payload"].shape == (32, 32)
    assert packed["2.weight.payload"].shape == (32, 16)
    assert packed["0.weight.scales"].shape == (1, 1)
    assert packed["2.weight.scales"].shape == (32, 1)
    for index in (0, 2):
        encoded = model[index].weight_fake_quantizer.quantize(model[index].materialize_effective_weight())
        assert torch.equal(packed[f"{index}.weight.payload"], encoded.payload)
        assert torch.equal(packed[f"{index}.weight.scales"], encoded.scales)
    fresh = _architecture(dtype=dtype).eval()
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    actual = fresh(inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert {key: id(value) for key, value in model.named_parameters()} == identities
    assert set(snapshot) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key])
    assert all(not child.training for child in model.modules())
    # Re-export is byte stable, not just numerically close.
    second = tmp_path / "second"
    controller.export(model, second)
    for filename in ("model.safetensors", "quantized.safetensors", "manifest.json"):
        assert (tmp_path / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.parametrize("scale_format", ["e8m0", "float32"])
def test_export_snapshots_packed_storage_before_next_projection(tmp_path, scale_format):
    from nemo_automodel._transformers.qat_checkpoint import QATCheckpoint

    model = _patched().eval()
    controller = _controller(scale_format=scale_format)
    controller.prepare(model)
    model_snapshot = {key: value.clone() for key, value in model.state_dict().items()}
    sources = []
    snapshots = []
    quantize = WeightFakeQuantizer.quantize

    def record_quantize(self, weight):
        """Instrument real encoding and reuse the preceding projection's storage.

        Args:
            weight: CPU float32 tensor of shape [out, in]. Never mutated.

        Returns:
            QuantizedWeight with the payload/scales layouts documented on
            WeightFakeQuantizer.quantize, retaining its real CPU storage.
        """
        assert not torch.is_grad_enabled()
        if sources:
            previous = sources[-1]
            payload, scales = snapshots[-1]
            assert torch.equal(previous.payload, payload)
            assert torch.equal(previous.scales, scales)
            # A snapshot taken only at final save would observe these mutations.
            previous.payload.bitwise_xor_(1)
            previous.scales.add_(1)
        encoded = quantize(self, weight)
        sources.append(encoded)
        snapshots.append((encoded.payload.clone(), encoded.scales.clone()))
        return encoded

    with patch.object(WeightFakeQuantizer, "quantize", autospec=True, side_effect=record_quantize):
        with patch.object(QATCheckpoint, "save", wraps=QATCheckpoint.save) as save:
            controller.export(model, tmp_path)
    save.assert_called_once()
    projections = save.call_args.args[1]
    assert [projection.weight_key for projection in projections] == ["0.weight", "2.weight"]
    assert len(sources) == len(snapshots) == len(projections) == 2
    packed = load_file(str(tmp_path / "quantized.safetensors"))
    for projection, source, snapshot in zip(projections, sources, snapshots):
        encoded = projection.encoded
        assert encoded.config == source.config
        assert encoded.shape == source.shape
        assert projection.dtype == torch.float32
        assert projection.layout == "out_in"
        for field, expected in zip(("payload", "scales"), snapshot):
            value = getattr(encoded, field)
            assert value.device.type == "cpu"
            assert not value.requires_grad
            assert value.grad_fn is None
            assert value.data_ptr() != getattr(source, field).data_ptr()
            assert torch.equal(value, expected)
            assert torch.equal(packed[f"{projection.weight_key}.{field}"], expected)
    assert torch.equal(sources[-1].payload, snapshots[-1][0])
    assert torch.equal(sources[-1].scales, snapshots[-1][1])
    assert all(torch.equal(value, model_snapshot[key]) for key, value in model.state_dict().items())


def test_subset_merges_untargeted_lora_and_preserves_unrelated_state(tmp_path):
    model = _patched()
    model[2].dropout_p = 0.2
    model.register_buffer("ordinary_lora_A_value", torch.arange(7))
    with torch.no_grad():
        model[2].lora_B.weight.normal_(std=0.1)
    controller = _one_rule("0")
    controller.prepare(model)
    model.eval()
    inputs = torch.randn(3, 32)
    merged = model[2].materialize_effective_weight().detach()
    reference = F.linear(model[1](model[0](inputs)), merged, model[2].bias)
    controller.export(model, tmp_path)
    ordinary = load_file(str(tmp_path / "model.safetensors"))
    assert torch.equal(ordinary["2.weight"], merged)
    assert torch.equal(ordinary["ordinary_lora_A_value"], torch.arange(7))
    assert "2.lora_A.weight" not in ordinary
    fresh = _architecture().eval()
    fresh.register_buffer("ordinary_lora_A_value", torch.zeros(7, dtype=torch.int64))
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    torch.testing.assert_close(fresh(inputs), reference, rtol=0, atol=0)


# Cold Torch/Transformers imports exceed the suite's five-second default;
# keep both the test and child process bounded, as in the resume tests.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_float32_scale_checkpoint_loads_in_fresh_process(tmp_path, dtype):
    torch.manual_seed(53)
    model = nn.Linear(128, 256, bias=False, dtype=dtype)
    patch_linear_module(model, dim=4, alpha=8, use_triton=False)
    controller = QAT(QATConfig(target_modules=("",), weight=WeightQuantizationConfig(scale_format="float32")))
    controller.prepare(model)
    with torch.no_grad():
        model.lora_B.weight.normal_(std=0.04)
    model.eval()
    inputs = (torch.arange(256, dtype=torch.float32).reshape(2, 128) / 100).to(dtype)
    expected = model(inputs).detach()
    controller.export(model, tmp_path)
    code = """
import sys
import torch
from torch import nn
from safetensors.torch import save_file
from nemo_automodel._transformers.qat import QAT
dtype = getattr(torch, sys.argv[2])
model = nn.Linear(128, 256, bias=False, dtype=dtype).eval()
QAT.load_quantized_checkpoint(model, sys.argv[1])
inputs = (torch.arange(256, dtype=torch.float32).reshape(2, 128) / 100).to(dtype)
save_file({"output": model(inputs).detach()}, sys.argv[3])
"""
    output_path = tmp_path / "fresh.safetensors"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(dtype).removeprefix("torch."), str(output_path)],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = load_file(str(output_path))["output"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_empty_ordinary_file_and_root_lora_model(tmp_path):
    model = nn.Linear(32, 32, bias=False)
    patch_linear_module(model, use_triton=False)
    controller = _one_rule("")
    controller.prepare(model)
    model.eval()
    controller.export(model, tmp_path)
    assert load_file(str(tmp_path / "model.safetensors")) == {}
    fresh = nn.Linear(32, 32, bias=False).eval()
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    inputs = torch.randn(2, 32)
    torch.testing.assert_close(fresh(inputs), model(inputs), rtol=0, atol=0)


def test_independent_export_controller_merges_untargeted_weights(tmp_path):
    model = _patched()
    _one_rule("0").prepare(model)
    model.eval()
    # Export must depend on the typed plan and model, not controller history.
    _one_rule("0").export(model, tmp_path)
    assert set(load_file(str(tmp_path / "quantized.safetensors"))) == {"0.weight.payload", "0.weight.scales"}
    assert [entry["weight_key"] for entry in _manifest(tmp_path)["weights"]] == ["0.weight"]
    fresh = _architecture().eval()
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert torch.equal(fresh[2].weight, model[2].materialize_effective_weight())
    inputs = torch.randn(2, 32)
    torch.testing.assert_close(fresh(inputs), model(inputs), rtol=1e-6, atol=1e-7)


def test_tied_ordinary_parameters_are_cloned_for_safetensors(tmp_path):
    model = _patched()
    model.register_parameter("alias_one", nn.Parameter(torch.ones(5), requires_grad=False))
    model.register_parameter("alias_two", model.alias_one)
    controller = _controller()
    controller.prepare(model)
    model.eval()
    controller.export(model, tmp_path)
    fresh = _architecture().eval()
    fresh.register_parameter("alias_one", nn.Parameter(torch.zeros(5), requires_grad=False))
    fresh.register_parameter("alias_two", fresh.alias_one)
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert fresh.alias_one is fresh.alias_two
    assert torch.equal(fresh.alias_one, torch.ones(5))


def test_checkpoint_owner_preserves_ordinary_state_with_empty_packed_file(tmp_path):
    from nemo_automodel._transformers.qat_checkpoint import QATCheckpoint

    model = nn.Linear(32, 32, bias=False)
    model.register_buffer("strided", torch.arange(12).reshape(3, 4).t())
    model.register_parameter("alias", model.weight)
    state = QATCheckpoint.model_state(model)
    snapshot = {key: value.clone() for key, value in state.items()}
    assert state["weight"].data_ptr() == model.weight.data_ptr()
    assert not state["strided"].is_contiguous()
    QATCheckpoint.save(state, [], tmp_path)
    assert set(state) == set(snapshot)
    assert all(torch.equal(value, snapshot[key]) for key, value in state.items())
    assert _manifest(tmp_path)["weights"] == []
    assert load_file(str(tmp_path / "quantized.safetensors")) == {}
    fresh = nn.Linear(32, 32, bias=False)
    fresh.register_buffer("strided", torch.zeros(4, 3, dtype=torch.int64))
    fresh.register_parameter("alias", fresh.weight)
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert fresh.alias is fresh.weight
    assert all(torch.equal(value, snapshot[key]) for key, value in fresh.state_dict().items())


def test_load_rejects_conflicting_tied_ordinary_values_before_mutation(tmp_path):
    model = _patched().eval()
    model.register_parameter("alias_one", nn.Parameter(torch.ones(5), requires_grad=False))
    model.register_parameter("alias_two", model.alias_one)
    controller = _controller()
    controller.prepare(model)
    controller.export(model, tmp_path)
    path = tmp_path / "model.safetensors"
    ordinary = load_file(str(path))
    ordinary["alias_two"].zero_()
    save_file(ordinary, str(path))
    _refresh_hash(tmp_path, path.name)
    fresh = _architecture()
    fresh.register_parameter("alias_one", nn.Parameter(torch.full((5,), 2.0), requires_grad=False))
    fresh.register_parameter("alias_two", fresh.alias_one)
    before = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="alias_two: conflicting checkpoint values for tied weights"):
        QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert fresh.alias_one is fresh.alias_two
    assert all(torch.equal(value, before[key]) for key, value in fresh.state_dict().items())


def test_export_requires_preparation_eval_and_materialized_state(tmp_path):
    model = _patched()
    controller = _controller()
    with pytest.raises(ValueError, match="eval"):
        controller.export(model, tmp_path)
    model.eval()
    with pytest.raises(ValueError, match="prepare"):
        controller.export(model, tmp_path)
    controller.prepare(model)
    model[1].train()
    with pytest.raises(ValueError, match="eval"):
        controller.export(model, tmp_path)
    meta = _patched(device="meta")
    controller.prepare(meta)
    meta.eval()
    with pytest.raises(ValueError, match="meta"):
        controller.export(meta, tmp_path)


class _ExtraState(nn.Module):
    def get_extra_state(self):
        return {"not": "a tensor"}


def test_export_does_not_silently_drop_non_tensor_state(tmp_path):
    model = _patched()
    model.add_module("extra", _ExtraState())
    controller = _controller()
    controller.prepare(model)
    model.eval()
    with pytest.raises(TypeError, match="non-tensor"):
        controller.export(model, tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    model = _patched()
    controller = _controller()
    controller.prepare(model)
    model.eval()
    controller.export(model, tmp_path)
    return tmp_path


@pytest.mark.parametrize("omit", [(0,), (0, 1)])
def test_version_one_legacy_scale_field_is_optional_and_defaults_to_e8m0(checkpoint, omit):
    reference = _architecture().eval()
    QAT.load_quantized_checkpoint(reference, checkpoint)
    manifest = _manifest(checkpoint)
    for index in omit:
        del manifest["weights"][index]["config"]["scale_format"]
    assert manifest["schema_version"] == 1
    _write_manifest(checkpoint, manifest)
    fresh = _architecture().eval()
    QAT.load_quantized_checkpoint(fresh, checkpoint)
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in fresh.state_dict().items())
    inputs = torch.randn(3, 32)
    torch.testing.assert_close(fresh(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("scale_format", [None, "fp32", "ue8m0", 32, True, []])
def test_manifest_rejects_explicit_invalid_scale_format(checkpoint, scale_format):
    manifest = _manifest(checkpoint)
    manifest["weights"][0]["config"]["scale_format"] = scale_format
    _write_manifest(checkpoint, manifest)
    fresh = _architecture()
    snapshot = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="scale_format"):
        QAT.load_quantized_checkpoint(fresh, checkpoint)
    assert all(torch.equal(value, snapshot[key]) for key, value in fresh.state_dict().items())


@pytest.mark.parametrize("corruption", ["unknown", "missing_block", "mxfp4_float32", "encoding_mismatch"])
def test_manifest_scale_encoding_fails_closed(checkpoint, corruption):
    manifest = _manifest(checkpoint)
    if corruption == "unknown":
        manifest["weights"][0]["config"]["scale_dtype"] = "float32"
    elif corruption == "missing_block":
        del manifest["weights"][0]["config"]["block_size"]
    elif corruption == "mxfp4_float32":
        manifest["weights"][1]["config"]["scale_format"] = "float32"
    else:
        manifest["weights"][0]["config"]["scale_format"] = "float32"
    _write_manifest(checkpoint, manifest)
    fresh = _architecture()
    snapshot = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        QAT.load_quantized_checkpoint(fresh, checkpoint)
    assert all(torch.equal(value, snapshot[key]) for key, value in fresh.state_dict().items())


@pytest.mark.parametrize("corruption", ["zero", "negative", "nan", "inf", "float16", "uint8", "omitted_encoding"])
def test_float32_checkpoint_rejects_corrupt_scales_before_loading(tmp_path, corruption):
    model = _patched().eval()
    controller = _controller(scale_format="float32")
    controller.prepare(model)
    controller.export(model, tmp_path)
    path = tmp_path / "quantized.safetensors"
    packed = load_file(str(path))
    if corruption == "omitted_encoding":
        manifest = _manifest(tmp_path)
        del manifest["weights"][0]["config"]["scale_format"]
        _write_manifest(tmp_path, manifest)
    elif corruption in ("float16", "uint8"):
        packed["0.weight.scales"] = packed["0.weight.scales"].to(getattr(torch, corruption))
    else:
        packed["0.weight.scales"].fill_({"zero": 0.0, "negative": -1.0, "nan": float("nan"), "inf": float("inf")}[corruption])
    save_file(packed, str(path))
    _refresh_hash(tmp_path, path.name)
    fresh = _architecture()
    snapshot = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises((ValueError, TypeError), match="scales"):
        QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert all(torch.equal(value, snapshot[key]) for key, value in fresh.state_dict().items())


@pytest.mark.parametrize("corruption", [
    "missing_format", "missing_weight_format", "unknown_field", "version", "bool_version",
    "traversal", "unsafe_dtype", "layout", "shape", "duplicate_weight", "payload_name", "weight_key",
])
def test_load_rejects_invalid_metadata_without_mutation(checkpoint, corruption):
    manifest = _manifest(checkpoint)
    if corruption == "missing_format":
        del manifest["format"]
    elif corruption == "missing_weight_format":
        del manifest["weights"][0]["config"]["format"]
    elif corruption == "unknown_field":
        manifest["extra"] = 1
    elif corruption == "version":
        manifest["schema_version"] = 2
    elif corruption == "bool_version":
        manifest["schema_version"] = True
    elif corruption == "traversal":
        manifest["quantized_file"] = "../quantized.safetensors"
    elif corruption == "unsafe_dtype":
        manifest["weights"][0]["dtype"] = "__import__('os').system('false')"
    elif corruption == "layout":
        manifest["weights"][0]["layout"] = "experts_in_out"
    elif corruption == "shape":
        manifest["weights"][0]["shape"] = [64, 32]
    elif corruption == "duplicate_weight":
        manifest["weights"].append(manifest["weights"][0])
    elif corruption == "payload_name":
        manifest["weights"][0]["payload"] = "../other"
    else:
        manifest["weights"][0]["weight_key"] = "../weight"
    _write_manifest(checkpoint, manifest)
    fresh = _architecture()
    snapshot = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        QAT.load_quantized_checkpoint(fresh, checkpoint)
    assert all(torch.equal(value, snapshot[key]) for key, value in fresh.state_dict().items())


@pytest.mark.parametrize(
    "corruption", ["payload_shape", "scale_shape", "payload_dtype", "scale_code", "missing", "extra"]
)
def test_load_validates_actual_packed_tensors(checkpoint, corruption):
    path = checkpoint / "quantized.safetensors"
    packed = load_file(str(path))
    if corruption == "payload_shape":
        packed["2.weight.payload"] = torch.zeros(16, 32, dtype=torch.uint8)
    elif corruption == "scale_shape":
        packed["0.weight.scales"] = torch.zeros(1, 2, dtype=torch.uint8)
    elif corruption == "payload_dtype":
        packed["0.weight.payload"] = packed["0.weight.payload"].float()
    elif corruption == "scale_code":
        packed["0.weight.scales"].fill_(255)
    elif corruption == "missing":
        del packed["0.weight.payload"]
    else:
        packed["unlisted"] = torch.zeros(1, dtype=torch.uint8)
    save_file(packed, str(path))
    _refresh_hash(checkpoint, path.name)
    with pytest.raises((ValueError, TypeError)):
        QAT.load_quantized_checkpoint(_architecture(), checkpoint)


def test_load_rejects_hash_mismatch_non_lora_contract_and_wrong_architecture(checkpoint):
    with pytest.raises(TypeError, match="NON-LoRA"):
        QAT.load_quantized_checkpoint(_patched(), checkpoint)
    with pytest.raises(ValueError, match="shape/dtype"):
        QAT.load_quantized_checkpoint(_architecture(dtype=torch.float64), checkpoint)
    with pytest.raises(ValueError, match="keys mismatch"):
        QAT.load_quantized_checkpoint(nn.Linear(32, 32), checkpoint)
    with (checkpoint / "quantized.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        QAT.load_quantized_checkpoint(_architecture(), checkpoint)


@pytest.mark.parametrize("corruption", ["ordinary_shape", "ordinary_dtype", "duplicate_weight"])
def test_mixed_checkpoint_state_is_rejected_before_loading(checkpoint, corruption):
    path = checkpoint / "model.safetensors"
    ordinary = load_file(str(path))
    if corruption == "ordinary_shape":
        ordinary["2.bias"] = torch.zeros(31)
    elif corruption == "ordinary_dtype":
        ordinary["2.bias"] = ordinary["2.bias"].double()
    else:
        ordinary["0.weight"] = torch.zeros(32, 32)
    save_file(ordinary, str(path))
    _refresh_hash(checkpoint, path.name)
    fresh = _architecture()
    before = {key: value.clone() for key, value in fresh.state_dict().items()}
    with pytest.raises(ValueError):
        QAT.load_quantized_checkpoint(fresh, checkpoint)
    assert all(torch.equal(value, before[key]) for key, value in fresh.state_dict().items())


def test_load_rejects_duplicate_json_fields_and_symlink_escape(checkpoint, tmp_path):
    path = checkpoint / "manifest.json"
    original = path.read_text()
    path.write_text(original.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match="duplicate JSON"):
        QAT.load_quantized_checkpoint(_architecture(), checkpoint)
    path.write_text(original)
    destination = tmp_path / "outside.safetensors"
    (checkpoint / "model.safetensors").rename(destination)
    # Use a separate directory so even a same-filesystem symlink escapes the checkpoint root.
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("manifest.json", "quantized.safetensors"):
        (nested / name).write_bytes((checkpoint / name).read_bytes())
    (nested / "model.safetensors").symlink_to(destination)
    with pytest.raises(ValueError, match="escapes directory"):
        QAT.load_quantized_checkpoint(_architecture(), nested)


def _experts(*, deepep=False):
    config = MoEConfig(
        n_routed_experts=2, n_shared_experts=0, n_activated_experts=1,
        n_expert_groups=1, n_limited_groups=1, train_gate=False,
        gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="softmax",
        route_scale=1.0, dim=32, inter_dim=32, moe_inter_dim=64,
        norm_topk_prob=True, dtype=torch.float32,
    )
    module = GroupedExpertsDeepEP(config) if deepep else GroupedExperts(config)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.04)
    return module


@pytest.mark.parametrize("deepep", [False, True])
@pytest.mark.parametrize("format, block, scale_format", [
    ("fp8", (32, 32), "e8m0"), ("mxfp4", (1, 32), "e8m0"), ("fp8", (32, 32), "float32"),
])
def test_grouped_export_canonical_axes_and_reload_without_lora(tmp_path, deepep, format, block, scale_format):
    base = _experts(deepep=deepep)
    model = nn.ModuleDict({"experts": patch_moe_module(base, dim=4, alpha=8)})
    experts = model["experts"]
    experts.use_torch_mm = True
    with torch.no_grad():
        experts.lora_gate_and_up_B.normal_(std=0.1)
        experts.lora_down_B.normal_(std=0.1)
    controller = QAT(QATConfig(
        target_modules=("experts",), weight=WeightQuantizationConfig(format, block, scale_format=scale_format)
    ))
    controller.prepare(model)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    controller.export(model, tmp_path)
    fresh = nn.ModuleDict({"experts": _experts(deepep=deepep)})
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    for weight, a, b in (
        ("gate_and_up_projs", "lora_gate_and_up_A", "lora_gate_and_up_B"),
        ("down_projs", "lora_down_A", "lora_down_B"),
    ):
        canonical = (
            getattr(experts, weight) + experts.scale * (getattr(experts, a) @ getattr(experts, b))
        ).transpose(-2, -1)
        expected = experts.weight_fake_quantizer(canonical).transpose(-2, -1)
        assert torch.equal(getattr(fresh["experts"], weight), expected)
    manifest = _manifest(tmp_path)
    shapes = {entry["weight_key"]: entry["shape"] for entry in manifest["weights"]}
    assert shapes["experts.gate_and_up_projs"] == [2, 128, 32]
    assert shapes["experts.down_projs"] == [2, 32, 64]
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_untargeted_grouped_lora_is_merged_in_original_storage_order(tmp_path):
    model = nn.ModuleDict({
        "dense": _patched(),
        "experts": patch_moe_module(_experts(), dim=4, alpha=8),
    })
    experts = model["experts"]
    with torch.no_grad():
        experts.lora_gate_and_up_B.normal_(std=0.1)
        experts.lora_down_B.normal_(std=0.1)
    controller = _one_rule("dense.[02]")
    controller.prepare(model)
    model.eval()
    controller.export(model, tmp_path)
    fresh = nn.ModuleDict({"dense": _architecture(), "experts": _experts()})
    QAT.load_quantized_checkpoint(fresh, tmp_path)
    assert torch.equal(
        fresh["experts"].gate_and_up_projs,
        experts.gate_and_up_projs + experts.scale * torch.bmm(experts.lora_gate_and_up_A, experts.lora_gate_and_up_B),
    )
    assert torch.equal(
        fresh["experts"].down_projs,
        experts.down_projs + experts.scale * torch.bmm(experts.lora_down_A, experts.lora_down_B),
    )


@pytest.mark.parametrize("invalid", ["backend", "dtype", "shape"])
def test_grouped_validation_rejects_backend_dtype_and_shape(invalid):
    experts = patch_moe_module(_experts(), dim=4)
    if invalid == "backend":
        experts.use_mxfp8 = True
    elif invalid == "dtype":
        experts.lora_down_B = nn.Parameter(experts.lora_down_B.to(torch.bfloat16))
    else:
        experts.down_projs = nn.Parameter(torch.empty(2, 31, 32), requires_grad=False)
    with pytest.raises((TypeError, ValueError)):
        _one_rule("").prepare(experts)
    assert experts.weight_fake_quantizer is None


def test_existing_dtensor_is_rejected_for_prepare_export_and_load(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = _patched()
        sharded = DTensor.from_local(model[0].weight.detach(), mesh, [Replicate()])
        model[0].weight = nn.Parameter(sharded, requires_grad=False)
        with pytest.raises(TypeError, match="DTensor"):
            _controller().prepare(model)
        assert model[2].weight_fake_quantizer is None
        model.eval()
        with pytest.raises(TypeError, match="DTensor"):
            _controller().export(model, tmp_path)
        fresh = _architecture()
        fresh[0].weight = nn.Parameter(sharded, requires_grad=False)
        with pytest.raises(TypeError, match="DTensor"):
            QAT.load_quantized_checkpoint(fresh, tmp_path)
    finally:
        dist.destroy_process_group()