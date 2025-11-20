# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import logging

import pytest
import torch
import torch.nn as nn
import types
import importlib
import sys

import nemo_automodel.components.quantization.qat as qat


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class FakeQuantizer:
    def __init__(self) -> None:
        self.prepared_with = None

    def prepare(self, model: nn.Module) -> nn.Module:
        self.prepared_with = model
        return IdentityWrapper(model)


def test_prepare_qat_model_known_mode(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)
    monkeypatch.setitem(qat._QUANTIZER_TO_MODE, FakeQuantizer, "test-qat")

    model = SimpleMLP()
    quantizer = FakeQuantizer()

    prepared, mode = qat.prepare_qat_model(model, quantizer)

    assert isinstance(prepared, IdentityWrapper)
    assert quantizer.prepared_with is model
    assert mode == "test-qat"

    # quick functional check
    x = torch.randn(2, 8)
    y = prepared(x)
    assert y.shape == (2, 4)


def test_prepare_qat_model_unknown_mode_warns(monkeypatch, caplog):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)
    if FakeQuantizer in qat._QUANTIZER_TO_MODE:
        monkeypatch.delitem(qat._QUANTIZER_TO_MODE, FakeQuantizer, raising=False)

    model = SimpleMLP()
    quantizer = FakeQuantizer()

    with caplog.at_level(logging.WARNING):
        prepared, mode = qat.prepare_qat_model(model, quantizer)

    assert isinstance(prepared, IdentityWrapper)
    assert mode is None
    assert "Unknown QAT quantizer" in caplog.text


def test_prepare_qat_model_missing_prepare_raises(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)

    model = SimpleMLP()

    class NotAQuantizer:
        pass

    with pytest.raises(ValueError):
        qat.prepare_qat_model(model, NotAQuantizer())


def test_prepare_qat_model_raises_without_torchao(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", False, raising=False)

    model = SimpleMLP()
    quantizer = FakeQuantizer()

    with pytest.raises(ImportError):
        qat.prepare_qat_model(model, quantizer)


def _install_fake_torchao(monkeypatch: pytest.MonkeyPatch):
    # Build a minimal fake torchao package tree needed by qat helpers
    torchao_mod = types.ModuleType("torchao")
    quantization_mod = types.ModuleType("torchao.quantization")
    qat_pkg = types.ModuleType("torchao.quantization.qat")
    qat_linear_mod = types.ModuleType("torchao.quantization.qat.linear")
    qat_api_mod = types.ModuleType("torchao.quantization.qat.api")
    qat_fake_quantizer_mod = types.ModuleType("torchao.quantization.qat.fake_quantizer")

    class Int4WeightOnlyQATQuantizer:
        def prepare(self, model: nn.Module) -> nn.Module:
            setattr(model, "quantizer_applied", "4w")
            return model

    class Int8DynActInt4WeightQATQuantizer:
        def prepare(self, model: nn.Module) -> nn.Module:
            setattr(model, "quantizer_applied", "8da4w")
            return model

    def disable_4w_fake_quant(*_args, **_kwargs):
        return "disabled_4w"

    def disable_8da4w_fake_quant(*_args, **_kwargs):
        return "disabled_8da4w"

    def enable_4w_fake_quant(*_args, **_kwargs):
        return "enabled_4w"

    def enable_8da4w_fake_quant(*_args, **_kwargs):
        return "enabled_8da4w"

    class FakeQuantizeConfig:  # sentinel type for isinstance checks
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeQuantizer(nn.Module):
        def __init__(self, cfg: FakeQuantizeConfig | None = None) -> None:
            super().__init__()
            self.cfg = cfg
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            self.calls += 1
            return x

    # Attach into fake modules
    qat_pkg.Int4WeightOnlyQATQuantizer = Int4WeightOnlyQATQuantizer
    qat_pkg.Int8DynActInt4WeightQATQuantizer = Int8DynActInt4WeightQATQuantizer
    qat_linear_mod.disable_4w_fake_quant = disable_4w_fake_quant
    qat_linear_mod.disable_8da4w_fake_quant = disable_8da4w_fake_quant
    qat_linear_mod.enable_4w_fake_quant = enable_4w_fake_quant
    qat_linear_mod.enable_8da4w_fake_quant = enable_8da4w_fake_quant
    qat_api_mod.FakeQuantizeConfig = FakeQuantizeConfig
    qat_fake_quantizer_mod.FakeQuantizer = FakeQuantizer

    # Wire sys.modules
    monkeypatch.setitem(sys.modules, "torchao", torchao_mod)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization_mod)
    monkeypatch.setitem(sys.modules, "torchao.quantization.qat", qat_pkg)
    monkeypatch.setitem(sys.modules, "torchao.quantization.qat.linear", qat_linear_mod)
    monkeypatch.setitem(sys.modules, "torchao.quantization.qat.api", qat_api_mod)
    monkeypatch.setitem(sys.modules, "torchao.quantization.qat.fake_quantizer", qat_fake_quantizer_mod)

    # Ensure our qat module sees torchao as available
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)
    # Patch mappings consistent with the module-level initialization
    monkeypatch.setitem(qat._QUANTIZER_TO_MODE, qat_pkg.Int8DynActInt4WeightQATQuantizer, "8da4w-qat")
    monkeypatch.setitem(qat._QUANTIZER_TO_MODE, qat_pkg.Int4WeightOnlyQATQuantizer, "4w-qat")
    monkeypatch.setitem(qat._DISABLE_FN_BY_MODE, "8da4w-qat", qat_linear_mod.disable_8da4w_fake_quant)
    monkeypatch.setitem(qat._ENABLE_FN_BY_MODE, "8da4w-qat", qat_linear_mod.enable_8da4w_fake_quant)
    monkeypatch.setitem(qat._DISABLE_FN_BY_MODE, "4w-qat", qat_linear_mod.disable_4w_fake_quant)
    monkeypatch.setitem(qat._ENABLE_FN_BY_MODE, "4w-qat", qat_linear_mod.enable_4w_fake_quant)

    return types.SimpleNamespace(
        Int4WeightOnlyQATQuantizer=Int4WeightOnlyQATQuantizer,
        Int8DynActInt4WeightQATQuantizer=Int8DynActInt4WeightQATQuantizer,
        disable_4w_fake_quant=disable_4w_fake_quant,
        disable_8da4w_fake_quant=disable_8da4w_fake_quant,
        enable_4w_fake_quant=enable_4w_fake_quant,
        enable_8da4w_fake_quant=enable_8da4w_fake_quant,
        FakeQuantizeConfig=FakeQuantizeConfig,
        FakeQuantizer=FakeQuantizer,
    )


def test_get_quantizer_mode_known_and_unknown(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)

    class QZ:
        pass

    monkeypatch.setitem(qat._QUANTIZER_TO_MODE, QZ, "qz-mode")
    assert qat.get_quantizer_mode(QZ()) == "qz-mode"

    class Unk:
        pass

    assert qat.get_quantizer_mode(Unk()) is None


def test_get_enable_disable_fake_quant(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", True, raising=False)

    def dfn():
        return "d"

    def efn():
        return "e"

    monkeypatch.setitem(qat._DISABLE_FN_BY_MODE, "x", dfn)
    monkeypatch.setitem(qat._ENABLE_FN_BY_MODE, "x", efn)

    assert qat.get_disable_fake_quant_fn("x") is dfn
    assert qat.get_enable_fake_quant_fn("x") is efn

    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", False, raising=False)
    assert qat.get_disable_fake_quant_fn("x") is None
    assert qat.get_enable_fake_quant_fn("x") is None


def test_prepare_qat_model_with_fake_torchao(monkeypatch):
    fake = _install_fake_torchao(monkeypatch)

    model = SimpleMLP()
    m1, mode1 = qat.prepare_qat_model(model, fake.Int8DynActInt4WeightQATQuantizer())
    assert m1 is model and mode1 == "8da4w-qat" and getattr(model, "quantizer_applied") == "8da4w"

    model2 = SimpleMLP()
    m2, mode2 = qat.prepare_qat_model(model2, fake.Int4WeightOnlyQATQuantizer())
    assert m2 is model2 and mode2 == "4w-qat" and getattr(model2, "quantizer_applied") == "4w"


class DummyLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # Minimal LoRA path
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = 1.0
        self.dropout_p = 0.0
        self.dropout_position = "post"  # default in qat implementation


def test_attach_qat_to_lora_linear_and_forward(monkeypatch):
    fake = _install_fake_torchao(monkeypatch)
    linear = DummyLoRALinear(8, 4, rank=3)
    # Deterministic
    torch.manual_seed(0)
    x = torch.randn(2, 8)

    act_cfg = fake.FakeQuantizeConfig("act")
    w_cfg = fake.FakeQuantizeConfig("w")

    # Attach QAT to LoRA linear
    qat._attach_qat_to_lora_linear(linear, act_cfg, w_cfg)

    # Attributes present
    assert hasattr(linear, "activation_fake_quantizer")
    assert hasattr(linear, "weight_fake_quantizer")
    assert isinstance(linear.activation_fake_quantizer, nn.Module)
    assert isinstance(linear.weight_fake_quantizer, nn.Module)

    # Class swapped (type name prefixed with "QAT")
    assert type(linear).__name__.startswith("QAT")

    # Forward matches expected computation (FakeQuantizer is identity)
    expected = torch.nn.functional.linear(x, linear.weight, None) + linear.lora_B(linear.lora_A(x)) * linear.scale
    out = linear(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_attach_qat_to_lora_linear_invalid_types_and_conflict(monkeypatch):
    _install_fake_torchao(monkeypatch)
    linear = DummyLoRALinear(4, 2, rank=2)

    with pytest.raises(TypeError):
        qat._attach_qat_to_lora_linear(linear, activation_qat_config="bad", weight_qat_config=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        qat._attach_qat_to_lora_linear(linear, activation_qat_config=None, weight_qat_config="bad")  # type: ignore[arg-type]

    # QLoRA base quantization conflict
    linear2 = DummyLoRALinear(4, 2, rank=2)
    setattr(linear2, "quant_state", object())
    with pytest.raises(ValueError):
        qat._attach_qat_to_lora_linear(linear2, None, None)


def test_swap_lora_linear_with_qat_traversal(monkeypatch):
    _install_fake_torchao(monkeypatch)

    class Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(DummyLoRALinear(6, 3, rank=2))

    model = Container()
    child: DummyLoRALinear = model.block[0]  # type: ignore[assignment]
    assert not type(child).__name__.startswith("QAT")

    qat.swap_lora_linear_with_qat(model, None, None)

    assert type(model.block[0]).__name__.startswith("QAT")  # type: ignore[index]
    # And forward works
    x = torch.randn(1, 6)
    out = model.block[0](x)  # type: ignore[index]
    assert out.shape == (1, 3)


def test_swap_lora_linear_with_qat_without_torchao_raises(monkeypatch):
    monkeypatch.setattr(qat, "HAVE_TORCHAO_QAT", False, raising=False)
    model = nn.Sequential(DummyLoRALinear(5, 2, rank=2))
    with pytest.raises(ImportError):
        qat.swap_lora_linear_with_qat(model, None, None)

