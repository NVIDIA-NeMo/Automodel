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


