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

"""Unit tests for the QLoRA configuration contract (CPU-only, no bitsandbytes)."""

from types import SimpleNamespace

import pytest
from torch import nn

from nemo_automodel.components.quantization import qlora


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: bool | str) -> None:
        self.kwargs = kwargs


class _FakeQuantState:
    pass


class _FakeQuantStateSubclass(_FakeQuantState):
    pass


class _FakeConfig:
    def __init__(self, values: dict[str, bool | str]) -> None:
        self._values = values

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def get(self, key: str, default: bool | str) -> bool | str:
        return self._values.get(key, default)


@pytest.fixture
def fake_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qlora, "HAS_BNB", True)
    monkeypatch.setattr(qlora, "HAS_TRANSFORMERS", True)
    monkeypatch.setattr(qlora, "transformers", SimpleNamespace(BitsAndBytesConfig=_FakeBitsAndBytesConfig))
    monkeypatch.setattr(
        qlora,
        "bitsandbytes",
        SimpleNamespace(functional=SimpleNamespace(QuantState=_FakeQuantState)),
    )


def test_create_bnb_config_4bit_defaults(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(_FakeConfig({"load_in_4bit": True}))

    assert isinstance(result, _FakeBitsAndBytesConfig)
    assert result.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_quant_storage": "bfloat16",
    }


def test_create_bnb_config_4bit_overrides(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(
        _FakeConfig(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_quant_storage": "float16",
            }
        )
    )

    assert isinstance(result, _FakeBitsAndBytesConfig)
    assert result.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_type": "fp4",
        "bnb_4bit_quant_storage": "float16",
    }


def test_create_bnb_config_8bit_only(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(_FakeConfig({"load_in_8bit": True}))

    assert isinstance(result, _FakeBitsAndBytesConfig)
    assert result.kwargs == {"load_in_8bit": True}


def test_create_bnb_config_8bit_when_4bit_disabled(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(_FakeConfig({"load_in_4bit": False, "load_in_8bit": True}))

    assert isinstance(result, _FakeBitsAndBytesConfig)
    assert result.kwargs == {"load_in_8bit": True}


def test_create_bnb_config_4bit_wins_when_both_enabled(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(_FakeConfig({"load_in_4bit": True, "load_in_8bit": True}))

    assert isinstance(result, _FakeBitsAndBytesConfig)
    assert result.kwargs["load_in_4bit"] is True
    assert "load_in_8bit" not in result.kwargs


def test_create_bnb_config_explicit_disable_returns_none(fake_optional_deps) -> None:
    result = qlora.create_bnb_config(_FakeConfig({"load_in_4bit": False, "load_in_8bit": False}))

    assert result is None


def test_create_bnb_config_requires_a_load_flag(fake_optional_deps) -> None:
    with pytest.raises(ValueError, match="load_in_4bit.*load_in_8bit"):
        qlora.create_bnb_config(_FakeConfig({}))


class _QuantizedModule(nn.Module):
    def __init__(self, state_type: type[_FakeQuantState] = _FakeQuantState) -> None:
        super().__init__()
        self.quant_state = state_type()


class _PlainModule(nn.Module):
    pass


class _Wrapper(nn.Module):
    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child


def test_verify_qlora_quantization_finds_quantized_child(fake_optional_deps) -> None:
    assert qlora.verify_qlora_quantization(_Wrapper(_QuantizedModule())) is True


def test_verify_qlora_quantization_returns_false_without_quant_state(fake_optional_deps) -> None:
    assert qlora.verify_qlora_quantization(_Wrapper(_PlainModule())) is False


def test_verify_qlora_quantization_ignores_unrelated_state(fake_optional_deps) -> None:
    module = _PlainModule()
    module.quant_state = "not a quant state"

    assert qlora.verify_qlora_quantization(_Wrapper(module)) is False


def test_verify_qlora_quantization_accepts_subclass(fake_optional_deps) -> None:
    assert qlora.verify_qlora_quantization(_Wrapper(_QuantizedModule(_FakeQuantStateSubclass))) is True
