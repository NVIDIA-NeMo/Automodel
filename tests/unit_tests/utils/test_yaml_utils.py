# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import enum
import functools

import pytest
import yaml

from nemo_automodel.components.utils.yaml_utils import safe_yaml_representers


class Colour(enum.Enum):
    RED = 1
    BLUE = 2


def some_func(x, y):
    return x + y


class SampleClass:
    def __init__(self, value: int):
        self.value = value


def _roundtrip(obj):
    """
    Dump `obj` to YAML inside the context manager and immediately load it back.
    Returns the loaded Python object (which should be a plain dict due to the
    representer design).
    """
    with safe_yaml_representers():
        dumped = yaml.safe_dump(obj)
        loaded = yaml.safe_load(dumped)
    return dumped, loaded


def test_function_representer():
    dumped, loaded = _roundtrip(some_func)

    assert loaded["_call_"] is False
    # fully-qualified name ends with the original symbol name
    assert loaded["_target_"].endswith(f"{some_func.__qualname__}")


def test_partial_representer():
    p = functools.partial(some_func, 3, y=4)
    dumped, loaded = _roundtrip(p)

    assert loaded["_partial_"] is True
    assert loaded["_args_"] == [3]  # original positional argument
    # kwargs from the partial should be present in the mapping
    assert loaded["y"] == 4
    assert loaded["_target_"].endswith(some_func.__qualname__)


def test_enum_representer():
    dumped, loaded = _roundtrip(Colour.RED)

    assert loaded["_call_"] is True
    assert loaded["_args_"] == [Colour.RED.value]
    assert loaded["_target_"].endswith(f"{Colour.__qualname__}")


def test_generic_object_representer():
    inst = SampleClass(7)
    dumped, loaded = _roundtrip(inst)

    # For normal instances `_call_` should be True
    assert loaded["_call_"] is True
    assert loaded["_target_"].endswith(f"{SampleClass.__qualname__}")


def test_representers_are_restored():
    """
    After exiting the context manager, trying to dump an otherwise unsupported
    object (e.g., a bare function) must raise the usual RepresenterError.
    """
    with safe_yaml_representers():  # Inside: should *not* raise
        yaml.safe_dump(some_func)

    # Outside: the original representers must be back
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(some_func)


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="PyTorch not installed") is None,
    reason="PyTorch not installed",
)
def test_torch_dtype_representer():
    import torch

    dumped, loaded = _roundtrip(torch.float32)
    assert loaded["_target_"] == "torch.float32"
    assert loaded["_call_"] is False


@pytest.mark.skipif(
    pytest.importorskip("transformers", reason="transformers not installed") is None,
    reason="transformers not installed",
)
def test_generation_config_representer():
    from transformers import GenerationConfig

    cfg = GenerationConfig.from_pretrained("gpt2")
    dumped, loaded = _roundtrip(cfg)

    assert loaded["_call_"] is True
    assert loaded["_target_"].endswith("GenerationConfig.from_dict")
    # Should contain a nested dictionary with at least one known key
    assert "config_dict" in loaded
    assert isinstance(loaded["config_dict"], dict)
