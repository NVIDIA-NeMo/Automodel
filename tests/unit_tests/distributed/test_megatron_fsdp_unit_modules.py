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
"""Unit tests for auto-deriving MegatronFSDP unit modules from ``_no_split_modules``."""

import pytest
from torch import nn

from nemo_automodel.components.distributed.parallelizer import _derive_megatron_fsdp_unit_modules


class FooLayer(nn.Module):
    """Stand-in transformer block whose class name is listed in ``_no_split_modules``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)


class BarLayer(nn.Module):
    """Second block class, used to emulate VLM/MoE models with multiple wrap classes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)


class _ModelWithNoSplit(nn.Module):
    """Tiny model exposing a configurable ``_no_split_modules`` for derivation tests."""

    def __init__(self, no_split_modules: list[str]) -> None:
        super().__init__()
        self._no_split_modules = no_split_modules
        self.layers = nn.ModuleList([FooLayer(), FooLayer()])
        self.head = BarLayer()


def test_derives_single_class_from_no_split_modules():
    """A ``_no_split_modules=["FooLayer"]`` model derives exactly ``[FooLayer]``."""
    model = _ModelWithNoSplit(["FooLayer"])

    derived = _derive_megatron_fsdp_unit_modules(model)

    assert derived == [FooLayer]


def test_derives_multiple_classes_for_vlm_style_no_split():
    """Multiple names (e.g. vision + language blocks) each resolve, de-duplicated by class."""
    model = _ModelWithNoSplit(["FooLayer", "BarLayer"])

    derived = _derive_megatron_fsdp_unit_modules(model)

    # Two FooLayer instances collapse to a single class; BarLayer appears once.
    assert set(derived) == {FooLayer, BarLayer}
    assert len(derived) == 2


def test_raises_when_no_split_modules_absent():
    """A model without ``_no_split_modules`` fails loud (not a later ZeroDivisionError)."""
    model = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="_no_split_modules"):
        _derive_megatron_fsdp_unit_modules(model)


def test_raises_when_no_split_modules_empty():
    """An empty ``_no_split_modules`` fails loud with an actionable message."""
    model = _ModelWithNoSplit([])

    with pytest.raises(ValueError, match="_no_split_modules"):
        _derive_megatron_fsdp_unit_modules(model)


def test_raises_when_no_names_match_instantiated_submodule():
    """Names that match no instantiated submodule fail loud instead of wrapping zero modules."""
    model = _ModelWithNoSplit(["NonexistentLayer"])

    with pytest.raises(ValueError, match="matched an instantiated submodule"):
        _derive_megatron_fsdp_unit_modules(model)
