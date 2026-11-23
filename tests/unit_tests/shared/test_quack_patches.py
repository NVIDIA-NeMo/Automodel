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

import sys
import types

import pytest

from nemo_automodel.shared.quack_patches import restore_quack_sub_packed_f32x2


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch, activation_attrs: dict, arch_attrs: dict
) -> types.ModuleType:
    """Install fake ``quack.activation`` and ``cutlass.cute.arch`` modules; return the fake activation module."""
    for name in ("quack", "cutlass", "cutlass.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    activation = types.ModuleType("quack.activation")
    activation.__dict__.update(activation_attrs)
    arch = types.ModuleType("cutlass.cute.arch")
    arch.__dict__.update(arch_attrs)
    monkeypatch.setitem(sys.modules, "quack.activation", activation)
    monkeypatch.setitem(sys.modules, "cutlass.cute.arch", arch)
    return activation


def test_binds_the_dsl_helper_when_quack_dropped_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def dsl_helper(src_a, src_b):
        return src_a, src_b

    activation = _install_fake_modules(monkeypatch, {}, {"sub_packed_f32x2": dsl_helper})

    restore_quack_sub_packed_f32x2()

    assert activation.sub_packed_f32x2 is dsl_helper


def test_keeps_quack_helper_that_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    def quack_helper(src_a, src_b):
        return src_a, src_b

    def dsl_helper(src_a, src_b):
        return src_b, src_a

    activation = _install_fake_modules(
        monkeypatch, {"sub_packed_f32x2": quack_helper}, {"sub_packed_f32x2": dsl_helper}
    )

    restore_quack_sub_packed_f32x2()

    assert activation.sub_packed_f32x2 is quack_helper


def test_leaves_quack_alone_without_the_dsl_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    activation = _install_fake_modules(monkeypatch, {}, {})

    restore_quack_sub_packed_f32x2()

    assert not hasattr(activation, "sub_packed_f32x2")


def test_does_nothing_without_quack(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes the import raise ImportError, as for an uninstalled package.
    monkeypatch.setitem(sys.modules, "quack.activation", None)

    restore_quack_sub_packed_f32x2()
