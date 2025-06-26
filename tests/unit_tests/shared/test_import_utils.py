# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import sys
import types
import uuid

import pytest
from packaging.version import Version as PkgVersion

import nemo_automodel.shared.import_utils as si


@pytest.fixture(scope="module")
def dummy_module() -> types.ModuleType:
    """
    Create an *empty* module that is present in ``sys.modules`` for the duration
    of the test session.  This allows us to test the *symbol-missing* branch of
    ``safe_import_from`` without provoking an ImportError.
    """
    name = f"dummy_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    yield module
    sys.modules.pop(name, None)


def _random_module_name() -> str:
    """Return a module name that is (very likely) missing."""
    return f"no_such_mod_{uuid.uuid4().hex}"


def test_safe_import_success():
    """Importing an existing std-lib module should succeed."""
    ok, mod = si.safe_import("math")
    assert ok is True
    assert mod.sqrt(4) == 2


def test_safe_import_placeholder():
    """
    Importing a non-existent module must:

    * return ``ok is False``,
    * return an *Unavailable* placeholder,
    * make that placeholder detectable via ``is_unavailable``.
    """
    ok, placeholder = si.safe_import(_random_module_name())
    assert ok is False
    assert si.is_unavailable(placeholder)


def test_safe_import_alt_object():
    """
    When an ``alt`` object is supplied, it must be returned instead of the
    placeholder and the success flag must still be *False*.
    """
    sentinel = object()
    ok, obj = si.safe_import(_random_module_name(), alt=sentinel)
    assert ok is False
    assert obj is sentinel


def test_safe_import_from_success():
    """Happy path: import a known symbol from a known module."""
    ok, sqrt = si.safe_import_from("math", "sqrt")
    assert ok is True
    assert sqrt(9) == 3


def test_safe_import_from_fallback(dummy_module):
    """
    Symbol is missing from *primary* module but present in *fallback* module.
    This should succeed and return the symbol from the fallback.
    """
    ok, sym = si.safe_import_from(dummy_module.__name__, "sqrt", fallback_module="math")
    assert ok is True
    assert sym(16) == 4


def test_safe_import_from_placeholder():
    """Module does not exist → placeholder result."""
    ok, placeholder = si.safe_import_from(_random_module_name(), "foo")
    assert ok is False
    assert si.is_unavailable(placeholder)


def test_safe_import_from_alt_object():
    """Verify the ``alt`` path."""
    sentinel = object()
    ok, obj = si.safe_import_from(_random_module_name(), "foo", alt=sentinel)
    assert ok is False
    assert obj is sentinel


def test_gpu_only_import_success():
    ok, mod = si.gpu_only_import("math")
    assert ok is True
    assert mod.pi == 3.141592653589793


def test_gpu_only_import_placeholder():
    ok, placeholder = si.gpu_only_import(_random_module_name())
    assert ok is False
    assert si.is_unavailable(placeholder)
    # The placeholder message must mention the special GPU install hint.
    assert "cuda" in getattr(placeholder, "_msg", "").lower()


def test_gpu_only_import_from_success():
    ok, const_pi = si.gpu_only_import_from("math", "pi")
    assert ok is True
    assert const_pi == 3.141592653589793


def test_gpu_only_import_from_placeholder():
    ok, placeholder = si.gpu_only_import_from(_random_module_name(), "foo")
    assert ok is False
    assert si.is_unavailable(placeholder)


def test_get_torch_version_type():
    """
    ``get_torch_version`` should *never* raise – even when torch is unavailable
    while building docs – and must always return a ``packaging.version.Version``.
    """
    ver = si.get_torch_version()
    assert isinstance(ver, PkgVersion)


def test_is_torch_min_version():
    """
    * A ridiculously low requirement must be satisfied.
    * A far-future version must *not* be satisfied.
    """
    assert si.is_torch_min_version("0.0.0") is True
    assert si.is_torch_min_version("9999.0.0", check_equality=False) is False


def test_is_unavailable_identifies_placeholder():
    """
    Direct construction of an *Unavailable* placeholder should round-trip
    through the predicate.
    """
    _, placeholder = si.safe_import(_random_module_name())
    assert si.is_unavailable(placeholder) is True
