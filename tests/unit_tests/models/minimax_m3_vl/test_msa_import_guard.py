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

"""The MSA kernel modules must signal a missing CuTe DSL the way the CI import walker expects.

The ``Installation Test`` workflow imports every module of the installed package on macOS, where
neither ``nvidia-cutlass-dsl`` nor ``cuda-bindings`` ships a wheel, and counts a module as gracefully
handled only when the traceback carries ``UnavailableError``. These tests replay that rule on CPU by
refusing every import of the msa extra (``cutlass``, ``cuda``, ``fmha_sm100``) through a
``sys.meta_path`` finder. Patching
``builtins.__import__`` (the technique of the TileLang precedent tests) is not enough here because
``safe_import`` goes through ``importlib.import_module``; and the real ``cutlass`` must be restored
on teardown rather than re-imported, since loading its MLIR extension twice aborts the interpreter.
"""

import importlib
import pkgutil
import sys

import pytest

from nemo_automodel.components.models.minimax_m3_vl import kernels, msa_bindings
from nemo_automodel.shared.import_utils import UnavailableError

_DSL_PACKAGES = ("cutlass", "cuda", "fmha_sm100")
_KERNEL_MODULES = sorted(f"{kernels.__name__}.{module.name}" for module in pkgutil.iter_modules(kernels.__path__))


class _BlockCuteDsl:
    """Meta-path finder that refuses every import of the CuTe DSL, the CUDA driver bindings and official MSA."""

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in _DSL_PACKAGES:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture
def without_cute_dsl(monkeypatch):
    """Simulate a host without the msa extra; teardown restores the real modules without reloading them."""
    for name in [name for name in sys.modules if name.partition(".")[0] in _DSL_PACKAGES]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BlockCuteDsl(), *sys.meta_path])
    msa_bindings.kernels.cache_clear()
    yield
    msa_bindings.kernels.cache_clear()


@pytest.mark.parametrize("module_name", _KERNEL_MODULES)
def test_kernel_modules_import_or_raise_unavailable_without_the_dsl(module_name, without_cute_dsl, monkeypatch):
    # The walker's own rule: every module either imports or fails with UnavailableError, never with
    # anything else. Parametrizing over the package directory covers kernel files added later.
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    try:
        importlib.import_module(module_name)
    except UnavailableError:
        pass
    finally:
        sys.modules.pop(module_name, None)


def test_require_cute_dsl_names_the_msa_extra(without_cute_dsl):
    with pytest.raises(UnavailableError, match=r"uv sync --extra msa"):
        kernels.require_cute_dsl()


def test_kernels_raise_unavailable_without_the_msa_extra(without_cute_dsl, monkeypatch):
    # Already-imported kernel modules would bypass require_cute_dsl; a host without the extra has none.
    for module_name in _KERNEL_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(UnavailableError, match=r"uv sync --extra msa"):
        msa_bindings.kernels()


def test_require_cute_dsl_passes_with_the_dsl_installed():
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    assert kernels.require_cute_dsl() is None
