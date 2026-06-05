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

"""Tests for lazy exports in nemo_automodel.recipes.dllm."""

import types

import pytest

import nemo_automodel.recipes.dllm as dllm_pkg


def test_lazy_getattr_imports_configured_recipe(monkeypatch):
    fake_recipe = object()
    calls = []

    def import_module(module_path, package):
        calls.append((module_path, package))
        return types.SimpleNamespace(DiffusionLMSFTRecipe=fake_recipe)

    monkeypatch.setattr(dllm_pkg.importlib, "import_module", import_module)

    assert dllm_pkg.__getattr__("DiffusionLMSFTRecipe") is fake_recipe
    assert calls == [(".train_ft", dllm_pkg.__name__)]
    assert dllm_pkg.__all__ == ["DiffusionLMSFTRecipe"]


def test_lazy_getattr_rejects_unknown_name():
    with pytest.raises(AttributeError, match="does_not_exist"):
        dllm_pkg.__getattr__("does_not_exist")
