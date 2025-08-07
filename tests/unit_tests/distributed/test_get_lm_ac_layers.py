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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import List

import pytest
import torch.nn as nn

from nemo_automodel.components.distributed.parallelizer import get_lm_ac_layers


def _install_fake_gemma3(monkeypatch):
    """Dynamically create a *minimal* fake Gemma3 hierarchy in ``sys.modules``.

    The real implementation is not required – only the class object is needed
    so that ``isinstance(model, Gemma3ForConditionalGeneration)`` evaluations
    inside the helper succeed.
    """

    # Build the nested module structure: transformers.models.gemma3.modeling_gemma3
    module_chain = [
        "transformers",
        "transformers.models",
        "transformers.models.gemma3",
        "transformers.models.gemma3.modeling_gemma3",
    ]

    parent_name = ""
    for mod_name in module_chain:
        if mod_name not in sys.modules:
            new_module = types.ModuleType(mod_name)
            sys.modules[mod_name] = new_module
        parent_name = mod_name  # keep creating deeper levels

    modeling_module = sys.modules[module_chain[-1]]

    # Create a *minimal* stub that mimics the HF class interface.
    class _FakeGemma3ForConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            # The real helper only inspects ``language_model.layers``.
            self.language_model = SimpleNamespace(layers=["layer1", "layer2"])
            # Provide an HF-style config with `tie_word_embeddings` attribute so
            # that other utilities (if imported elsewhere) do not error out.
            self.config = SimpleNamespace(tie_word_embeddings=False)

    modeling_module.Gemma3ForConditionalGeneration = _FakeGemma3ForConditionalGeneration

    # Return the freshly created class so tests can instantiate it easily.
    return _FakeGemma3ForConditionalGeneration


@pytest.fixture(autouse=True)
def fake_gemma3(monkeypatch):
    """Ensure the fake Gemma3 class is always available during this module."""
    cls = _install_fake_gemma3(monkeypatch)
    yield cls
    # Cleanup not strictly necessary – pytest runs each test in isolated proc –
    # but we remove the deepest module to avoid side-effects on other suites.
    sys.modules.pop("transformers.models.gemma3.modeling_gemma3", None)


def test_returns_layers_for_gemma3_instance(fake_gemma3):
    """When the model *is* Gemma3, the helper should return `language_model.layers`."""
    model = fake_gemma3()
    layers = get_lm_ac_layers(model)
    assert layers == model.language_model.layers  # identity check


def test_returns_layers_for_standard_hf_model(fake_gemma3):
    """Models exposing ``model.layers`` should be handled by the fallback branch."""

    class _DummyInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers: List[str] = ["block1", "block2"]

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _DummyInner()

    dummy = _DummyModel()
    layers = get_lm_ac_layers(dummy)
    assert layers == dummy.model.layers


def test_unknown_model_returns_empty_list(fake_gemma3):
    """Models that do not meet any criteria should yield an empty list."""

    class _Irrelevant(nn.Module):
        pass

    layers = get_lm_ac_layers(_Irrelevant())
    assert layers == []
