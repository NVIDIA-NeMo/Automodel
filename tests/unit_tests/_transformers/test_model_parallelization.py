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

"""Tests for adapter-owned parallelization metadata."""

import importlib
import sys

import pytest
import torch.nn as nn

from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel._transformers.model_parallelization import configure_parallelization_metadata
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer, _extract_model_layer_groups


def test_adapter_attaches_layer_metadata_without_model_imports():
    class Qwen2VLForConditionalGeneration(nn.Module):
        pass

    before = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}
    result = configure_parallelization_metadata(Qwen2VLForConditionalGeneration)
    after = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}

    assert result is Qwen2VLForConditionalGeneration
    assert "language" in result.parallel_layer_groups
    assert after == before


def test_adapter_metadata_drives_generic_layer_extraction():
    class Model(nn.Module):
        parallel_layer_groups = {"language": ("decoder.layers",)}

        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    model = Model()
    assert _extract_model_layer_groups(model) == {"language": list(model.decoder.layers)}


@pytest.mark.parametrize(
    ("architecture", "model_package"),
    [
        ("NemotronHForCausalLM", "nemotron_v3"),
        ("DeepseekV4ForCausalLM", "deepseek_v4"),
        ("Qwen3_5ForCausalLM", "qwen3_5"),
        ("Qwen3_5ForConditionalGeneration", "qwen3_5"),
    ],
)
def test_hf_wrapper_restores_specialized_parallelizer(architecture, model_package):
    # Remote model classes retain their snapshot-specific module path when wrapped.
    upstream_class = type(architecture, (nn.Module,), {"__module__": "transformers_modules.repo.snapshot.modeling"})
    wrapped_class = _get_mixin_wrapped_class(upstream_class)
    sidecar = importlib.import_module(f"nemo_automodel.components.models.{model_package}.parallelization")

    assert get_model_parallelizer(wrapped_class()) is sidecar.PARALLELIZER
    assert not hasattr(upstream_class, "parallelizer")


@pytest.mark.parametrize("inherited", [False, True])
def test_hf_wrapper_preserves_explicit_parallelizer(inherited):
    custom_parallelizer = ModelParallelizer()
    parent = type("Parent", (nn.Module,), {"parallelizer": custom_parallelizer} if inherited else {})
    upstream_class = type("NemotronHForCausalLM", (parent,), {} if inherited else {"parallelizer": custom_parallelizer})

    wrapped_class = _get_mixin_wrapped_class(upstream_class)

    assert get_model_parallelizer(wrapped_class()) is custom_parallelizer
    assert upstream_class.parallelizer is custom_parallelizer


@pytest.mark.parametrize(
    "architecture", ["UnregisteredModel", "LlamaForCausalLM", "Qwen3ForCausalLM", "Qwen3VLForConditionalGeneration"]
)
def test_hf_wrapper_keeps_default_without_compatible_sidecar(architecture, monkeypatch):
    def unexpected_import(name, package=None):
        pytest.fail(f"HF model without a compatible sidecar must not import {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    upstream_class = type(architecture, (nn.Module,), {})
    wrapped_class = _get_mixin_wrapped_class(upstream_class)

    assert type(get_model_parallelizer(wrapped_class())) is ModelParallelizer
    assert not hasattr(upstream_class, "parallelizer")


def test_hf_sidecar_does_not_import_native_model(monkeypatch):
    real_import = importlib.import_module

    def import_without_native_model(name, package=None):
        assert name != "nemo_automodel.components.models.nemotron_v3.model"
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_native_model)
    upstream_class = type("NemotronHForCausalLM", (nn.Module,), {})
    wrapped_class = _get_mixin_wrapped_class(upstream_class)

    assert type(get_model_parallelizer(wrapped_class())).__name__ == "NemotronHModelParallelizer"
