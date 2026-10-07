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

"""Tests for HF wrapper selection and model-owned parallelization metadata."""

import importlib
import sys

import pytest
import torch.nn as nn

from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer, _extract_model_layer_groups


def test_adapter_attaches_layer_metadata_without_model_imports():
    class Qwen2VLForConditionalGeneration(nn.Module):
        pass

    before = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}
    result = _get_mixin_wrapped_class(Qwen2VLForConditionalGeneration)
    after = {name for name in sys.modules if name.startswith("transformers.models.") and ".modeling_" in name}

    assert issubclass(result, Qwen2VLForConditionalGeneration)
    assert "language" in result.parallelizer.layer_group_paths
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

    assert get_model_parallelizer(wrapped_class()) is (
        sidecar.VLM_PARALLELIZER if architecture == "Qwen3_5ForConditionalGeneration" else sidecar.PARALLELIZER
    )
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_hf_only_sidecar_selects_decoder_instead_of_larger_unrelated_container(wrapped):
    upstream = type("GPT2LMHeadModel", (nn.Module,), {})
    model_class = _get_mixin_wrapped_class(upstream) if wrapped else upstream
    model = model_class()
    model.transformer = nn.Module()
    model.transformer.h = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
    model.unrelated = nn.ModuleList([nn.Linear(2, 2) for _ in range(8)])

    assert _extract_model_layer_groups(model) == {"language": list(model.transformer.h)}
    assert not hasattr(upstream, "parallelizer")


def test_existing_layer_override_wins_over_registered_sidecar_layout():
    upstream = type(
        "Qwen2VLForConditionalGeneration",
        (nn.Module,),
        {"parallel_layer_groups": {"language": ("decoder.layers",)}},
    )
    model = _get_mixin_wrapped_class(upstream)()
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList([nn.Linear(2, 2)])

    assert _extract_model_layer_groups(model) == {"language": list(model.decoder.layers)}


@pytest.mark.runtime_budget(
    20,
    hard_timeout=40,
    reason="A fresh Python process must import torch and every sidecar to detect eager native-model imports.",
)
def test_all_sidecars_import_without_concrete_model_implementations():
    # A fresh process catches package __init__ imports hidden by pytest collection.
    import subprocess
    import textwrap

    program = textwrap.dedent("""
        import sys
        from nemo_automodel.components.models.parallelization import MODEL_PARALLELIZERS, _load_reference

        def implementations():
            return {
                name for name in sys.modules
                if (name.startswith("nemo_automodel.components.models.") and name.endswith(".model"))
                or (name.startswith("transformers.models.") and ".modeling_" in name)
            }

        before = implementations()
        for reference in sorted(set(MODEL_PARALLELIZERS.values())):
            _load_reference(reference)
        assert implementations() == before, implementations() - before
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("package", "name", "module", "target"),
    [
        ("qwen2", "Qwen2ForCausalLM", "model", "Qwen2ForCausalLM"),
        ("qwen3", "Qwen3ForCausalLM", "model", "Qwen3ForCausalLM"),
        ("bagel", "BagelForUnifiedMultimodal", "model", "BagelForUnifiedMultimodal"),
        ("minimax_m3_vl", "ModelClass", "model", "MiniMaxM3SparseForConditionalGeneration"),
        ("step3p7", "Step3p7Config", "configuration_step3p7", "Step3p7Config"),
        ("llama_nemotron_vl", "LlamaNemotronVLModel", "model", "LlamaNemotronVLModel"),
        ("ministral_bidirectional", "Ministral3BidirectionalModel", "model", "Ministral3BidirectionalModel"),
        ("mistral3_vlm", "Mistral3FP8VLMForConditionalGeneration", "model", "Mistral3FP8VLMForConditionalGeneration"),
    ],
)
def test_lazy_package_preserves_public_exports(package, name, module, target):
    prefix = f"nemo_automodel.components.models.{package}"
    package_module = importlib.import_module(prefix)
    implementation = importlib.import_module(f"{prefix}.{module}")
    assert getattr(package_module, name) is getattr(implementation, target)
