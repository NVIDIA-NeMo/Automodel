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

"""Architecture contracts for model-owned parallelization strategies."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from nemo_automodel._transformers.model_init import _attach_parallelization_strategy, _parallelizer_module_path
from nemo_automodel.components.distributed import parallelizer


@pytest.mark.parametrize(
    ("module_name", "strategy_class_name"),
    [
        (
            "nemo_automodel.components.models.nemotron_v3.parallelizer",
            "NemotronHParallelizationStrategy",
        ),
        (
            "nemo_automodel.components.models.qwen3_5.parallelizer",
            "Qwen3_5ParallelizationStrategy",
        ),
        (
            "nemo_automodel.components.models.deepseek_v4.parallelizer",
            "DeepseekV4ParallelizationStrategy",
        ),
        (
            "nemo_automodel.components.models.diffusion_gemma.parallelizer",
            "DiffusionGemmaParallelizationStrategy",
        ),
        (
            "nemo_automodel._diffusers.wan.parallelizer",
            "WanParallelizationStrategy",
        ),
        (
            "nemo_automodel._diffusers.hunyuan_video.parallelizer",
            "HunyuanParallelizationStrategy",
        ),
    ],
)
def test_model_sidecars_return_strategy_without_mutating_global_registry(
    module_name: str, strategy_class_name: str
) -> None:
    """A sidecar can provide its strategy without global registration state."""
    original_registry = dict(parallelizer.PARALLELIZATION_STRATEGIES)
    sidecar = importlib.import_module(module_name)

    strategy = sidecar.get_parallelization_strategy()
    model = SimpleNamespace(_nemo_parallelization_strategy_factory=sidecar.get_parallelization_strategy)

    assert isinstance(strategy, getattr(sidecar, strategy_class_name))
    assert parallelizer.get_parallelization_strategy(model) is strategy
    assert parallelizer.PARALLELIZATION_STRATEGIES == original_registry


def test_parallelizer_has_no_direct_model_package_import() -> None:
    """The generic runtime must not import a concrete model package."""
    tree = ast.parse(Path(parallelizer.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    assert not any(name.startswith("nemo_automodel.components.models") for name in imports)


def test_parallelizer_contains_only_generic_strategy_types() -> None:
    """Concrete model policies must live beside the model that owns them."""
    tree = ast.parse(Path(parallelizer.__file__).read_text(encoding="utf-8"))
    strategy_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("ParallelizationStrategy")
    }

    assert strategy_classes == {"ParallelizationStrategy", "DefaultParallelizationStrategy"}


def test_native_model_import_defers_sidecar_loading_in_a_clean_process() -> None:
    """A native model loads its sidecar only when FSDP2 requests the factory."""
    script = """
import sys
from nemo_automodel.components.distributed.parallelizer import PARALLELIZATION_STRATEGIES

assert "NemotronHForCausalLM" not in PARALLELIZATION_STRATEGIES
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
assert "nemo_automodel.components.models.nemotron_v3.parallelizer" not in sys.modules
strategy = NemotronHForCausalLM._nemo_parallelization_strategy_factory()
assert strategy.__class__.__name__ == "NemotronHParallelizationStrategy"
assert "nemo_automodel.components.models.nemotron_v3.parallelizer" in sys.modules
assert "NemotronHForCausalLM" not in PARALLELIZATION_STRATEGIES
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_diffusers_adapter_defers_sidecar_loading_in_a_clean_process() -> None:
    """The Diffusers adapter loads strategy code only for a matching transformer."""
    script = """
import sys
from torch import nn
from nemo_automodel._diffusers.auto_diffusion_pipeline import _attach_diffusion_parallelization_strategy

sidecar = "nemo_automodel._diffusers.parallelizer"
wan_sidecar = "nemo_automodel._diffusers.wan.parallelizer"
hunyuan_sidecar = "nemo_automodel._diffusers.hunyuan_video.parallelizer"
assert sidecar not in sys.modules
assert wan_sidecar not in sys.modules
assert hunyuan_sidecar not in sys.modules
WanTransformer3DModel = type("WanTransformer3DModel", (nn.Module,), {})
model = WanTransformer3DModel()
_attach_diffusion_parallelization_strategy(model)
assert sidecar in sys.modules
assert wan_sidecar in sys.modules
assert hunyuan_sidecar not in sys.modules
assert model._nemo_parallelization_strategy.__class__.__name__ == "WanParallelizationStrategy"
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_adapter_attaches_a_lazy_factory_without_eager_import() -> None:
    """Adapter-owned policy remains lazy for an upstream class sharing an architecture name."""
    script = """
import sys
from types import SimpleNamespace
from nemo_automodel._transformers.model_init import _attach_parallelization_strategy

model = SimpleNamespace()
_attach_parallelization_strategy(model, ["NemotronHForCausalLM"])
assert "nemo_automodel.components.models.nemotron_v3.parallelizer" not in sys.modules
strategy = model._nemo_parallelization_strategy_factory()
assert strategy.__class__.__name__ == "NemotronHParallelizationStrategy"

unrelated_model = SimpleNamespace()
_attach_parallelization_strategy(unrelated_model, ["LlamaForCausalLM"])
assert not hasattr(unrelated_model, "_nemo_parallelization_strategy_factory")
"""
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_adapter_derives_tp_sidecar_from_registry_or_model_type() -> None:
    """TP sidecars follow the owning model directory without a plan registry."""
    registered_model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    upstream_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))

    assert _parallelizer_module_path(registered_model, "LlamaForCausalLM") == (
        "nemo_automodel.components.models.llama.parallelizer"
    )
    assert _parallelizer_module_path(upstream_model, "Qwen3ForCausalLM") == (
        "nemo_automodel.components.models.qwen3.parallelizer"
    )
    assert _parallelizer_module_path(
        SimpleNamespace(config=SimpleNamespace(model_type="mixtral")), "MixtralForCausalLM"
    ) == ("nemo_automodel.components.models.mixtral.parallelizer")


def test_model_adapter_attaches_tp_factory_from_the_conventional_sidecar() -> None:
    """Loading an upstream model attaches only its model-directory sidecar."""
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))

    _attach_parallelization_strategy(model, ["Qwen3ForCausalLM"])

    assert callable(model._nemo_tp_plan_factory)


def test_model_adapter_attaches_mixtral_legacy_plan_by_model_type() -> None:
    """An upstream Mixtral model gets its plan from the Mixtral sidecar."""
    model = SimpleNamespace(config=SimpleNamespace(model_type="mixtral"))

    _attach_parallelization_strategy(model, ["MixtralForCausalLM"])

    plan = model._nemo_tp_plan_factory(model, sequence_parallel=True)
    assert "model.layers.*.self_attn.qkv_proj" in plan
    assert "model.layers.*.input_layernorm" in plan


def test_deepseek_strategy_owns_its_custom_fsdp_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek selects its FSDP implementation through its strategy hook."""
    from nemo_automodel.components.models.deepseek_v4 import fsdp as deepseek_fsdp
    from nemo_automodel.components.models.deepseek_v4.parallelizer import get_parallelization_strategy

    calls: list[tuple[nn.Module, dict[str, object]]] = []

    def custom_fully_shard(module: nn.Module, **kwargs: object) -> nn.Module:
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(deepseek_fsdp, "fully_shard_deepseek_v4", custom_fully_shard)

    module = nn.Linear(2, 2)
    mesh = object()
    result = get_parallelization_strategy()._shard_module(
        module,
        mesh=mesh,
        mp_policy=None,
        offload_policy=None,
        reshard_after_forward=False,
    )

    assert result is module
    assert calls == [
        (
            module,
            {
                "mesh": mesh,
                "mp_policy": None,
                "offload_policy": None,
                "reshard_after_forward": False,
            },
        )
    ]
