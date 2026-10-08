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

"""Compiled mHC and RMSNorm cores: opt-in (`compile_hc`, `compile_norm`), once per process, allclose to eager."""

from dataclasses import replace

import pytest
import torch
from torch._dynamo.utils import counters as dynamo_counters

import nemo_automodel.components.models.deepseek_v41.layers as v41_layers
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection, DeepseekV41RMSNorm
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.test_attention import _arithmetic_config
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config

_CORE_NAMES = ("_rms_norm", "_hc_project", "_hc_collapse", "_hc_expand", "_NORM_COMPILED", "_HC_COMPILED")


@pytest.fixture
def restore_cores():
    """Snapshot and restore the module-level dispatchers and once-per-process flags around a test."""
    saved = {name: getattr(v41_layers, name) for name in _CORE_NAMES}
    yield
    for name, value in saved.items():
        setattr(v41_layers, name, value)


def _compile_backend() -> BackendConfig:
    return replace(_backend(), compile_norm=True, compile_hc=True)


def test_compile_hc_is_a_backend_knob_that_defaults_off() -> None:
    assert BackendConfig().compile_hc is False
    assert _backend().compile_hc is False
    assert not hasattr(DeepseekV41TextConfig(), "hc_impl")  # moved out of the model config


def test_default_model_leaves_the_cores_eager(restore_cores) -> None:
    with torch.device("meta"):
        DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    assert v41_layers._hc_project is v41_layers._hc_project_core and not v41_layers._HC_COMPILED
    assert v41_layers._rms_norm is v41_layers._rms_norm_core and not v41_layers._NORM_COMPILED


def test_model_hooks_compile_the_cores_once(restore_cores) -> None:
    config = _tiny_config()
    with torch.device("meta"):
        DeepseekV41ForCausalLM(config, backend=_compile_backend())
    assert v41_layers._HC_COMPILED and v41_layers._NORM_COMPILED
    assert v41_layers._hc_project is not v41_layers._hc_project_core
    assert v41_layers._hc_expand is not v41_layers._hc_expand_core
    assert v41_layers._rms_norm is not v41_layers._rms_norm_core
    compiled = tuple(getattr(v41_layers, name) for name in _CORE_NAMES)
    with torch.device("meta"):
        DeepseekV41ForCausalLM(config, backend=_compile_backend())
    assert tuple(getattr(v41_layers, name) for name in _CORE_NAMES) == compiled  # second model: no recompile


def _hc_case(dtype: torch.dtype):
    torch.manual_seed(43)
    module = DeepseekV41HyperConnection(_arithmetic_config(hc_eps=1e-6), sinkhorn_backend="torch")
    with torch.no_grad():
        module.fn.normal_(std=0.1)
        module.base.normal_(std=0.3)
        module.scale.copy_(torch.tensor([0.9, 1.1, 0.7]))
    norm = DeepseekV41RMSNorm(16, 1e-6, dtype)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    x = torch.randn(2, 3, 4, 16, dtype=dtype, requires_grad=True)
    previous = torch.rand(2, 3, 4, requires_grad=True)
    upstream = torch.randn(2, 3, 4, 16, dtype=dtype)
    parameters = [module.fn, module.base, module.scale, norm.weight]

    def run():
        mix = module(x)
        collapsed = module.collapse(x, previous)
        normed = norm(collapsed)
        expanded = module.expand(normed, x, mix)
        loss = (expanded.float() * upstream.float()).sum() + mix.pre.square().sum() + mix.post.sum()
        grads = torch.autograd.grad(loss, [x, previous, *parameters])
        return (mix.pre, mix.post, mix.comb, collapsed, normed, expanded), grads

    return run


@pytest.mark.runtime_budget(
    30,
    reason="compiles the three mHC cores and the fp32 RMSNorm core with torch.compile for one dtype (about 10 s on CI)",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_cores_match_eager_and_reach_modules_built_earlier(restore_cores, dtype: torch.dtype) -> None:
    run = _hc_case(dtype)
    eager_out, eager_grads = run()
    # Modules were built before the hooks ran: the dispatchers must still route them to the compiled cores.
    v41_layers.compile_hc_cores()
    v41_layers.compile_norm_core()
    assert v41_layers._HC_COMPILED and v41_layers._NORM_COMPILED
    frames_before = dynamo_counters["frames"]["ok"]
    compiled_out, compiled_grads = run()
    assert dynamo_counters["frames"]["ok"] > frames_before  # the compiled path ran, not a silent eager fallback
    # fp32: fusion only reorders summations; bf16 streams may differ by one ulp (the default bf16 tolerance)
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else {}
    for actual, expected in zip(compiled_out, eager_out):
        torch.testing.assert_close(actual, expected, **tolerance)
    for actual, expected in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(actual, expected, **tolerance)


def test_hooks_are_idempotent(restore_cores) -> None:
    v41_layers.compile_hc_cores()
    first = (v41_layers._hc_project, v41_layers._hc_collapse, v41_layers._hc_expand)
    v41_layers.compile_hc_cores()
    assert (v41_layers._hc_project, v41_layers._hc_collapse, v41_layers._hc_expand) == first
    v41_layers.compile_norm_core()
    norm = v41_layers._rms_norm
    v41_layers.compile_norm_core()
    assert v41_layers._rms_norm is norm
