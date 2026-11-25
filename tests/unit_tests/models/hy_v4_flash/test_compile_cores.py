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

"""Compiled RMSNorm and hyper-connection cores (``compile_norm``, ``compile_hc``): what the recipes run."""

from collections.abc import Iterator

import pytest
import torch
from torch._dynamo.utils import counters as dynamo_counters

import nemo_automodel.components.models.hy_v4_flash.layers as layers
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4_flash.layers import HY4FlashHyperConnection, HY4FlashRMSNorm
from tests.unit_tests.models.hy_v4_flash.tiny import tiny_text_config

_CORE_NAMES = ("_rms_norm", "_hc_collapse", "_hc_expand", "_NORM_COMPILED", "_HC_COMPILED")


@pytest.fixture
def restore_cores() -> Iterator[None]:
    """Snapshot and restore the module-level dispatchers and once-per-process flags around a test."""
    saved = {name: getattr(layers, name) for name in _CORE_NAMES}
    yield
    for name, value in saved.items():
        setattr(layers, name, value)


def _bf16_spacing(*values: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 values at the magnitude of the largest argument: 2^(e - 7) for values in [2^e, 2^(e + 1))."""
    magnitude = torch.stack([value.float().abs() for value in values]).amax(0).clamp_min(1e-30)
    return torch.pow(2.0, torch.floor(torch.log2(magnitude)) - 7)


def test_knobs_default_off() -> None:
    assert BackendConfig().compile_norm is False and BackendConfig().compile_hc is False


def test_hooks_are_idempotent(restore_cores: None) -> None:
    layers.compile_hc_cores()
    first = (layers._hc_collapse, layers._hc_expand)
    layers.compile_hc_cores()
    assert (layers._hc_collapse, layers._hc_expand) == first
    layers.compile_norm_core()
    norm = layers._rms_norm
    layers.compile_norm_core()
    assert layers._rms_norm is norm


@pytest.mark.runtime_budget(
    30, reason="compiles the RMSNorm core and the two hyper-connection cores with torch.compile for one dtype"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_cores_reach_modules_built_earlier(restore_cores: None, dtype: torch.dtype) -> None:
    """The hooks the recipes call swap the cores under modules that already exist.

    fp32: compiled equals eager to rounding. BF16: the compiled RMSNorm multiplies by the weight in fp32 and rounds
    once, where the eager core (the reference order) rounds to BF16 first; the two differ by at most one unit in the
    last place, and so do the hyper-connection outputs.
    """
    torch.manual_seed(0)
    config = tiny_text_config()
    connection = HY4FlashHyperConnection(config)
    connection.init_weights(config.initializer_range)
    norm = HY4FlashRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * torch.randn(config.hidden_size)).to(dtype))
    streams = torch.randn(2, 9, config.hc_mult, config.hidden_size).to(dtype)

    def run(
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each core on fixed inputs: with ``inputs`` the norm and the expand consume those instead of the chain."""
        post, collapsed = connection(streams)
        norm_input, expand_post, expand_output = inputs or (collapsed, post, norm(collapsed))
        normed = norm(norm_input)
        return post, collapsed, normed, connection.expand(expand_post, expand_output, streams)

    eager = run()
    layers.compile_hc_cores()
    layers.compile_norm_core()
    assert layers._HC_COMPILED and layers._NORM_COMPILED
    frames_before = dynamo_counters["frames"]["ok"]
    compiled = run((eager[1], eager[0], eager[2]))  # every core sees exactly the eager run's inputs
    assert dynamo_counters["frames"]["ok"] > frames_before  # the compiled path ran, not a silent eager fallback

    if dtype == torch.float32:
        for actual, expected in zip(compiled, eager):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return
    torch.testing.assert_close(compiled[0], eager[0], rtol=1e-5, atol=1e-6)  # ``post`` is fp32 in either dtype
    for actual, expected in zip(compiled[1:], eager[1:]):
        assert actual.dtype == torch.bfloat16
        assert bool(((actual.float() - expected.float()).abs() <= _bf16_spacing(actual, expected)).all())
    # The single rounding: on the eager collapse output, the compiled norm is the fp32 product rounded once.
    collapsed = eager[1]
    normalized = collapsed.float() * torch.rsqrt(
        collapsed.float().square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    once = (norm.weight.float() * normalized).to(torch.bfloat16)
    assert float((norm(collapsed) == once).float().mean()) > 0.99
