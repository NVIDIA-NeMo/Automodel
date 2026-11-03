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

"""``te_fp8.exclude_fqns`` keeps real Transformer Engine modules out of the FP8 autocast."""

import types
from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.common.utils import HAVE_TE, TEFp8Config


def _fp8_capable() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


pytestmark = pytest.mark.skipif(not HAVE_TE or not _fp8_capable(), reason="requires Transformer Engine and an FP8 GPU")


def _two_te_linears() -> nn.Sequential:
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(
            body=te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16),
            lm_head=te.Linear(256, 512, bias=False, params_dtype=torch.bfloat16),
        )
    ).to("cuda")


def test_excluded_te_linear_runs_in_the_model_dtype_inside_the_fp8_autocast():
    model = _two_te_linears()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    cfg = TEFp8Config(recipe="current")
    assert cfg.exclude_modules(model) == ["lm_head"]

    with torch.no_grad():
        hidden = model.body(x)  # plain bf16 GEMM, outside any FP8 scope
        reference_head = F.linear(hidden, model.lm_head.weight)
        with cfg.maybe_te_autocast():
            fp8_body = model.body(x)
            head_inside_autocast = model.lm_head(hidden)

    # The module left in FP8 really ran in FP8 (its output deviates from the bf16 GEMM) ...
    assert not torch.equal(fp8_body, hidden)
    # ... while the excluded module produced the bf16 GEMM result inside the same autocast.
    torch.testing.assert_close(head_inside_autocast, reference_head)
    assert model.lm_head.fp8 is False and model.body.fp8 is True

    # Gradients flow through the wrapped forward as through the plain one.
    with cfg.maybe_te_autocast():
        model(x).float().sum().backward()
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()


def test_excluded_te_linear_keeps_an_installed_instance_forward():
    """A forward installed before the exclusion, as tensor-parallel styles do, still runs and runs outside FP8."""

    def doubled_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return type(module).forward(module, x) * 2

    model = _two_te_linears()
    model.lm_head.forward = types.MethodType(doubled_forward, model.lm_head)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    cfg = TEFp8Config(recipe="current")
    assert cfg.exclude_modules(model) == ["lm_head"]

    with torch.no_grad():
        reference = 2 * F.linear(x, model.lm_head.weight)
        with cfg.maybe_te_autocast():
            out = model.lm_head(x)
    torch.testing.assert_close(out, reference)
    assert model.lm_head.fp8 is False
    assert model.lm_head._te_fp8_excluded_forward.__func__ is doubled_forward
