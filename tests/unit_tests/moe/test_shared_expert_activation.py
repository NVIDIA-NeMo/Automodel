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

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.layers import _clamped_swiglu_act

pytestmark = pytest.mark.timeout(600)  # one inductor graph compiles on the CPU


def _eager(gate, up, limit):
    g = gate.float().clamp(max=limit)
    u = up.float().clamp(min=-limit, max=limit)
    return (F.silu(g) * u).to(gate.dtype)


@pytest.mark.parametrize("limit", [2.0, 7.0])
def test_compiled_clamped_swiglu_matches_the_eager_chain(limit):
    torch.manual_seed(3)
    gate = (torch.randn(6, 8) * limit * 1.5).to(torch.bfloat16).requires_grad_(True)
    up = (torch.randn(6, 8) * limit * 1.5).to(torch.bfloat16).requires_grad_(True)
    grad = torch.randn(6, 8).to(torch.bfloat16)

    out = _clamped_swiglu_act(gate, up, limit)
    out.backward(grad)

    gate_e = gate.detach().clone().requires_grad_(True)
    up_e = up.detach().clone().requires_grad_(True)
    ref = _eager(gate_e, up_e, limit)
    ref.backward(grad)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(gate.grad.float(), gate_e.grad.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(up.grad.float(), up_e.grad.float(), rtol=1e-2, atol=1e-2)
    # the clamp masks: no gradient through saturated entries, some gradient elsewhere
    saturated_gate = gate.detach().float() > limit
    saturated_up = up.detach().float().abs() > limit
    assert saturated_gate.any() and saturated_up.any()
    assert torch.all(gate.grad.float()[saturated_gate] == 0)
    assert torch.all(up.grad.float()[saturated_up] == 0)
    assert torch.any(gate.grad.float()[~saturated_gate] != 0)


@pytest.mark.runtime_budget(30, hard_timeout=90, reason="Dynamo traces the island twice (static, then symbolic rows)")
def test_distinct_row_counts_recompile_once_and_never_hit_the_recompile_limit():
    """Variable-length batches give the shared expert a new token count each step. Dynamo's default shape policy
    compiles a static graph, then one symbolic graph that every later shape reuses; `dynamic=False` with
    `fullgraph=True` raised `FailOnRecompileLimitHit` after `recompile_limit` distinct shapes instead. The island's
    own function is re-wrapped with the eager backend so the test measures the shape policy, not inductor's CPU
    build (the one-GPU contract probe sweeps the production island itself)."""
    torch._dynamo.reset()
    torch.manual_seed(5)
    island = torch.compile(_clamped_swiglu_act._torchdynamo_orig_callable, fullgraph=True, backend="eager")
    limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    with torch._dynamo.config.patch(**{limit_name: 3}):
        for rows in range(4, 4 + 8):  # eight distinct shapes against a limit of three recompiles
            gate = (torch.randn(rows, 8) * 3).to(torch.bfloat16)
            up = (torch.randn(rows, 8) * 3).to(torch.bfloat16)
            out = island(gate, up, 2.0)
            torch.testing.assert_close(out, _eager(gate, up, 2.0), rtol=0, atol=0)
