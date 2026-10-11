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

"""Rotary precision, partial-subspace and packed-position mathematical controls."""

import json
from collections.abc import Callable

import pytest
import torch

from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


@pytest.mark.parametrize("span", [0, 32, 64, 192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_complex_oracle_and_packed_resets(span: int, dtype: torch.dtype) -> None:
    torch.manual_seed(984)
    config = Step5Config()
    config.rope_theta = 10000.0
    positions = torch.tensor([[0, 1, 8191, 131071, 0, 11], [7, 0, 31, 0, 65535, 128001]])
    hidden = torch.randn(2, 6, 3, 192, dtype=dtype, requires_grad=True)
    actual = Step5RotaryEmbedding(config, 0, span=span)(hidden, positions)
    if span == 0:
        assert actual is hidden
        return
    # Complex multiplication is independent of the production half-split formula.
    # The checkpoint rounds the positive power before reciprocal. Negative
    # powers are mathematically equal but change FP32 phases at long positions.
    inverse = 1.0 / (10000.0 ** (torch.arange(0, span, 2, dtype=torch.float32) / span))
    phases = positions.float().unsqueeze(-1) * inverse
    table_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    table = torch.complex(phases.cos().to(table_dtype).double(), phases.sin().to(table_dtype).double())
    half = span // 2
    values = torch.complex(hidden[..., :half].double(), hidden[..., half:span].double())
    rotated = values * table.unsqueeze(-2)
    expected = torch.cat((rotated.real, rotated.imag, hidden[..., span:].double()), -1).to(dtype)
    tolerance = 0.03125 if dtype == torch.bfloat16 else 5e-7
    torch.testing.assert_close(actual, expected, rtol=0.003 if dtype == torch.bfloat16 else 5e-7, atol=tolerance)
    assert torch.equal(actual[..., span:], hidden[..., span:])
    if span:
        assert torch.equal(actual[0, 0], hidden[0, 0])
        assert torch.equal(actual[0, 4], hidden[0, 4])
    upstream = torch.randn_like(hidden)
    observed_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    torch.testing.assert_close(observed_grad, expected_grad, rtol=0.005, atol=tolerance)


def test_mixed_rotary_shapes_share_bounded_compile_entries(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, str], None]
) -> None:
    """Count Dynamo shape guards on metadata tensors without requiring CUDA.

    No reset occurs between cases. This catches compiler-cache exhaustion but
    does not replace the matching functional test using actual CUDA kernels.
    """
    from nemo_automodel.components.models.step5 import rope as implementation

    counts = {"full": 0, "partial": 0, "single_head": 0}

    def full_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["full"] += 1
        return graph.forward

    def partial_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["partial"] += 1
        return graph.forward

    def single_head_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["single_head"] += 1
        return graph.forward

    for name, backend in [("full", full_backend), ("partial", partial_backend), ("single_head", single_head_backend)]:
        attribute = f"_apply_{name}_rotary_compiled"
        original = getattr(implementation, attribute)._torchdynamo_orig_callable
        monkeypatch.setattr(
            implementation, attribute, torch.compile(original, backend=backend, fullgraph=True, dynamic=True)
        )
    for sequence in [4096, 8192, 0, 1]:
        for span, heads, dim in [
            (64, 64, 192),
            (64, 4, 192),
            (192, 64, 192),
            (192, 4, 192),
            (32, 16, 256),
            (32, 1, 256),
        ]:
            for grad in [True, False]:
                hidden = torch.empty(1, sequence, heads, dim, device="meta", dtype=torch.bfloat16, requires_grad=grad)
                cos = torch.empty(1, sequence, 1, span // 2, device="meta", dtype=torch.float32)
                with torch.set_grad_enabled(grad):
                    output = implementation._apply_rotary_compiled(hidden, cos, cos, span)
                assert output.shape == hidden.shape
                if sequence == 0:
                    assert output is hidden
    assert 0 < counts["full"] <= 8
    assert 0 < counts["partial"] <= 8
    assert 0 < counts["single_head"] <= 8
    record_property("compile_entries", json.dumps(counts))
