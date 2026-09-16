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

"""CPU coverage of optional CuTe dispatch and tensor validation."""

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next import cute_qsa
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    gathered_qsa_gqa_attention,
    qsa_gqa_attention,
)


def test_cpu_cute_dispatch_keeps_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(4)
    inputs = [torch.randn(1, 7, heads, 4, requires_grad=True) for heads in (4, 2, 2)]
    routes = torch.arange(7).view(1, 1, 7).expand(1, 7, 7)
    monkeypatch.setattr(cute_qsa, "safe_import", lambda *args: pytest.fail("CPU must not load CuTe"))
    actual = qsa_gqa_attention(*inputs, routes, backend="cute")
    expected = gathered_qsa_gqa_attention(*inputs, routes)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, inputs, dy)
    refs = torch.autograd.grad(expected, inputs, dy)
    for result, reference in zip((actual, *grads), (expected, *refs)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cute_qsa._load_kernels.cache_clear()
    monkeypatch.setattr(cute_qsa, "safe_import", lambda name: (False, None))
    with pytest.raises(ImportError, match="FlashAttention.*SM90"):
        cute_qsa._load_kernels()
    cute_qsa._load_kernels.cache_clear()


@pytest.mark.parametrize(
    "case", ["rank", "batch", "kv_shape", "head_dim", "gqa", "empty", "routes", "route_dtype", "dtype", "cpu"]
)
def test_cute_contract_rejects_invalid_inputs(case: str) -> None:
    q = torch.empty(1, 3, 4, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    routes = torch.zeros(1, 3, 4, dtype=torch.int64)
    error, match = ValueError, "same CUDA device"
    if case == "rank":
        q = q.squeeze(0)
        match = "layout"
    elif case == "batch":
        k = k.expand(2, -1, -1, -1)
        v = v.expand_as(k)
        match = "batch"
    elif case == "kv_shape":
        v = v[:, :3]
        match = "matching K/V"
    elif case == "head_dim":
        q = q[..., :128]
        match = "head_dim"
    elif case == "gqa":
        q = q[:, :, :3]
        match = "divisible"
    elif case == "empty":
        q = q[:, :0]
        match = "nonempty"
    elif case == "routes":
        routes = routes[:, :2]
        match = "routes must have shape"
    elif case == "route_dtype":
        routes = routes.float()
        error, match = TypeError, "signed"
    elif case == "dtype":
        q = q.float()
        error, match = TypeError, "BF16"
    with pytest.raises(error, match=match):
        cute_qsa.cute_sparse_gqa_attention(q, k, v, routes)
