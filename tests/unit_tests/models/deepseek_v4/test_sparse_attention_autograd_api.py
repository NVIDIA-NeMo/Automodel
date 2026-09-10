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

"""Exercise optional autograd arguments without CUDA or TileLang kernels."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("optional_arguments", [(), (0.25,), (0.25, False), (0.25, True)])
def test_direct_apply_keeps_optional_argument_backward_arity(monkeypatch, chunked, optional_arguments):
    package = importlib.import_module("nemo_automodel.components.models.deepseek_v4.kernels")

    def forward(q, kv, sink, indices, **kwargs):
        return q.clone(), torch.zeros(q.shape[:-1])

    def backward(q, kv, sink, output, grad, indices, lse, **kwargs):
        return grad, torch.ones_like(kv), torch.ones_like(sink)

    monkeypatch.setattr(
        package, "tilelang_sparse_mla_fwd", SimpleNamespace(sparse_mqa_fwd_interface=forward), raising=False
    )
    monkeypatch.setattr(
        package, "tilelang_sparse_mla_bwd", SimpleNamespace(sparse_mqa_bwd_interface=backward), raising=False
    )
    spec = importlib.util.spec_from_file_location(
        "_sparse_attention_api_test", Path(package.__file__).with_name("sparse_attention.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operation = module.DeepSeekV4SparseAttentionHeadChunked if chunked else module.DeepSeekV4SparseAttention
    q = torch.randn(1, 3, 4, 8, requires_grad=True)
    kv = torch.randn(1, 3, 8, requires_grad=True)
    sink = torch.randn(4, requires_grad=True)
    indices = torch.zeros(1, 3, 64, dtype=torch.int32)
    arguments = (q, kv, sink, indices, 2) if chunked else (q, kv, sink, indices)
    output = operation.apply(*arguments, *optional_arguments)
    output.sum().backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q), rtol=0, atol=0)
    assert kv.grad is not None and torch.isfinite(kv.grad).all()
    torch.testing.assert_close(sink.grad, torch.ones_like(sink), rtol=0, atol=0)
