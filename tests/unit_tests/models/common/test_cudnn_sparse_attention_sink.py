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
"""The shared cuDNN sparse attention carries DeepSeek-V4.1's learnable sink and can run without FlashMLA."""

from __future__ import annotations

import pytest
import torch

import nemo_automodel.components.models.common.cudnn_sparse_attention as shared_cudnn

TOKENS, HEADS, DIM, WIDTH = 3, 64, 512, 2


def _accept_cpu_tensors(operation: str, *tensors: torch.Tensor) -> tuple[int, int]:
    assert tensors
    return 10, 0


class _FakeDsa:
    """cuDNN >= 1.29 namespace: forward and backward wrappers with the real tensor contract."""

    def __init__(self) -> None:
        self.forward_sink: torch.Tensor | None = None
        self.backward_sink: torch.Tensor | None = None

    def sparse_attention_forward_wrapper(self, q, kv, topk_idxs, *, attn_sink, topk_length, softmax_scale):
        assert q.shape == (TOKENS, HEADS, DIM) and q.dtype == torch.bfloat16
        assert kv.shape[1] == DIM and kv.ndim == 2
        assert topk_idxs.ndim == 2 and topk_idxs.dtype == torch.int32
        assert attn_sink.shape == (HEADS,) and attn_sink.dtype == torch.float32
        assert softmax_scale == 0.125
        self.forward_sink = attn_sink.clone()
        out = torch.full((TOKENS, HEADS, 512), 7.0, dtype=torch.bfloat16)
        return {"out": out, "max_logits": torch.zeros(TOKENS, HEADS), "lse": torch.ones(TOKENS, HEADS)}

    def sparse_attention_backward_wrapper(
        self, q, kv, out, d_out, lse, attn_sink, indices, *, softmax_scale, topk_length
    ):
        self.backward_sink = attn_sink.clone()
        return {
            "dq": torch.full_like(q, 3.0),
            "dkv": torch.full_like(kv, 5.0),
            "d_sink": torch.arange(HEADS, dtype=torch.float32),
        }


def _inputs():
    q = torch.ones(TOKENS, HEADS, DIM, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.ones(TOKENS, 1, DIM, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.arange(WIDTH, dtype=torch.int32).view(1, 1, -1).expand(TOKENS, -1, -1).contiguous()
    return q, kv, indices


@pytest.fixture
def fake_dsa(monkeypatch: pytest.MonkeyPatch) -> _FakeDsa:
    dsa = _FakeDsa()
    monkeypatch.setattr(shared_cudnn, "_HAS_CUDNN_DSA", True)
    monkeypatch.setattr(shared_cudnn, "_HAS_FLASH_MLA", False)  # no FlashMLA: the cuDNN forward must carry the run
    monkeypatch.setattr(shared_cudnn, "_CUDNN_DSA", dsa)
    monkeypatch.setattr(shared_cudnn, "_require_cuda_tensors", _accept_cpu_tensors)
    return dsa


def test_availability_accepts_the_cudnn_forward_without_flash_mla(
    fake_dsa: _FakeDsa, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert shared_cudnn.is_cudnn_sparse_attention_available()
    monkeypatch.setattr(shared_cudnn, "_CUDNN_DSA", object())  # an older frontend without the forward wrapper
    assert not shared_cudnn.is_cudnn_sparse_attention_available()
    monkeypatch.setattr(shared_cudnn, "_HAS_FLASH_MLA", True)
    assert shared_cudnn.is_cudnn_sparse_attention_available()


def test_sink_reaches_both_kernels_and_receives_its_gradient(fake_dsa: _FakeDsa) -> None:
    q, kv, indices = _inputs()
    sink = torch.linspace(-2.0, 2.0, HEADS, dtype=torch.float32, requires_grad=True)
    out = shared_cudnn.cudnn_sparse_attention(
        q, kv, indices, softmax_scale=0.125, all_rows_nonempty=True, attn_sink=sink
    )
    assert out.shape == (TOKENS, HEADS, 512)
    torch.testing.assert_close(fake_dsa.forward_sink, sink.detach())
    out.sum().backward()
    torch.testing.assert_close(fake_dsa.backward_sink, sink.detach())
    torch.testing.assert_close(sink.grad, torch.arange(HEADS, dtype=torch.float32))
    torch.testing.assert_close(q.grad, torch.full_like(q, 3.0))
    torch.testing.assert_close(kv.grad, torch.full_like(kv, 5.0))


def test_without_a_sink_the_kernels_see_minus_inf_and_nothing_else_changes(fake_dsa: _FakeDsa) -> None:
    q, kv, indices = _inputs()
    out = shared_cudnn.cudnn_sparse_attention(q, kv, indices, softmax_scale=0.125, all_rows_nonempty=True)
    assert torch.isneginf(fake_dsa.forward_sink).all()
    out.sum().backward()
    assert torch.isneginf(fake_dsa.backward_sink).all()
    assert q.grad is not None and kv.grad is not None


def test_sink_shape_and_dtype_are_validated(fake_dsa: _FakeDsa) -> None:
    q, kv, indices = _inputs()
    with pytest.raises(ValueError, match="attn_sink must be an FP32 tensor"):
        shared_cudnn.cudnn_sparse_attention(
            q,
            kv,
            indices,
            softmax_scale=0.125,
            all_rows_nonempty=True,
            attn_sink=torch.zeros(HEADS, dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="attn_sink must be an FP32 tensor"):
        shared_cudnn.cudnn_sparse_attention(
            q, kv, indices, softmax_scale=0.125, all_rows_nonempty=True, attn_sink=torch.zeros(HEADS + 1)
        )
