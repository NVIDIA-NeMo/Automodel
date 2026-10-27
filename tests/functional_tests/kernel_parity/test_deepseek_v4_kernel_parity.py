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

"""DeepSeek V4 CUDA kernel forward/backward parity with the Torch references."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    build_dsv4_sparse_topk_indices,
    dsv4_indexer_scores,
    dsv4_indexer_topk_scores,
    dsv4_sinkhorn_normalize,
    dsv4_sparse_attention,
    is_dsv4_kernel_available,
)
from tests.unit_tests.models.deepseek_v4.test_dsv4_layers import (
    _can_run_tilelang_indexer,
    _can_run_tilelang_sparse_attn,
    _run_forward_backward,
)
from tests.unit_tests.models.deepseek_v4.test_dsv4_vision import (
    _can_run_tilelang_sparse_attention,
    check_visual_sparse_attention_module_matches_dense_forward_backward,
)

pytestmark = pytest.mark.timeout(300)


@pytest.mark.skipif(
    not is_dsv4_kernel_available("sinkhorn") or not torch.cuda.is_available(),
    reason="TileKernels sinkhorn kernel is not installed on a CUDA environment",
)
def test_sinkhorn_tilelang_backend_matches_torch() -> None:
    torch.manual_seed(123)
    x = torch.randn(2, 128, 4, 4, device="cuda")
    grad = torch.randn_like(x)
    expected, (expected_grad,) = _run_forward_backward(
        lambda x_: dsv4_sinkhorn_normalize(x_, backend="torch", repeat=5, eps=1e-6),
        (x,),
        grad,
    )
    actual, (actual_grad,) = _run_forward_backward(
        lambda x_: dsv4_sinkhorn_normalize(x_, backend="tilelang", repeat=5, eps=1e-6),
        (x,),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not is_dsv4_kernel_available("sinkhorn") or not torch.cuda.is_available(),
    reason="TileKernels sinkhorn kernel is not installed on a CUDA environment",
)
def test_sinkhorn_tilelang_backend_accepts_non_contiguous_grad() -> None:
    torch.manual_seed(123)
    x = torch.randn(2, 128, 4, 4, device="cuda")
    grad = torch.randn_like(x).transpose(-1, -2)
    assert not grad.is_contiguous()
    expected, (expected_grad,) = _run_forward_backward(
        lambda x_: dsv4_sinkhorn_normalize(x_, backend="torch", repeat=5, eps=1e-6),
        (x,),
        grad,
    )
    actual, (actual_grad,) = _run_forward_backward(
        lambda x_: dsv4_sinkhorn_normalize(x_, backend="tilelang", repeat=5, eps=1e-6),
        (x,),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not _can_run_tilelang_sparse_attn(),
    reason="Vendored Miles DSV4 sparse-attention kernel is not available on a CUDA environment",
)
def test_sparse_attention_tilelang_backend_matches_torch() -> None:
    torch.manual_seed(123)
    bsz, seq, heads, dim, key_len, topk_len = 1, 16, 8, 128, 20, 16
    q = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(bsz, key_len, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda")
    topk = torch.stack(
        [torch.stack([torch.randperm(key_len, device="cuda")[:topk_len] for _ in range(seq)]) for _ in range(bsz)]
    ).to(torch.int32)
    topk[:, :, -4:] = -1
    grad = torch.randn(bsz, seq, dim, heads, device="cuda", dtype=q.dtype).transpose(2, 3)
    assert not grad.is_contiguous()

    expected, expected_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(
            q_,
            kv_,
            sinks_,
            topk,
            dim**-0.5,
            backend="sparse_torch",
        ),
        (q, kv, sinks),
        grad,
    )
    actual, actual_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(
            q_,
            kv_,
            sinks_,
            topk,
            dim**-0.5,
            backend="tilelang",
        ),
        (q, kv, sinks),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not _can_run_tilelang_sparse_attn(),
    reason="Vendored Miles DSV4 sparse-attention kernel is not available on a CUDA environment",
)
def test_sparse_attention_tilelang_backend_matches_torch_at_sixty_four_heads() -> None:
    """The single 64-head backward kernel (512 threads) matches the torch reference at the V4 head shape."""
    torch.manual_seed(7)
    bsz, seq, heads, dim, key_len, topk_len = 1, 8, 64, 512, 64, 32
    q = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(bsz, key_len, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda")
    topk = torch.stack(
        [torch.stack([torch.randperm(key_len, device="cuda")[:topk_len] for _ in range(seq)]) for _ in range(bsz)]
    ).to(torch.int32)
    topk[:, :, -4:] = -1
    grad = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=q.dtype)

    expected, expected_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(q_, kv_, sinks_, topk, dim**-0.5, backend="sparse_torch"),
        (q, kv, sinks),
        grad,
    )
    actual, actual_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(q_, kv_, sinks_, topk, dim**-0.5, backend="tilelang"),
        (q, kv, sinks),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not _can_run_tilelang_sparse_attn(),
    reason="Vendored Miles DSV4 sparse-attention kernel is not available on a CUDA environment",
)
def test_sparse_attention_tilelang_backend_matches_torch_with_causal_padding_shape() -> None:
    torch.manual_seed(123)
    bsz, seq, heads, dim, n_pooled = 1, 192, 128, 128, 48
    q = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(bsz, seq + n_pooled, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda")
    topk = build_dsv4_sparse_topk_indices(
        batch_size=bsz,
        seq_len=seq,
        key_len=seq + n_pooled,
        window_size=128,
        device=q.device,
        compress_ratio=4,
        n_pooled=n_pooled,
    )
    assert (topk == -1).any()
    grad = torch.randn_like(q).transpose(2, 3).contiguous().transpose(2, 3)
    assert not grad.is_contiguous()

    expected, expected_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(
            q_,
            kv_,
            sinks_,
            topk,
            dim**-0.5,
            backend="sparse_torch",
        ),
        (q, kv, sinks),
        grad,
    )
    actual, actual_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dsv4_sparse_attention(
            q_,
            kv_,
            sinks_,
            topk,
            dim**-0.5,
            backend="tilelang",
        ),
        (q, kv, sinks),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not _can_run_tilelang_indexer(),
    reason="Miles DSV4 indexer kernel is not installed on a CUDA environment",
)
def test_indexer_tilelang_backend_matches_torch() -> None:
    torch.manual_seed(123)
    bsz, seq, heads, dim, pooled = 2, 17, 8, 128, 5
    q = torch.rand(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16) + 0.1
    pooled_kv = torch.rand(bsz, pooled, dim, device="cuda", dtype=torch.bfloat16) + 0.1
    weights = torch.randn(bsz, seq, heads, device="cuda") * 0.01
    q_pos = torch.arange(seq, device="cuda")
    pooled_pos = torch.arange(pooled, device="cuda")
    valid = pooled_pos.unsqueeze(0) < ((q_pos + 1) // 4).unsqueeze(1)
    expected = dsv4_indexer_scores(
        q,
        pooled_kv,
        weights,
        compress_ratio=4,
        softmax_scale=dim**-0.5,
        backend="torch",
    )
    expected = torch.where(valid.unsqueeze(0), expected, torch.full_like(expected, float("-inf")))
    actual = dsv4_indexer_scores(
        q,
        pooled_kv,
        weights,
        compress_ratio=4,
        softmax_scale=dim**-0.5,
        backend="tilelang",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not _can_run_tilelang_indexer(),
    reason="Vendored Miles DSV4 indexer kernel is not available on a CUDA environment",
)
def test_indexer_topk_tilelang_backend_matches_torch() -> None:
    torch.manual_seed(123)
    bsz, seq, heads, dim, pooled, topk = 1, 17, 8, 128, 5, 4
    q = torch.rand(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16) + 0.1
    pooled_kv = torch.rand(bsz, pooled, dim, device="cuda", dtype=torch.bfloat16) + 0.1
    weights = torch.randn(bsz, seq, heads, device="cuda") * 0.01
    topk_indices = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk).expand(bsz, seq, -1)
    q_pos = torch.arange(seq, device="cuda")
    valid_end = ((q_pos + 1) // 4).view(1, seq, 1)
    topk_indices = torch.where(topk_indices < valid_end, topk_indices, torch.full_like(topk_indices, -1))
    grad = torch.randn(bsz, seq, topk, device="cuda")

    expected, expected_grads = _run_forward_backward(
        lambda q_, pooled_kv_, weights_: dsv4_indexer_topk_scores(
            q_,
            pooled_kv_,
            weights_,
            topk_indices,
            compress_ratio=4,
            softmax_scale=dim**-0.5,
            backend="torch",
        ),
        (q, pooled_kv, weights),
        grad,
    )
    actual, actual_grads = _run_forward_backward(
        lambda q_, pooled_kv_, weights_: dsv4_indexer_topk_scores(
            q_,
            pooled_kv_,
            weights_,
            topk_indices,
            compress_ratio=4,
            softmax_scale=dim**-0.5,
            backend="tilelang",
        ),
        (q, pooled_kv, weights),
        grad,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not _can_run_tilelang_sparse_attention(),
    reason="TileLang sparse attention requires CUDA compute capability 8.9 or newer",
)
def test_visual_sparse_attention_module_matches_dense_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    check_visual_sparse_attention_module_matches_dense_forward_backward(
        monkeypatch, device="cuda", dtype=torch.bfloat16, uses_tilelang=True
    )


@pytest.mark.skipif(
    not _can_run_tilelang_sparse_attn(),
    reason="Vendored DSV4 sparse-attention CUDA kernel is unavailable",
)
@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_flash_full_shape_sparse_attention_graph(compress_ratio: int) -> None:
    """Check full Flash attention shapes against a dense FP32 Torch oracle.

    This isolates fixed-key attention arithmetic and graph replay. It does not
    validate the learned indexer or pretrained checkpoint loading.
    """
    import json
    from pathlib import Path

    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dense_attention_topk_torch

    torch.manual_seed(1234)
    bsz, seq, heads, dim = 1, 4096, 64, 512
    pooled = seq // compress_ratio if compress_ratio else 0
    q = torch.randn(bsz, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(bsz, seq + pooled, dim, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(heads, device="cuda")
    grad = torch.randn_like(q)
    compressed = None
    if compress_ratio == 4:
        # Distinct valid keys, selected from the causal compressed history.
        # Later queries omit valid keys once history exceeds index_topk=512.
        history = (torch.arange(seq, device="cuda") + 1) // compress_ratio
        permutation = torch.rand(seq, pooled, device="cuda")
        permutation.masked_fill_(torch.arange(pooled, device="cuda")[None] >= history[:, None], -1)
        values, indices = permutation.topk(512, dim=-1)
        compressed = torch.where(values >= 0, indices, -1).unsqueeze(0)
    topk = build_dsv4_sparse_topk_indices(
        batch_size=bsz,
        seq_len=seq,
        key_len=seq + pooled,
        window_size=128,
        device=q.device,
        compress_ratio=compress_ratio,
        compressed_topk=compressed,
        n_pooled=pooled,
    ).to(torch.int32)
    fn = lambda q_, kv_, sinks_: dsv4_sparse_attention(q_, kv_, sinks_, topk, dim**-0.5, backend="tilelang")

    expected, expected_grads = _run_forward_backward(
        lambda q_, kv_, sinks_: dense_attention_topk_torch(q_, kv_, sinks_, topk, dim**-0.5),
        (q.float(), kv.float(), sinks),
        grad.float(),
    )
    reference = (expected, *expected_grads)
    eager = []
    for _ in range(3):
        output, gradients = _run_forward_backward(fn, (q, kv, sinks), grad)
        eager.append((output, *gradients))

    leaves = [value.detach().clone().requires_grad_(True) for value in (q, kv, sinks)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for value in leaves:
                value.grad = None
            output = fn(*leaves)
            output.backward(grad)
        for value in leaves:
            value.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn(*leaves)
            output.backward(grad)
        graph_results = []
        for _ in range(3):
            graph.replay()
            graph_results.append(tuple(value.detach().clone() for value in (output, *(v.grad for v in leaves))))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
        actual = actual.float()
        expected = expected.float()
        difference = actual - expected
        return {
            "relative_rmse": (
                difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-20)
            ).item(),
            "max_abs": difference.abs().max().item(),
            "fraction_changed": (difference != 0).float().mean().item(),
        }

    labels = ("output", "dq", "dkv", "dsinks")
    result = {
        "shape": [bsz, seq, heads, dim],
        "key_length": seq + pooled,
        "topk_width": topk.shape[-1],
        "compress_ratio": compress_ratio,
        "reference": "independent dense-mask FP32 Torch",
        "eager_vs_reference": {name: metrics(a, b) for name, a, b in zip(labels, eager[0], reference, strict=True)},
        "eager_repeats": [
            {name: metrics(a, b) for name, a, b in zip(labels, row, eager[0], strict=True)} for row in eager[1:]
        ],
        "graph_vs_eager": [
            {name: metrics(a, b) for name, a, b in zip(labels, row, eager[0], strict=True)} for row in graph_results
        ],
    }
    Path(f"flash_attention_graph_ratio{compress_ratio}.json").write_text(json.dumps(result, indent=2))
    for row in (eager[0], *graph_results):
        torch.testing.assert_close(row[0].float(), reference[0], rtol=2e-2, atol=2e-2)
        for actual, expected in zip(row[1:], reference[1:], strict=True):
            torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)
    for row in graph_results:
        torch.testing.assert_close(row[0], eager[0][0], rtol=0, atol=0)
    # An independent wrong-gradient control must fail the same oracle gate.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(-graph_results[0][2].float(), reference[2], rtol=5e-2, atol=5e-2)
    graph.reset()
