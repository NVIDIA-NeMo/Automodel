# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the standalone helpers in
``nemo_automodel.components.models.deepseek_v4.layers``.

Pieces here are easy to construct in isolation (grouped output projection,
the Hyper-Connections weight builder, and the partial-RoPE helper).
Full-model behaviour is covered by ``test_dsv4_model_smoke.py``.
"""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
    DeepseekV4RotaryEmbedding,
    _apply_partial_rope_interleaved,
    _build_indexer_topk_compressed_mask,
    _rms_norm_last_dim,
    _yarn_correction_dim,
    _yarn_correction_range,
    _yarn_linear_ramp,
    build_causal_padding_mask,
    build_packed_causal_padding_mask,
    eager_attention_with_sink,
)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    build_dsv4_sparse_topk_indices,
    dense_attention_topk_torch,
    dsv4_indexer_scores,
    dsv4_indexer_topk_scores,
    dsv4_sinkhorn_normalize,
    dsv4_sparse_attention,
    extract_indexer_topk_scores_torch,
    indexer_scores_torch,
    is_dsv4_kernel_available,
    sinkhorn_normalize_torch,
)


def _run_forward_backward(fn, inputs, grad_output):
    leaves = [input_.detach().clone().requires_grad_(True) for input_ in inputs]
    output = fn(*leaves)
    torch.autograd.backward(output, grad_output.to(dtype=output.dtype, device=output.device))
    return output.detach(), [leaf.grad.detach() for leaf in leaves]


class TestDeepseekV4AttentionMask:
    def test_indexer_topk_mask_preserves_pool_zero_with_clamped_invalid_entries(self):
        """A valid pool index 0 must survive duplicate writes from clamped ``-1`` slots."""
        attention_mask = torch.zeros(1, 1, 1, 1)
        indexer_topk = torch.tensor([[[2, 0, -1, -1]]])

        min_val = torch.finfo(attention_mask.dtype).min
        expected_compressed_mask = torch.tensor([[[[0.0, min_val, 0.0, min_val, min_val]]]])
        compressed_mask = _build_indexer_topk_compressed_mask(attention_mask, indexer_topk, n_pooled=5).unsqueeze(1)
        torch.testing.assert_close(compressed_mask, expected_compressed_mask)

    def test_indexer_topk_mask_drops_high_invalid_entries(self):
        """Positive out-of-range pool IDs must not become scatter/gather indices."""
        attention_mask = torch.zeros(1, 1, 1, 1)
        indexer_topk = torch.tensor([[[2, 99, 0, -1]]])

        min_val = torch.finfo(attention_mask.dtype).min
        expected_compressed_mask = torch.tensor([[[[0.0, min_val, 0.0, min_val, min_val]]]])
        compressed_mask = _build_indexer_topk_compressed_mask(attention_mask, indexer_topk, n_pooled=5).unsqueeze(1)
        torch.testing.assert_close(compressed_mask, expected_compressed_mask)

    def test_sparse_topk_builder_drops_high_invalid_entries(self):
        compressed_topk = torch.tensor([[[0, 99], [1, -1], [99, 0], [-1, -1]]])
        topk = build_dsv4_sparse_topk_indices(
            batch_size=1,
            seq_len=4,
            key_len=6,
            window_size=2,
            device=torch.device("cpu"),
            compress_ratio=4,
            compressed_topk=compressed_topk,
            n_pooled=2,
        )

        assert (topk >= 6).sum().item() == 0
        assert topk[0, 0, -2].item() == 4
        assert topk[0, 0, -1].item() == -1

    def test_packed_topk_uses_padded_lengths_for_tail_padding(self):
        seq_len = 8
        real_seq_lens = torch.tensor([[3, 2]], dtype=torch.long)
        padded_seq_lens = torch.tensor([[3, 5]], dtype=torch.long)

        real_mask = build_packed_causal_padding_mask(
            real_seq_lens,
            seq_len=seq_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            sliding_window=4,
        )
        real_topk = build_dsv4_sparse_topk_indices(
            batch_size=1,
            seq_len=seq_len,
            key_len=seq_len,
            window_size=4,
            device=torch.device("cpu"),
            attention_mask=real_mask,
        )

        padded_mask = build_packed_causal_padding_mask(
            padded_seq_lens,
            seq_len=seq_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            sliding_window=4,
        )
        padded_topk = build_dsv4_sparse_topk_indices(
            batch_size=1,
            seq_len=seq_len,
            key_len=seq_len,
            window_size=4,
            device=torch.device("cpu"),
            attention_mask=padded_mask,
        )

        assert (real_topk[0, 5:] >= 0).sum(dim=-1).eq(0).all()
        assert (padded_topk[0, 5:] >= 0).sum(dim=-1).gt(0).all()


class TestRMSNormLastDim:
    def _reference(self, x, eps):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

    def test_matches_reference_forward_backward(self):
        eps = 1e-6
        x = torch.randn(2, 3, 5, 7)
        grad = torch.randn_like(x)
        expected, (expected_grad,) = _run_forward_backward(lambda x_: self._reference(x_, eps), (x,), grad)
        actual, (actual_grad,) = _run_forward_backward(lambda x_: _rms_norm_last_dim(x_, eps), (x,), grad)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bf16 RMSNorm parity")
    def test_cuda_bfloat16_matches_reference_forward_backward(self):
        eps = 1e-6
        x = torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn_like(x)
        expected, (expected_grad,) = _run_forward_backward(lambda x_: self._reference(x_, eps), (x,), grad)
        actual, (actual_grad,) = _run_forward_backward(lambda x_: _rms_norm_last_dim(x_, eps), (x,), grad)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)

    def test_bias_free_linear_matches_post_scale_forward_backward(self):
        eps = 1e-6
        x = torch.randn(2, 5, 7)
        weight = torch.randn(3, 7)
        grad = torch.randn(2, 5, 3)

        def reference(x_, weight_):
            scale = torch.rsqrt(x_.square().mean(-1, keepdim=True) + eps)
            return torch.nn.functional.linear(x_, weight_) * scale

        expected, expected_grads = _run_forward_backward(reference, (x, weight), grad)
        actual, actual_grads = _run_forward_backward(
            lambda x_, weight_: torch.nn.functional.linear(_rms_norm_last_dim(x_, eps), weight_),
            (x, weight),
            grad,
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)


class TestDeepseekV4GroupedLinear:
    """Block-diagonal grouped linear backing the V4 attention output
    projection ``wo_a`` (n_groups=8 in DSV4-Flash).
    """

    def test_weight_shape(self):
        # n_heads=64, head_dim=512, n_groups=8 -> in_per_group=4096, out_total=8192
        proj = DeepseekV4GroupedLinear(in_features_per_group=4096, out_features=8192, n_groups=8)
        assert proj.weight.shape == (8192, 4096)
        assert proj.bias is None  # bias=False default

    def test_n_groups_attribute(self):
        proj = DeepseekV4GroupedLinear(64, 256, n_groups=4)
        assert proj.n_groups == 4

    def test_forward_shape(self):
        bsz, seq, n_groups, in_per = 2, 4, 8, 64
        out_total = 256
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        x = torch.randn(bsz, seq, n_groups, in_per)
        out = proj(x)
        assert out.shape == (bsz, seq, n_groups, out_total // n_groups)

    def test_forward_matches_per_group_matmul(self):
        """Output should equal a per-group matmul: y[g] = x[g] @ W[g].T."""
        n_groups, in_per = 4, 8
        out_total = 16
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        with torch.no_grad():
            proj.weight.normal_()
        x = torch.randn(3, n_groups, in_per)
        out = proj(x)
        w_ref = proj.weight.view(n_groups, out_total // n_groups, in_per)
        out_ref = torch.stack([x[:, g, :] @ w_ref[g].t() for g in range(n_groups)], dim=1)
        torch.testing.assert_close(out, out_ref)


class TestDeepseekV4HyperConnection:
    """``compute_weights`` returns the (pre, post, comb) tensors used at
    each HC site.  Shapes are deterministic given ``hc_mult`` and
    ``hidden_size``; values change with the learned parameters.
    """

    @pytest.fixture
    def hc(self):
        # ``DeepseekV4HyperConnection`` allocates ``fn``/``base``/``scale``
        # via ``torch.empty(...)``; those are uninitialized memory and may
        # contain NaN bit patterns.  Zero them so the Sinkhorn-row test has
        # a well-defined starting point (real model loads init from the
        # checkpoint via the state-dict adapter, not via ``empty``).
        m = DeepseekV4HyperConnection(
            hc_mult=4,
            hidden_size=16,
            hc_sinkhorn_iters=4,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )
        with torch.no_grad():
            m.fn.zero_()
            m.base.zero_()
            m.scale.zero_()
        return m

    def test_parameter_dtypes_are_fp32(self, hc):
        # HC params must stay fp32 even when the surrounding model is bf16.
        assert hc.fn.dtype == torch.float32
        assert hc.base.dtype == torch.float32
        assert hc.scale.dtype == torch.float32

    def test_compute_weights_output_shapes(self, hc):
        bsz, seq, hc_mult, hidden = 2, 5, 4, 16
        x = torch.randn(bsz, seq, hc_mult, hidden)
        pre, post, comb = hc.compute_weights(x)
        assert pre.shape == (bsz, seq, hc_mult)
        assert post.shape == (bsz, seq, hc_mult)
        assert comb.shape == (bsz, seq, hc_mult, hc_mult)

    def test_post_uses_2x_sigmoid(self, hc):
        """``post`` is ``2 * sigmoid(...)``, so it can exceed 1."""
        x = torch.randn(2, 3, 4, 16)
        with torch.no_grad():
            hc.scale.data[1].fill_(50.0)
            hc.base.data[hc.hc_mult : 2 * hc.hc_mult].fill_(50.0)
        _, post, _ = hc.compute_weights(x)
        assert post.max().item() > 1.5

    def test_comb_rows_sum_close_to_one(self, hc):
        """After softmax+sinkhorn, ``comb`` is doubly-(near-)stochastic."""
        x = torch.randn(1, 2, 4, 16)
        _, _, comb = hc.compute_weights(x)
        row_sums = comb.sum(dim=-1)
        col_sums = comb.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-2)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=0, atol=1e-2)


class TestDeepseekV4OptimizedKernels:
    """Numerical equivalence tests for optional DSV4 kernel dispatch."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
            ),
        ],
    )
    def test_eager_attention_with_sink_matches_unpacked_packed_sequence(self, device):
        torch.manual_seed(123)
        batch, seq_len, heads, dim = 2, 6, 3, 8
        seq_lens = torch.tensor([[3, 2, 0], [1, 4, 0]], dtype=torch.long)
        sliding_window = 3
        q = torch.randn(batch, heads, seq_len, dim, device=device)
        kv = torch.randn(batch, 1, seq_len, dim, device=device)
        sinks = torch.randn(heads, device=device)
        grad = torch.randn(batch, seq_len, heads, dim, device=device)
        grad = torch.where(
            (
                torch.arange(seq_len, device=device).expand(batch, -1)
                < seq_lens.to(device=device).sum(dim=-1, keepdim=True)
            )
            .unsqueeze(-1)
            .unsqueeze(-1),
            grad,
            torch.zeros_like(grad),
        )

        class DummyModule:
            num_key_value_groups = heads
            training = False

            def __init__(self, sinks):
                self.sinks = sinks

        def packed_attention(q_, kv_, sinks_):
            attention_mask = build_packed_causal_padding_mask(
                seq_lens,
                seq_len=seq_len,
                dtype=q_.dtype,
                device=q_.device,
                sliding_window=sliding_window,
            )
            return eager_attention_with_sink(
                DummyModule(sinks_),
                q_,
                kv_,
                kv_,
                attention_mask,
                scaling=dim**-0.5,
            )[0]

        def unpacked_attention(q_, kv_, sinks_):
            outputs = []
            for batch_idx in range(batch):
                batch_outputs = []
                offset = 0
                for length in seq_lens[batch_idx].tolist():
                    if length == 0:
                        continue
                    attention_mask = build_causal_padding_mask(
                        attention_mask=None,
                        seq_len=length,
                        dtype=q_.dtype,
                        device=q_.device,
                        batch_size=1,
                        sliding_window=sliding_window,
                    )
                    doc_output = eager_attention_with_sink(
                        DummyModule(sinks_),
                        q_[batch_idx : batch_idx + 1, :, offset : offset + length],
                        kv_[batch_idx : batch_idx + 1, :, offset : offset + length],
                        kv_[batch_idx : batch_idx + 1, :, offset : offset + length],
                        attention_mask,
                        scaling=dim**-0.5,
                    )[0]
                    batch_outputs.append(doc_output)
                    offset += length
                padding = seq_len - offset
                if padding:
                    batch_outputs.append(q_.new_zeros(1, padding, heads, dim))
                outputs.append(torch.cat(batch_outputs, dim=1))
            return torch.cat(outputs, dim=0)

        expected, expected_grads = _run_forward_backward(unpacked_attention, (q, kv, sinks), grad)
        actual, actual_grads = _run_forward_backward(packed_attention, (q, kv, sinks), grad)

        rtol, atol = (1e-4, 1e-5) if device == "cuda" else (1e-5, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)

    def test_sinkhorn_torch_backend_matches_reference(self):
        torch.manual_seed(123)
        x = torch.randn(2, 3, 4, 4)
        grad = torch.randn_like(x)
        expected, (expected_grad,) = _run_forward_backward(
            lambda x_: sinkhorn_normalize_torch(x_, repeat=5, eps=1e-6),
            (x,),
            grad,
        )
        actual, (actual_grad,) = _run_forward_backward(
            lambda x_: dsv4_sinkhorn_normalize(x_, backend="torch", repeat=5, eps=1e-6),
            (x,),
            grad,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(
        not is_dsv4_kernel_available("sinkhorn") or not torch.cuda.is_available(),
        reason="TileKernels sinkhorn kernel is not installed on a CUDA environment",
    )
    def test_sinkhorn_tilelang_backend_matches_torch(self):
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
    def test_sinkhorn_tilelang_backend_accepts_non_contiguous_grad(self):
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

    def test_sparse_attention_torch_matches_dense_topk_reference(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim = 2, 7, 3, 8
        n_pooled = 3
        q = torch.randn(bsz, seq, heads, dim)
        kv = torch.randn(bsz, seq + n_pooled, dim)
        sinks = torch.randn(heads)
        grad = torch.randn_like(q)
        topk = build_dsv4_sparse_topk_indices(
            batch_size=bsz,
            seq_len=seq,
            key_len=seq + n_pooled,
            window_size=4,
            device=q.device,
            compress_ratio=4,
            n_pooled=n_pooled,
        )
        expected, expected_grads = _run_forward_backward(
            lambda q_, kv_, sinks_: dense_attention_topk_torch(q_, kv_, sinks_, topk, dim**-0.5),
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
                backend="sparse_torch",
            ),
            (q, kv, sinks),
            grad,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_sparse_attention_torch_ignores_high_invalid_topk(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim = 1, 3, 2, 8
        q = torch.randn(bsz, seq, heads, dim)
        kv = torch.randn(bsz, 4, dim)
        sinks = torch.randn(heads)
        grad = torch.randn_like(q)
        topk = torch.tensor([[[0, 99, -1], [1, 2, 99], [3, -1, 99]]])

        expected, expected_grads = _run_forward_backward(
            lambda q_, kv_, sinks_: dense_attention_topk_torch(q_, kv_, sinks_, topk, dim**-0.5),
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
                backend="sparse_torch",
            ),
            (q, kv, sinks),
            grad,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_sparse_attention_matches_current_eager_mask_path(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim = 2, 7, 3, 8
        n_pooled, ratio, window = 3, 4, 4
        q = torch.randn(bsz, seq, heads, dim)
        kv = torch.randn(bsz, seq + n_pooled, dim)
        sinks = torch.randn(heads)
        grad = torch.randn_like(q)

        base_mask = build_causal_padding_mask(
            attention_mask=None,
            seq_len=seq,
            dtype=q.dtype,
            device=q.device,
            batch_size=bsz,
            sliding_window=window,
        )
        min_val = torch.finfo(q.dtype).min
        q_pos = torch.arange(seq, device=q.device)
        p_pos = torch.arange(n_pooled, device=q.device)
        allowed = p_pos.unsqueeze(0) < ((q_pos + 1) // ratio).unsqueeze(1)
        compressed_mask = torch.where(
            allowed,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), min_val, dtype=q.dtype, device=q.device),
        )
        attention_mask = torch.cat([base_mask, compressed_mask.expand(bsz, seq, n_pooled).unsqueeze(1)], dim=-1)
        topk = build_dsv4_sparse_topk_indices(
            batch_size=bsz,
            seq_len=seq,
            key_len=seq + n_pooled,
            window_size=window,
            device=q.device,
            attention_mask=attention_mask,
            compress_ratio=ratio,
            n_pooled=n_pooled,
        )

        class DummyModule:
            num_key_value_groups = heads

            def __init__(self, sinks):
                self.sinks = sinks
                self.training = False

        expected, expected_grads = _run_forward_backward(
            lambda q_, kv_, sinks_: eager_attention_with_sink(
                DummyModule(sinks_),
                q_.transpose(1, 2),
                kv_.unsqueeze(1),
                kv_.unsqueeze(1),
                attention_mask,
                scaling=dim**-0.5,
            )[0],
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
                backend="sparse_torch",
            ),
            (q, kv, sinks),
            grad,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not is_dsv4_kernel_available("sparse_attn") or not torch.cuda.is_available(),
        reason="Vendored Miles DSV4 sparse-attention kernel is not available on a CUDA environment",
    )
    def test_sparse_attention_tilelang_backend_matches_torch(self):
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
        not is_dsv4_kernel_available("sparse_attn") or not torch.cuda.is_available(),
        reason="Vendored Miles DSV4 sparse-attention kernel is not available on a CUDA environment",
    )
    def test_sparse_attention_tilelang_backend_matches_torch_with_causal_padding_shape(self):
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

    def test_indexer_scores_torch_backend_matches_reference(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim, pooled = 2, 7, 3, 8, 5
        q = torch.randn(bsz, seq, heads, dim)
        pooled_kv = torch.randn(bsz, pooled, dim)
        weights = torch.randn(bsz, seq, heads)
        grad = torch.randn(bsz, seq, pooled)
        expected, expected_grads = _run_forward_backward(
            lambda q_, pooled_kv_, weights_: indexer_scores_torch(q_, pooled_kv_, weights_, dim**-0.5),
            (q, pooled_kv, weights),
            grad,
        )
        actual, actual_grads = _run_forward_backward(
            lambda q_, pooled_kv_, weights_: dsv4_indexer_scores(
                q_,
                pooled_kv_,
                weights_,
                compress_ratio=4,
                softmax_scale=dim**-0.5,
                backend="torch",
            ),
            (q, pooled_kv, weights),
            grad,
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_indexer_topk_scores_torch_backend_matches_reference(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim, pooled, topk = 2, 7, 3, 8, 5, 4
        q = torch.randn(bsz, seq, heads, dim)
        pooled_kv = torch.randn(bsz, pooled, dim)
        weights = torch.randn(bsz, seq, heads)
        topk_indices = torch.randint(0, pooled, (bsz, seq, topk))
        topk_indices[:, :, -1] = -1
        grad = torch.randn(bsz, seq, topk)

        expected, expected_grads = _run_forward_backward(
            lambda q_, pooled_kv_, weights_: extract_indexer_topk_scores_torch(
                indexer_scores_torch(q_, pooled_kv_, weights_, dim**-0.5),
                topk_indices,
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
                backend="torch",
            ),
            (q, pooled_kv, weights),
            grad,
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_indexer_topk_scores_torch_ignores_high_invalid_topk(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim, pooled = 1, 3, 2, 8, 4
        q = torch.randn(bsz, seq, heads, dim)
        pooled_kv = torch.randn(bsz, pooled, dim)
        weights = torch.randn(bsz, seq, heads)
        topk_indices = torch.tensor([[[0, 99, -1], [1, 2, 99], [3, -1, 99]]])
        grad = torch.randn(bsz, seq, topk_indices.shape[-1])

        expected, expected_grads = _run_forward_backward(
            lambda q_, pooled_kv_, weights_: extract_indexer_topk_scores_torch(
                indexer_scores_torch(q_, pooled_kv_, weights_, dim**-0.5),
                topk_indices,
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
                backend="torch",
            ),
            (q, pooled_kv, weights),
            grad,
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.skipif(
        not is_dsv4_kernel_available("indexer") or not torch.cuda.is_available(),
        reason="Miles DSV4 indexer kernel is not installed on a CUDA environment",
    )
    def test_indexer_tilelang_backend_matches_torch(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim, pooled = 2, 16, 8, 128, 4
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
        not is_dsv4_kernel_available("indexer") or not torch.cuda.is_available(),
        reason="Vendored Miles DSV4 indexer kernel is not available on a CUDA environment",
    )
    def test_indexer_topk_tilelang_backend_matches_torch(self):
        torch.manual_seed(123)
        bsz, seq, heads, dim, pooled, topk = 1, 16, 8, 128, 4, 4
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


class TestApplyPartialRopeInterleaved:
    """Released DSV4-Flash uses INTERLEAVED RoPE pairs ``(2k, 2k+1)``."""

    def _reference_apply(self, x, cos, sin, rd):
        half = rd // 2
        nope, rope = x[..., :-rd], x[..., -rd:]
        rope_pairs = rope.unflatten(-1, (-1, 2))
        a, b = rope_pairs[..., 0], rope_pairs[..., 1]
        c = cos[..., :half]
        s = sin[..., :half]
        while c.ndim < a.ndim:
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
        new_a = a * c - b * s
        new_b = a * s + b * c
        new_rope = torch.stack([new_a, new_b], dim=-1).flatten(-2)
        return torch.cat([nope, new_rope], dim=-1)

    def _make_cos_sin(self, batch, seq, rd):
        """Build the Llama-style ``cat([f, f], -1)`` cos/sin tensors that
        ``_apply_partial_rope_interleaved`` consumes (it uses only the
        first half).
        """
        half = rd // 2
        freqs = torch.arange(half, dtype=torch.float32) * 0.1
        pos = torch.arange(seq, dtype=torch.float32).unsqueeze(-1) * freqs
        cos_h = pos.cos()
        sin_h = pos.sin()
        cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        return cos, sin

    def test_preserves_nope_prefix(self):
        bsz, heads, seq, rd, nope = 1, 2, 4, 8, 16
        x = torch.randn(bsz, heads, seq, nope + rd)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y[..., :nope], x[..., :nope])

    def test_inverse_with_negated_sin_round_trips(self):
        """Rotating with ``sin`` then ``-sin`` recovers the input."""
        bsz, heads, seq, rd = 2, 4, 3, 8
        x = torch.randn(bsz, heads, seq, rd + 4)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        rotated = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        unrotated = _apply_partial_rope_interleaved(rotated, cos, -sin, rope_head_dim=rd)
        torch.testing.assert_close(unrotated, x, rtol=1e-4, atol=1e-5)

    def test_zero_angles_is_identity(self):
        bsz, heads, seq, rd = 1, 1, 2, 4
        x = torch.randn(bsz, heads, seq, rd)
        cos = torch.ones(bsz, seq, rd)
        sin = torch.zeros(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y, x)

    def test_matches_reference_forward_backward(self):
        bsz, heads, seq, rd, nope = 2, 3, 4, 8, 5
        x = torch.randn(bsz, heads, seq, nope + rd, requires_grad=True)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        cos = cos.clone().requires_grad_()
        sin = sin.clone().requires_grad_()

        x_ref = x.detach().clone().requires_grad_()
        cos_ref = cos.detach().clone().requires_grad_()
        sin_ref = sin.detach().clone().requires_grad_()
        expected = self._reference_apply(x_ref, cos_ref, sin_ref, rd)
        actual = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)

        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(cos.grad, cos_ref.grad)
        torch.testing.assert_close(sin.grad, sin_ref.grad)


class TestYaRNHelpers:
    """Sanity checks on the three pure-math helpers that build the YaRN
    correction ramp.  Reference math: ``dsv4flash/inference/model.py:
    precompute_freqs_cis`` (the inner ``find_correction_dim`` /
    ``find_correction_range`` / ``linear_ramp_factor`` helpers).
    """

    def test_correction_dim_monotonic_in_rotations(self):
        """``find_correction_dim`` is monotonically *decreasing* in
        ``num_rotations`` (more rotations ⇒ lower dim index, since the
        higher-frequency dims rotate more often within a fixed window).
        """
        # DSV4-Flash compress-rope: dim=64, base=160000, max_seq_len=65536
        d_low = _yarn_correction_dim(num_rotations=1, dim=64, base=160000, max_seq_len=65536)
        d_high = _yarn_correction_dim(num_rotations=32, dim=64, base=160000, max_seq_len=65536)
        assert d_high < d_low

    def test_correction_range_clamped(self):
        """``find_correction_range`` clamps to ``[0, dim-1]``."""
        low, high = _yarn_correction_range(low_rot=32, high_rot=1, dim=64, base=160000, max_seq_len=65536)
        assert 0 <= low <= high <= 63

    def test_linear_ramp_endpoints_and_clamp(self):
        """Ramp is ``0`` below ``min_v``, ``1`` above ``max_v``, linear in
        between.  ``dim=32`` matches DSV4 inv_freq length.
        """
        ramp = _yarn_linear_ramp(min_v=15.0, max_v=25.0, dim=32)
        # Below min: 0
        assert torch.all(ramp[:16] == 0.0)
        # Above max: 1
        assert torch.all(ramp[26:] == 1.0)
        # In between: in [0, 1]
        assert torch.all((ramp >= 0.0) & (ramp <= 1.0))

    def test_linear_ramp_handles_min_eq_max(self):
        """When ``min == max`` the helper bumps ``max`` by 1e-3 to avoid
        division by zero; the ramp should still be a valid step function."""
        ramp = _yarn_linear_ramp(min_v=5.0, max_v=5.0, dim=10)
        assert torch.all((ramp >= 0.0) & (ramp <= 1.0))


class TestDeepseekV4RotaryEmbeddingYaRN:
    """``DeepseekV4RotaryEmbedding`` with and without ``rope_scaling``.

    DSV4-Flash uses YaRN on the compress-rope path only:
        rope_theta=160000, factor=16, original_max_position_embeddings=65536,
        beta_fast=32, beta_slow=1.
    """

    def _yarn_kwargs(self, **overrides):
        kwargs = dict(
            rope_theta=160000.0,
            head_dim=128,
            partial_rotary_factor=0.5,  # qk_rope_head_dim=64
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
        )
        kwargs.update(overrides)
        return kwargs

    def test_no_rope_scaling_is_plain_rope(self):
        """``rope_scaling=None`` ⇒ plain ``1 / theta^(2i/d)`` inv_freq."""
        rope = DeepseekV4RotaryEmbedding(rope_theta=10000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None)
        dim = 64  # head_dim * partial_rotary_factor
        expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        torch.testing.assert_close(rope.inv_freq, expected)

    def test_yarn_attenuates_high_freq_dims_by_factor(self):
        """The highest-frequency dims (above the correction range) are
        divided by ``factor`` exactly; lowest-frequency dims (below the
        correction range) are unchanged.
        """
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())

        ratio = yarn.inv_freq / plain.inv_freq

        # Low-frequency dims (small i, low rotation rate) are below the
        # correction-range floor — ramp=0, smooth=1, so inv_freq is unchanged.
        torch.testing.assert_close(ratio[:8], torch.ones(8), rtol=0, atol=1e-5)

        # High-frequency dims (large i, fast rotation) are above the ceiling —
        # ramp=1, smooth=0, so inv_freq /= factor (exactly 1/16).
        torch.testing.assert_close(ratio[-4:], torch.full((4,), 1.0 / 16.0), rtol=0, atol=1e-5)

        # Middle band is monotonically interpolating between 1.0 and 1/16.
        mid = ratio[8:-4]
        assert torch.all(mid <= ratio[7:-4][:-1] + 1e-6)
        assert torch.all(mid >= 1.0 / 16.0 - 1e-6)

    def test_yarn_factor_one_is_no_op(self):
        """``factor=1`` ⇒ ``inv_freq / 1 * (1-smooth) + inv_freq*smooth`` = ``inv_freq``."""
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            **self._yarn_kwargs(
                rope_scaling={
                    "type": "yarn",
                    "factor": 1,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq, rtol=0, atol=1e-7)

    def test_yarn_zero_original_max_pos_is_no_op(self):
        """``original_max_position_embeddings=0`` short-circuits YaRN
        (matches reference's ``if original_seq_len > 0`` gate)."""
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            **self._yarn_kwargs(
                rope_scaling={
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 0,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq, rtol=0, atol=1e-7)

    def test_yarn_unrecognized_type_is_no_op(self):
        """A ``rope_scaling`` dict whose ``type`` is not ``"yarn"`` should
        be ignored (the gate is exact-string-match insensitive only to case).
        """
        plain = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.5, rope_scaling=None
        )
        yarn = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0,
            head_dim=128,
            partial_rotary_factor=0.5,
            rope_scaling={"type": "linear", "factor": 16},
        )
        torch.testing.assert_close(yarn.inv_freq, plain.inv_freq)

    def test_yarn_matches_reference_math_pointwise(self):
        """Recompute YaRN's ``inv_freq`` from the reference formula and
        check pointwise equality (catches any drift in the helper port).
        """
        rope_theta = 160000.0
        dim = 64
        factor = 16
        orig = 65536
        beta_fast = 32
        beta_slow = 1

        # Reference formula from dsv4flash/inference/model.py:
        #   freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))
        #   low, high = find_correction_range(beta_fast, beta_slow, dim, base, orig)
        #   smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        #   freqs = freqs / factor * (1 - smooth) + freqs * smooth
        plain_freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        low, high = _yarn_correction_range(beta_fast, beta_slow, dim, rope_theta, orig)
        smooth = 1.0 - _yarn_linear_ramp(low, high, dim // 2)
        expected = plain_freqs / factor * (1.0 - smooth) + plain_freqs * smooth

        yarn = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())
        torch.testing.assert_close(yarn.inv_freq, expected, rtol=0, atol=1e-7)

    def test_yarn_forward_returns_correct_shape_and_dtype(self):
        """Smoke check: the YaRN-modified rotary still produces ``(cos, sin)``
        sized to ``qk_rope_head_dim`` and downcasts to ``x.dtype`` if the
        forward returns BF16-casted tensors.
        """
        rope = DeepseekV4RotaryEmbedding(**self._yarn_kwargs())
        bsz, seq = 2, 16
        x = torch.zeros(bsz, seq, dtype=torch.bfloat16)
        position_ids = torch.arange(seq).unsqueeze(0).expand(bsz, -1)
        cos, sin = rope(x, position_ids)
        # rope_head_dim = head_dim * partial_rotary_factor = 64
        assert cos.shape == (bsz, seq, 64)
        assert sin.shape == (bsz, seq, 64)
        assert not torch.isnan(cos).any() and not torch.isnan(sin).any()
