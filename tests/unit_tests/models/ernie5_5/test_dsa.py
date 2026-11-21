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

"""Tests for the ERNIE 5.5 dynamic sparse attention module.

The tests run on CPU and carry the ``cpu_only`` marker, except the two that compare the indexer rotation on CPU
with its CUDA path. The eager path is tested end to end. Of the cuDNN path, the tests cover what runs before the
kernels are called: path selection, argument validation and the merge of remote and local indices.
"""

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import dsa as dsa_module
from nemo_automodel.components.models.ernie5_5.dsa import (
    Ernie5_5DSAAttention,
    Ernie5_5DSAFP32Parameters,
    Ernie5_5DSASparseFP32Parameters,
    Ernie5_5Indexer,
    apply_indexer_rope,
    apply_main_rope,
    cudnn_dsa_core,
    ernie5_5_index_scores,
    freeze_ernie5_5_indexer_params,
    latent_mqa_core,
    merge_remote_and_window,
    rotate_indexer_activation,
    select_window_and_remote,
)

_GEOMETRY = (8, 2, 6, 4, 4, 2, 3, 2, 4, 1)


def _dsa_layer(backend: BackendConfig | None = None) -> Ernie5_5DSAAttention:
    """Build a tiny layer: hidden 8, 2 heads, window 2, 2 remote keys, at most 16 tokens on the eager path."""
    return Ernie5_5DSAAttention(
        *_GEOMETRY, window=2, remote_topk=2, dtype=torch.float32, max_eager_sequence=16, backend=backend
    )


def _identity(value: torch.Tensor) -> torch.Tensor:
    """Stand in for the position rotation that the indexer takes as its ``rope`` argument.

    Args:
        value: Tensor of any shape.

    Returns:
        The input tensor itself, unchanged.
    """
    return value


def _cudnn_core_inputs(cu_seqlens: list[int]) -> list[torch.Tensor]:
    """Build shape-consistent CPU inputs of ``cudnn_dsa_core``: three tokens, two heads, three value channels.

    Args:
        cu_seqlens: Cumulative document lengths for the int32 ``cu_seqlens`` tensor.

    Returns:
        The eleven positional tensors of ``cudnn_dsa_core``, in its argument order.
    """
    return [
        torch.ones(3, 2, 4),
        torch.ones(3, 4),
        torch.ones(3, 1, 4),
        torch.ones(3, 4),
        torch.ones(3, 1),
        torch.zeros(2),
        torch.ones(2, 3, 4),
        torch.zeros(3, 6),
        torch.ones(2, 1),
        torch.zeros(2, 1),
        torch.tensor(cu_seqlens, dtype=torch.int32),
    ]


# Rotary helpers


@pytest.mark.cpu_only
def test_indexer_and_main_rope_pair_different_channels() -> None:
    """The indexer layout pairs channel i with i + rope / 2; the main layout pairs adjacent channels."""
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
    frequencies = torch.full((1, 1, 4), math.pi / 2)
    indexer = apply_indexer_rope(x, frequencies)
    main = apply_main_rope(x[..., :4], frequencies)
    torch.testing.assert_close(indexer, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0, 5.0, 6.0]]]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(main, torch.tensor([[[[-2.0, -4.0, 1.0, 3.0]]]]), atol=1e-6, rtol=0)


@pytest.mark.cpu_only
def test_apply_indexer_rope_is_an_unscaled_rotation() -> None:
    """With equal angles in a pair the rotated channels keep their norm, and no scale argument is accepted."""
    x = torch.tensor([[[[3.0, 4.0, 7.0]]]])
    frequencies = torch.full((1, 1, 2), 0.7)
    rotated = apply_indexer_rope(x, frequencies)
    torch.testing.assert_close(rotated[..., :2].norm(), torch.tensor(5.0), atol=1e-5, rtol=0)
    assert rotated[0, 0, 0, 2].item() == 7.0
    with pytest.raises(TypeError, match="mscale"):
        apply_indexer_rope(x, frequencies, mscale=2.0)


@pytest.mark.parametrize(
    ("x_shape", "frequencies_shape", "message"),
    [
        ((2, 3, 6), (2, 3, 4), "Expected x"),
        ((2, 3, 1, 6), (3, 4), "Expected x"),
        ((2, 3, 1, 6), (1, 3, 4), "Expected x"),
        ((2, 3, 1, 6), (2, 5, 4), "Expected x"),
        ((2, 3, 1, 6), (2, 3, 0), "RoPE width must be positive, even"),
        ((2, 3, 1, 6), (2, 3, 3), "RoPE width must be positive, even"),
        ((2, 3, 1, 6), (2, 3, 8), "RoPE width must be positive, even"),
    ],
)
@pytest.mark.cpu_only
def test_apply_indexer_rope_rejects_invalid_shapes(
    x_shape: tuple[int, ...], frequencies_shape: tuple[int, ...], message: str
) -> None:
    """A wrong rank, a batch or sequence mismatch and a zero, odd or too wide rotary width raise ``ValueError``."""
    with pytest.raises(ValueError, match=message):
        apply_indexer_rope(torch.zeros(x_shape), torch.zeros(frequencies_shape))


@pytest.mark.parametrize(
    ("x_shape", "frequencies_shape", "message"),
    [
        ((2, 3, 4), (2, 3, 4), "matching even-width frequencies"),
        ((2, 3, 1, 4), (3, 4), "matching even-width frequencies"),
        ((2, 3, 1, 4), (2, 3, 2), "matching even-width frequencies"),
        ((2, 3, 1, 3), (2, 3, 3), "matching even-width frequencies"),
        ((2, 3, 1, 4), (2, 5, 4), "Expected x"),
        ((2, 3, 1, 0), (2, 3, 0), "RoPE width must be positive, even"),
    ],
)
@pytest.mark.cpu_only
def test_apply_main_rope_rejects_invalid_shapes(
    x_shape: tuple[int, ...], frequencies_shape: tuple[int, ...], message: str
) -> None:
    """A wrong rank, unequal or odd widths, a batch or sequence mismatch and a zero width raise ``ValueError``."""
    with pytest.raises(ValueError, match=message):
        apply_main_rope(torch.zeros(x_shape), torch.zeros(frequencies_shape))


# Indexer activation rotation


def _indexer_activations(dtype: torch.dtype) -> torch.Tensor:
    """Return seeded activations of shape [3, 5, 64, 128], the indexer width of the released model.

    Args:
        dtype: ``torch.bfloat16`` or ``torch.float32``. The FP32 values are not rounded, so almost none of them is
            representable in BF16.

    Returns:
        Tensor of that dtype with standard-normal values.
    """
    generator = torch.Generator().manual_seed(11)
    return torch.randn(3, 5, 64, 128, generator=generator).to(dtype)


def _exact_rotation(values: torch.Tensor) -> torch.Tensor:
    """Return the scaled Hadamard transform of the given values in float64, without any rounding.

    Args:
        values: Tensor of shape [..., 128].

    Returns:
        Float64 tensor of the same shape: ``values @ H / sqrt(128)`` with the 128 x 128 Sylvester matrix ``H``.
    """
    hadamard = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    return values.double() @ hadamard.T * 128**-0.5


@pytest.mark.cpu_only
def test_rotation_is_the_scaled_hadamard_transform_and_returns_bf16() -> None:
    """Four exactly representable values give the exact transform, in BF16, for FP32 and BF16 input."""
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    for dtype in (torch.float32, torch.bfloat16):
        rotated = rotate_indexer_activation(values.to(dtype))
        assert rotated.dtype == torch.bfloat16
        torch.testing.assert_close(rotated.float(), torch.tensor([[5.0, -1.0, -2.0, 0.0]]), rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotation_rounds_once_after_the_transform(dtype: torch.dtype) -> None:
    """The result is within one BF16 step of the exact transform of the BF16-rounded input."""
    values = _indexer_activations(dtype)
    rotated = rotate_indexer_activation(values)
    exact = _exact_rotation(values.to(torch.bfloat16))
    assert rotated.shape == values.shape and rotated.dtype == torch.bfloat16
    # One rounding to BF16 leaves at most one BF16 step (2 ** -7 relative) between the result and the exact value.
    assert bool(((rotated.double() - exact).abs() <= exact.abs() * 2**-7 + 1e-30).all())
    # Rounding the exact value once gives the same bits, except where the FP32 arithmetic lands on the other side
    # of a rounding tie: a few elements in ten million. A second rounding would change a large share of them.
    correctly_rounded = exact.to(torch.bfloat16)
    assert torch.count_nonzero(rotated.view(torch.int16) != correctly_rounded.view(torch.int16)) <= 4


@pytest.mark.cpu_only
def test_rotation_rounds_an_fp32_input_to_bf16_first() -> None:
    """An FP32 input gives the bits of its BF16-rounded copy, not those of the FP32 transform."""
    values = _indexer_activations(torch.float32)
    rounded = values.to(torch.bfloat16)
    assert torch.count_nonzero(rounded.float() != values) > values.numel() // 2
    from_fp32 = rotate_indexer_activation(values)
    from_bf16 = rotate_indexer_activation(rounded)
    assert torch.equal(from_fp32.view(torch.int16), from_bf16.view(torch.int16))
    # The transform of the unrounded FP32 values is a different function: its rounded result differs in many places.
    unrounded = _exact_rotation(values).to(torch.bfloat16)
    assert torch.count_nonzero(unrounded.view(torch.int16) != from_fp32.view(torch.int16)) > values.numel() // 100


@pytest.mark.cpu_only
@pytest.mark.parametrize("channels", [3, 6, 100])
def test_rotation_and_indexer_reject_a_channel_count_that_is_not_a_power_of_two(channels: int) -> None:
    """The rotation raises for such a width, and the indexer rejects it at construction."""
    with pytest.raises(ValueError, match="power-of-two channel count"):
        rotate_indexer_activation(torch.zeros(2, channels))
    with pytest.raises(ValueError, match="head_dim must be a power of two"):
        Ernie5_5Indexer(8, 6, 4, channels, dtype=torch.float32)
    with pytest.raises(ValueError, match="head_dim must be a power of two"):
        Ernie5_5DSAAttention(8, 2, 6, 4, 4, 2, 3, 2, channels, 1, window=2, remote_topk=2, dtype=torch.float32)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_rotates_its_query_and_key_and_returns_bf16(dtype: torch.dtype) -> None:
    """The indexer applies the rotation to the position-rotated query and key, whatever its projection dtype."""
    torch.manual_seed(77)
    model = Ernie5_5Indexer(8, 6, 4, 4, dtype=dtype)
    hidden, latent = torch.randn(1, 5, 8).to(dtype), torch.randn(1, 5, 6).to(dtype)
    q, k, _ = model(hidden, latent, rope=_identity)
    assert q.dtype == k.dtype == torch.bfloat16
    expected_q = rotate_indexer_activation(model.wq_b(latent).unflatten(-1, (4, 4)))
    expected_k = rotate_indexer_activation(model.k_norm(model.wk(hidden).float()).to(dtype))
    assert torch.equal(q.view(torch.int16), expected_q.view(torch.int16))
    assert torch.equal(k.view(torch.int16), expected_k.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares the CPU arithmetic with the CUDA path")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotation_is_bitwise_the_same_on_cpu_and_cuda(dtype: torch.dtype) -> None:
    """The CUDA path, which is the kernel when the package is installed, returns the bits of the CPU path."""
    values = _indexer_activations(dtype)
    on_cpu = rotate_indexer_activation(values)
    on_cuda = rotate_indexer_activation(values.cuda())
    assert on_cuda.is_cuda and on_cuda.dtype == torch.bfloat16
    assert torch.equal(on_cpu.view(torch.int16), on_cuda.cpu().view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares the two CUDA paths of the rotation")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotation_on_cuda_is_the_same_with_and_without_the_kernel(dtype: torch.dtype) -> None:
    """Without the optional kernel package the PyTorch arithmetic runs on CUDA and returns the same bits."""
    values = _indexer_activations(dtype).cuda()
    with_default_path = rotate_indexer_activation(values)
    with patch.object(dsa_module, "_HAS_FAST_HADAMARD", False):
        with_pytorch = rotate_indexer_activation(values)
    assert torch.equal(with_default_path.view(torch.int16), with_pytorch.view(torch.int16))


# Indexer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.cpu_only
def test_indexer_head_weights_round_before_scaling_and_detach_inputs(dtype: torch.dtype) -> None:
    """Head weights are rounded to BF16 before they are scaled, and no gradient reaches the indexer inputs."""
    torch.manual_seed(2202)
    model = Ernie5_5Indexer(8, 6, 4, 4, dtype=dtype)
    hidden = torch.randn(1, 12, 8).to(dtype).requires_grad_()
    latent = torch.randn(1, 12, 6).to(dtype).requires_grad_()
    identity = lambda value: value
    q, k, weights = model(hidden, latent, rope=identity)
    expected = model.weights_proj(hidden.detach()).to(torch.bfloat16) * 4**-0.5 * 4**-0.5
    assert q.shape == (1, 12, 4, 4)
    assert k.shape == (1, 12, 4)
    assert weights.shape == (1, 12, 4) and weights.dtype == torch.bfloat16
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    (q.float().square().sum() + k.float().square().sum() + weights.float().square().sum()).backward()
    assert hidden.grad is None and latent.grad is None
    assert model.weights_proj.weight.grad is not None
    assert torch.isfinite(model.weights_proj.weight.grad).all()


@pytest.mark.cpu_only
def test_empty_sequence_and_fp32_key_norm_contract() -> None:
    """An empty sequence gives empty outputs of the documented shapes, and a BF16 key norm is rejected."""
    model = Ernie5_5Indexer(8, 6, 4, 4, dtype=torch.float32)
    identity = lambda value: value
    q, k, weights = model(torch.empty(1, 0, 8), torch.empty(1, 0, 6), rope=identity)
    assert q.shape == (1, 0, 4, 4)
    assert k.shape == (1, 0, 4)
    assert weights.shape == (1, 0, 4)
    model.k_norm.bfloat16()
    with pytest.raises(RuntimeError, match="LayerNorm parameters must remain FP32"):
        model(torch.empty(1, 0, 8), torch.empty(1, 0, 6), rope=identity)


@pytest.mark.parametrize(
    ("cast", "dtype"),
    [
        ("weight", torch.bfloat16),
        ("bias", torch.bfloat16),
        ("norm", torch.float64),
        ("indexer", torch.bfloat16),
    ],
)
@pytest.mark.cpu_only
def test_indexer_rejects_a_key_norm_that_is_not_fp32(cast: str, dtype: torch.dtype) -> None:
    """A key norm whose weight or bias is not FP32, also after a cast of the whole indexer, raises in forward."""
    model = Ernie5_5Indexer(8, 6, 4, 4, dtype=torch.float32)
    hidden, latent = torch.randn(1, 3, 8), torch.randn(1, 3, 6)
    model(hidden, latent, rope=_identity)
    if cast == "weight":
        model.k_norm.weight = nn.Parameter(model.k_norm.weight.detach().to(dtype))
    elif cast == "bias":
        model.k_norm.bias = nn.Parameter(model.k_norm.bias.detach().to(dtype))
    elif cast == "norm":
        model.k_norm.to(dtype)
    else:
        model.to(dtype)
        hidden, latent = hidden.to(dtype), latent.to(dtype)
    with pytest.raises(RuntimeError, match="LayerNorm parameters must remain FP32"):
        model(hidden, latent, rope=_identity)


@pytest.mark.cpu_only
def test_invalid_indexer_initialization_is_rejected() -> None:
    """A non-finite norm epsilon at construction and a non-finite ``init_std`` raise ``ValueError``."""
    with pytest.raises(ValueError, match="positive"):
        Ernie5_5Indexer(8, 6, 4, 4, dtype=torch.float32, norm_eps=float("nan"))
    model = Ernie5_5Indexer(8, 6, 4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        model.init_weights(float("inf"))


# Index scores, key selection and the eager core


@pytest.mark.cpu_only
def test_select_window_remote_and_document_boundaries() -> None:
    """The selection is the causal window plus the top remote keys, never crosses a document and skips padding."""
    scores = torch.zeros(1, 9, 9)
    scores[0, 5, :4] = torch.tensor([0.3, 9.0, 2.0, 8.0])
    docs = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, -1]])
    chosen = select_window_and_remote(scores, docs, window=2, remote_topk=2)
    assert chosen[0, 5].nonzero().flatten().tolist() == [1, 3, 4, 5]
    assert chosen[0, 6].nonzero().flatten().tolist() == [6]
    assert chosen[0, 7].nonzero().flatten().tolist() == [6, 7]
    assert not chosen[0, 8].any()


@pytest.mark.cpu_only
def test_latent_core_matches_explicit_attention_and_query_key_gradients() -> None:
    """The eager core and its query and key gradients equal a per-query, per-head loop over the selected keys."""
    torch.manual_seed(5563)
    query = torch.randn(1, 4, 2, 3, requires_grad=True)
    kv = torch.randn(1, 4, 3, requires_grad=True)
    rq, rk = query.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    scores = torch.randn(1, 4, 4)
    docs = torch.tensor([[0, 0, 0, -1]])
    sink = torch.randn(2)
    value_up = torch.randn(2, 2, 2)
    gate = torch.randn(1, 4, 4)
    u, v = torch.randn(2, 1), torch.randn(2, 1)
    actual = latent_mqa_core(
        query, kv, scores, docs, sink, value_up, gate, u, v, window=2, remote_topk=1, softmax_scale=0.5
    )

    selected = ((0,), (0, 1), (0, 1, 2), ())
    rows = []
    for qi, keys in enumerate(selected):
        head_outputs = []
        for head in range(2):
            if not keys:
                head_outputs.append(torch.zeros(2))
                continue
            logits = torch.stack([(rq[0, qi, head] * rk[0, key]).sum() * 0.5 for key in keys])
            probs = torch.cat((logits, sink[head : head + 1])).softmax(0)[:-1]
            latent = sum(probs[i] * rk[0, key, :2] for i, key in enumerate(keys))
            head_outputs.append(value_up[head] @ latent)
        rows.append(torch.stack(head_outputs))
    heads = torch.stack(rows).unsqueeze(0)
    expected = ((torch.eye(2) + v @ u.T) @ heads).reshape(1, 4, 4) * gate.sigmoid()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    got = torch.autograd.grad(actual.square().sum(), (query, kv))
    want = torch.autograd.grad(expected.square().sum(), (rq, rk))
    for lhs, rhs in zip(got, want, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=1e-5)


@pytest.mark.cpu_only
def test_scores_match_explicit_head_and_key_loops_and_gradients() -> None:
    """Index scores and their gradients equal explicit loops over heads and keys."""
    torch.manual_seed(5561)
    query = torch.randn(1, 3, 2, 4, requires_grad=True)
    key = torch.randn(1, 5, 4, requires_grad=True)
    weights = torch.randn(1, 3, 2, requires_grad=True)
    ref_q, ref_k, ref_w = (x.detach().clone().requires_grad_() for x in (query, key, weights))
    actual = ernie5_5_index_scores(query, key, weights)
    expected = torch.stack(
        [
            torch.stack(
                [
                    sum((ref_q[0, i, head] * ref_k[0, j]).sum().relu() * ref_w[0, i, head] for head in range(2))
                    for j in range(5)
                ]
            )
            for i in range(3)
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    got = torch.autograd.grad(actual.square().sum(), (query, key, weights))
    want = torch.autograd.grad(expected.square().sum(), (ref_q, ref_k, ref_w))
    for lhs, rhs in zip(got, want, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=1e-6, rtol=1e-6)


# Merge of remote and local indices for the cuDNN path


@pytest.mark.cpu_only
def test_remote_prefix_and_window_match_dense_oracle_for_packed_documents() -> None:
    """For two documents and padding in one row, the merged indices equal the dense selection of the eager path."""
    torch.manual_seed(5582)
    docs = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1]])
    scores = torch.randn(1, 12, 12)
    selected = select_window_and_remote(scores, docs, window=2, remote_topk=2)[0]
    starts = torch.tensor([0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0], dtype=torch.int32)
    lengths = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 0], dtype=torch.int32)
    valid = docs[0] >= 0
    remote = torch.full((12, 2), -1, dtype=torch.int32)
    for query in range(12):
        candidates = torch.nonzero(selected[query] & (torch.arange(12) <= query - 2), as_tuple=True)[0]
        remote[query, : candidates.numel()] = candidates.to(torch.int32)
    union, prefix_lengths = merge_remote_and_window(remote, starts, lengths, valid, window=2)
    assert torch.equal(prefix_lengths, (lengths - 2).clamp_min(0))
    for query in range(12):
        got = union[query][union[query] >= 0].tolist()
        assert len(got) == len(set(got))
        assert set(got) == set(torch.nonzero(selected[query], as_tuple=True)[0].tolist())


@pytest.mark.cpu_only
def test_future_cross_document_and_padded_remote_indices_are_discarded() -> None:
    """Remote indices inside the window, after the query, in another document or of an invalid query become -1."""
    remote = torch.tensor([[0, 1, 2, 7], [0, 5, 6, 9], [0, 1, 2, 3]], dtype=torch.int32)
    starts = torch.tensor([0, 5, 0], dtype=torch.int32)
    lengths = torch.tensor([4, 3, 0], dtype=torch.int32)
    valid = torch.tensor([True, True, False])
    union, _ = merge_remote_and_window(remote, starts, lengths, valid, window=2)
    assert union[0].tolist() == [0, 1, -1, -1, 2, 3]
    assert union[1].tolist() == [-1, 5, -1, -1, 6, 7]
    assert torch.all(union[2] == -1)


@pytest.mark.cpu_only
def test_merge_remote_and_window_worked_example() -> None:
    """A hand-computed row: remote order is kept and window slots before the start of the document are -1."""
    # One row holds a document at offsets 0-3 and a document at offsets 4-9; the window is 3. The four queries
    # are the tokens at offsets 1, 9 and 8, and an invalid query with the metadata of offset 9.
    remote = torch.tensor([[0, -1], [5, 4], [6, 3], [4, 5]], dtype=torch.int32)
    starts = torch.tensor([0, 4, 4, 4], dtype=torch.int32)
    lengths = torch.tensor([2, 6, 5, 6], dtype=torch.int32)
    valid = torch.tensor([True, True, True, False])
    union, remote_lengths = merge_remote_and_window(remote, starts, lengths, valid, window=3)
    assert union.dtype == torch.int32 and remote_lengths.dtype == torch.int32
    # Offset 1: both causal tokens are inside the window, so the remote range is empty and one window slot is unused.
    assert union[0].tolist() == [-1, -1, -1, 0, 1]
    # Offset 9: the remote range is offsets 4-6; both remote keys are kept in their input order.
    assert union[1].tolist() == [5, 4, 7, 8, 9]
    # Offset 8: the remote range is offsets 4-5; 6 is inside the window and 3 belongs to the other document.
    assert union[2].tolist() == [-1, -1, 6, 7, 8]
    # An invalid query keeps no index; its remote length still follows from its causal length.
    assert union[3].tolist() == [-1, -1, -1, -1, -1]
    assert remote_lengths.tolist() == [0, 3, 2, 3]


@pytest.mark.parametrize("window", [False, True, 0, -1, 1.5])
@pytest.mark.cpu_only
def test_invalid_window_is_rejected(window: bool | int | float) -> None:
    """A window that is not a positive integer, booleans included, raises ``ValueError``."""
    remote = torch.full((1, 1), -1, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    valid = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="positive integer"):
        merge_remote_and_window(remote, starts, lengths, valid, window=window)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("remote_rank", "remote_indices must be int32"),
        ("remote_dtype", "remote_indices must be int32"),
        ("starts_shape", "per-query metadata matching"),
        ("lengths_shape", "per-query metadata matching"),
        ("valid_shape", "per-query metadata matching"),
        ("starts_dtype", "int32 starts/lengths and bool validity"),
        ("lengths_dtype", "int32 starts/lengths and bool validity"),
        ("valid_dtype", "int32 starts/lengths and bool validity"),
        ("starts_device", "must share a device"),
        ("lengths_device", "must share a device"),
        ("valid_device", "must share a device"),
    ],
)
@pytest.mark.cpu_only
def test_merge_remote_and_window_rejects_invalid_tensors(case: str, message: str) -> None:
    """A wrong rank, shape, dtype or device of the indices or the per-query metadata raises ``ValueError``."""
    remote = torch.full((2, 3), -1, dtype=torch.int32)
    starts = torch.zeros(2, dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    valid = torch.ones(2, dtype=torch.bool)
    if case == "remote_rank":
        remote = remote[0]
    elif case == "remote_dtype":
        remote = remote.long()
    elif case == "starts_shape":
        starts = torch.zeros(3, dtype=torch.int32)
    elif case == "lengths_shape":
        lengths = torch.ones(2, 1, dtype=torch.int32)
    elif case == "valid_shape":
        valid = torch.ones(1, dtype=torch.bool)
    elif case == "starts_dtype":
        starts = starts.long()
    elif case == "lengths_dtype":
        lengths = lengths.float()
    elif case == "valid_dtype":
        valid = valid.to(torch.int32)
    elif case == "starts_device":
        starts = torch.zeros(2, dtype=torch.int32, device="meta")
    elif case == "lengths_device":
        lengths = torch.ones(2, dtype=torch.int32, device="meta")
    else:
        valid = torch.ones(2, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match=message):
        merge_remote_and_window(remote, starts, lengths, valid, window=2)


# cuDNN core: checks that run before the kernels


@pytest.mark.cpu_only
def test_invalid_sparse_core_shapes_fail_before_backend() -> None:
    """Reject incompatible gate and latent shapes before launching cuDNN."""
    query = torch.empty(3, 2, 4)
    args = [
        query,
        torch.empty(3, 4),
        torch.empty(3, 1, 4),
        torch.empty(3, 4),
        torch.empty(3, 1),
        torch.empty(2),
        torch.empty(2, 3, 4),
        torch.empty(3, 5),
        torch.empty(2, 1),
        torch.empty(1, 2),
        torch.tensor([0, 3], dtype=torch.int32),
    ]
    opts = dict(max_seqlen=3, window=2, remote_topk=2, softmax_scale=0.5)
    with pytest.raises(ValueError, match="output gate"):
        cudnn_dsa_core(*args, **opts)
    args[1] = torch.empty(3, 5)
    with pytest.raises(ValueError, match="latent KV"):
        cudnn_dsa_core(*args, **opts)


@pytest.mark.cpu_only
def test_invalid_sparse_limits_fail_before_backend() -> None:
    """Reject an empty local window and remote selection budget."""
    args = [
        torch.empty(3, 2, 4),
        torch.empty(3, 4),
        torch.empty(3, 1, 4),
        torch.empty(3, 4),
        torch.empty(3, 1),
        torch.empty(2),
        torch.empty(2, 3, 4),
        torch.empty(3, 6),
        torch.empty(2, 1),
        torch.empty(1, 2),
        torch.tensor([0, 3], dtype=torch.int32),
    ]
    with pytest.raises(ValueError, match="positive window"):
        cudnn_dsa_core(*args, max_seqlen=3, window=0, remote_topk=2, softmax_scale=0.5)
    with pytest.raises(ValueError, match="positive window"):
        cudnn_dsa_core(*args, max_seqlen=3, window=2, remote_topk=0, softmax_scale=0.5)


@pytest.mark.cpu_only
def test_sparse_core_rejects_a_row_with_several_documents() -> None:
    """``cu_seqlens`` with more than two entries raises ``NotImplementedError`` before a kernel is called."""
    with pytest.raises(NotImplementedError, match="core does not support a row that holds several documents"):
        cudnn_dsa_core(*_cudnn_core_inputs([0, 1, 3]), max_seqlen=2, window=2, remote_topk=2, softmax_scale=0.5)


@pytest.mark.cpu_only
def test_sparse_core_returns_only_the_output() -> None:
    """With both kernels replaced by fixed results, the core returns one tensor of shape [tokens, heads * value_dim]."""
    module = "nemo_automodel.components.models.ernie5_5.dsa"
    no_remote_keys = torch.full((3, 1, 2), -1, dtype=torch.int32)
    with (
        patch(f"{module}.cudnn_indexer_topk", return_value=no_remote_keys) as topk,
        patch(f"{module}.cudnn_sparse_attention", return_value=torch.ones(3, 2, 4)) as attention,
    ):
        output = cudnn_dsa_core(*_cudnn_core_inputs([0, 3]), max_seqlen=3, window=2, remote_topk=2, softmax_scale=0.5)
    assert topk.call_count == 1 and attention.call_count == 1
    assert isinstance(output, torch.Tensor)
    # Four attended latent channels of one, an all-ones value projection, an identity head mix and a gate of 0.5.
    torch.testing.assert_close(output, torch.full((3, 6), 2.0), rtol=0, atol=0)


# Attention layer


@pytest.mark.cpu_only
def test_attention_is_causal_per_document_and_gives_the_indexer_no_gradient() -> None:
    """Outputs depend only on earlier tokens of the same document, and only the attention is trained."""
    torch.manual_seed(5567)
    model = _dsa_layer()
    x = torch.randn(1, 9, 8, requires_grad=True)
    docs = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, -1]])
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 0]])
    output, extra = model(x, document_ids=docs, position_ids=positions)
    assert extra is None
    assert output.shape == (1, 9, 8) and torch.isfinite(output).all()
    assert torch.equal(output[0, -1], torch.zeros_like(output[0, -1]))
    changed = x.detach().clone()
    changed[:, 6:] += 10
    changed_output, _ = model(changed, document_ids=docs, position_ids=positions)
    torch.testing.assert_close(output[:, :6], changed_output[:, :6])

    task = output.square().mean()
    indexer_params = tuple(model.indexer.parameters())
    assert all(grad is None for grad in torch.autograd.grad(task, indexer_params, retain_graph=True, allow_unused=True))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    task.backward()
    optimizer.step()
    after = dict(model.named_parameters())
    assert not torch.equal(before["o_proj.weight"], after["o_proj.weight"])
    for name, value in before.items():
        if name.startswith("indexer."):
            assert torch.equal(value, after[name])


@pytest.mark.cpu_only
def test_meta_initialization_and_eager_sequence_limit() -> None:
    """``init_weights`` fills every parameter after meta materialization; the eager path caps the sequence."""
    with torch.device("meta"):
        model = _dsa_layer()
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.init_weights()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert torch.count_nonzero(model.vha_postmix_V) == 0
    with pytest.raises(ValueError, match="max_eager_sequence"):
        model(
            torch.zeros(1, 17, 8),
            document_ids=torch.zeros(1, 17, dtype=torch.long),
            position_ids=torch.zeros(1, 17, dtype=torch.long),
        )


@pytest.mark.cpu_only
def test_backend_selects_the_path_without_changing_the_parameters() -> None:
    """``attn="cudnn"`` selects the sparse core; names, shapes and initial values are the same on both paths."""
    torch.manual_seed(11)
    eager = _dsa_layer()
    torch.manual_seed(11)
    sparse = _dsa_layer(BackendConfig(attn="cudnn"))
    assert not eager.uses_cudnn and sparse.uses_cudnn
    assert type(eager.fp32_params) is Ernie5_5DSAFP32Parameters
    assert type(sparse.fp32_params) is Ernie5_5DSASparseFP32Parameters
    for dense in ("eager", "torch", "sdpa"):
        assert not _dsa_layer(BackendConfig(attn=dense)).uses_cudnn
    eager_state, sparse_state = eager.state_dict(), sparse.state_dict()
    assert list(eager_state) == list(sparse_state)
    for name, value in eager_state.items():
        assert torch.equal(value, sparse_state[name])
    assert {"fp32_params.softmax_offset", "fp32_params.vha_postmix_U", "fp32_params.vha_postmix_V"} <= set(eager_state)


@pytest.mark.parametrize("attn", ["tilelang", "fa4", "magi", "te", "flex", "cudn"])
@pytest.mark.cpu_only
def test_backends_without_a_path_are_rejected(attn: str) -> None:
    """Any name other than ``cudnn`` and the dense backends raises instead of falling back to the eager path."""
    backend = BackendConfig(attn="eager")
    backend.attn = attn
    with pytest.raises(NotImplementedError, match=f"got attn='{attn}'"):
        _dsa_layer(backend)


@pytest.mark.cpu_only
def test_metadata_of_the_other_path_is_rejected() -> None:
    """Each path accepts only its own document metadata."""
    hidden = torch.zeros(1, 4, 8)
    positions = torch.arange(4).unsqueeze(0)
    documents = torch.zeros(1, 4, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    eager = _dsa_layer()
    with pytest.raises(ValueError, match="eager DSA path requires document_ids"):
        eager(hidden, position_ids=positions)
    with pytest.raises(ValueError, match="eager DSA path requires document_ids"):
        eager(hidden, position_ids=positions, document_ids=documents, cu_seqlens=cu_seqlens)
    with pytest.raises(ValueError, match="eager DSA path requires document_ids"):
        eager(hidden, position_ids=positions, document_ids=documents, max_seqlen=4)
    sparse = _dsa_layer(BackendConfig(attn="cudnn"))
    with pytest.raises(ValueError, match="cuDNN DSA path requires cu_seqlens and max_seqlen"):
        sparse(hidden, position_ids=positions, document_ids=documents)
    with pytest.raises(ValueError, match="cuDNN DSA path requires cu_seqlens and max_seqlen"):
        sparse(hidden, position_ids=positions, cu_seqlens=cu_seqlens)
    with pytest.raises(ValueError, match="cuDNN DSA path requires cu_seqlens and max_seqlen"):
        sparse(
            hidden,
            position_ids=positions,
            document_ids=documents,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )
    with pytest.raises(ValueError, match="one unpadded packed row"):
        sparse(
            torch.zeros(2, 4, 8),
            position_ids=positions.expand(2, -1),
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
        )


@pytest.mark.cpu_only
def test_several_documents_in_one_row_are_rejected_on_the_cudnn_path() -> None:
    """``cu_seqlens`` with more than two entries raises ``NotImplementedError`` before the row shape is checked."""
    sparse = _dsa_layer(BackendConfig(attn="cudnn"))
    two_documents = torch.tensor([0, 2, 5], dtype=torch.int32)
    positions = torch.tensor([[0, 1, 0, 1, 2]])
    with pytest.raises(NotImplementedError, match="path does not support a row that holds several documents"):
        sparse(torch.zeros(1, 5, 8), position_ids=positions, cu_seqlens=two_documents, max_seqlen=3)
    # A batch of two rows is invalid as well; the check of the document count runs first.
    with pytest.raises(NotImplementedError, match="cu_seqlens must have two entries, got 3"):
        sparse(
            torch.zeros(2, 5, 8),
            position_ids=positions.expand(2, -1),
            cu_seqlens=two_documents,
            max_seqlen=3,
        )


@pytest.mark.parametrize(
    ("values", "dtype", "max_seqlen", "error", "message"),
    [
        ([[0, 4]], torch.int32, 4, ValueError, "one-dimensional"),
        ([4], torch.int32, 4, ValueError, "one-dimensional"),
        ([0, 4], torch.int64, 4, TypeError, "must be int32"),
        ([1, 4], torch.int32, 3, ValueError, "start at zero"),
        ([0, 3], torch.int32, 3, ValueError, "end at total_key_tokens"),
        ([0, 5], torch.int32, 5, ValueError, "end at total_key_tokens"),
        ([0, 4], torch.int32, 3, ValueError, "max_seqlen must equal"),
    ],
)
@pytest.mark.cpu_only
def test_malformed_cu_seqlens_and_max_seqlen_are_rejected_before_the_kernels(
    values: list[int] | list[list[int]], dtype: torch.dtype, max_seqlen: int, error: type[Exception], message: str
) -> None:
    """``cu_seqlens`` other than a one-dimensional int32 ``[0, sequence]``, or a wrong ``max_seqlen``, raises on CPU."""
    sparse = _dsa_layer(BackendConfig(attn="cudnn"))
    with pytest.raises(error, match=message):
        sparse(
            torch.zeros(1, 4, 8),
            position_ids=torch.arange(4).unsqueeze(0),
            cu_seqlens=torch.tensor(values, dtype=dtype),
            max_seqlen=max_seqlen,
        )


@pytest.mark.cpu_only
def test_cudnn_path_requires_a_positive_remote_topk_at_construction() -> None:
    """``remote_topk=0`` builds a window-only layer on the eager path and raises ``ValueError`` on the cuDNN path."""
    for backend in (None, BackendConfig(attn="eager"), BackendConfig(attn="torch"), BackendConfig(attn="sdpa")):
        eager = Ernie5_5DSAAttention(*_GEOMETRY, window=2, remote_topk=0, dtype=torch.float32, backend=backend)
        assert not eager.uses_cudnn and eager.remote_topk == 0
    output, _ = eager(
        torch.ones(1, 5, 8),
        document_ids=torch.zeros(1, 5, dtype=torch.long),
        position_ids=torch.arange(5).unsqueeze(0),
    )
    assert output.shape == (1, 5, 8) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match="positive remote top-k; got remote_topk=0"):
        Ernie5_5DSAAttention(
            *_GEOMETRY, window=2, remote_topk=0, dtype=torch.float32, backend=BackendConfig(attn="cudnn")
        )


@pytest.mark.parametrize("attn", ["eager", "cudnn"])
@pytest.mark.parametrize("window", [0, -1])
@pytest.mark.cpu_only
def test_non_positive_window_is_rejected_at_construction_on_both_paths(attn: str, window: int) -> None:
    """A window below one raises ``ValueError`` in the constructor, whichever path the backend selects."""
    with pytest.raises(ValueError, match="Invalid rotary dimension, window"):
        Ernie5_5DSAAttention(
            *_GEOMETRY, window=window, remote_topk=2, dtype=torch.float32, backend=BackendConfig(attn=attn)
        )


@pytest.mark.parametrize("init_std", [float("inf"), float("nan"), 0.0, -0.02])
@pytest.mark.cpu_only
def test_layer_init_weights_rejects_an_invalid_std_without_changing_parameters(init_std: float) -> None:
    """A non-finite or non-positive ``init_std`` raises ``ValueError`` before any parameter is written."""
    model = _dsa_layer()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="init_std must be finite and positive"):
        model.init_weights(init_std)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("attn", ["eager", "cudnn"])
@pytest.mark.parametrize("name", ["softmax_offset", "vha_postmix_U", "vha_postmix_V"])
@pytest.mark.cpu_only
def test_sink_and_virtual_head_parameters_must_stay_fp32_on_both_paths(attn: str, name: str) -> None:
    """A sink or virtual-head parameter that is not FP32 raises ``RuntimeError`` before either core is called."""
    layer = _dsa_layer(BackendConfig(attn=attn))
    holder = layer.fp32_params
    setattr(holder, name, nn.Parameter(getattr(holder, name).detach().to(torch.bfloat16)))
    if attn == "cudnn":
        metadata = {"cu_seqlens": torch.tensor([0, 4], dtype=torch.int32), "max_seqlen": 4}
    else:
        metadata = {"document_ids": torch.zeros(1, 4, dtype=torch.long)}
    with pytest.raises(RuntimeError, match="sink and VHA parameters must remain FP32"):
        layer(torch.zeros(1, 4, 8), position_ids=torch.arange(4).unsqueeze(0), **metadata)


# Indexer freeze


@pytest.mark.cpu_only
def test_freeze_indexer_params_freezes_only_indexer_parameters() -> None:
    """Every parameter of every indexer in the model is frozen, and no other parameter changes its flag."""
    model = nn.ModuleDict({"first": _dsa_layer(), "second": _dsa_layer(), "head": nn.Linear(8, 8)})
    model["head"].bias.requires_grad_(False)
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert all(flags[name] for name in flags if ".indexer." in name)
    freeze_ernie5_5_indexer_params(model)
    indexer_names = [name for name in flags if ".indexer." in name]
    assert len(indexer_names) == 2 * len(list(model["first"].indexer.parameters())) == 10
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (False if name in indexer_names else flags[name]), name
    standalone = Ernie5_5Indexer(8, 6, 4, 4, dtype=torch.float32)
    freeze_ernie5_5_indexer_params(standalone)
    assert not any(parameter.requires_grad for parameter in standalone.parameters())


@pytest.mark.cpu_only
def test_freeze_indexer_params_sets_flags_once_and_can_be_repeated() -> None:
    """The freeze does not persist through ``requires_grad_(True)``; a second call restores it."""
    model = _dsa_layer()
    freeze_ernie5_5_indexer_params(model)
    freeze_ernie5_5_indexer_params(model)
    assert not any(parameter.requires_grad for parameter in model.indexer.parameters())
    model.requires_grad_(True)
    assert all(parameter.requires_grad for parameter in model.parameters())
    freeze_ernie5_5_indexer_params(model)
    assert not any(parameter.requires_grad for parameter in model.indexer.parameters())
    assert all(
        parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("indexer.")
    )
