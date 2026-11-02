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

"""SM90 FA4 QSA parity against an independent dense FP64 attention oracle."""

import functools

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.qwen3_8_flash_next.fa4_qsa import fa4_sparse_gqa_attention
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import qsa_gqa_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="FA4 QSA implements SM90 CUDA kernels",
)


def _assert_parity(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """Compare tensors with matching semantic shapes, using FP64 relative L2.

    Args:
        actual: CUDA tensor with arbitrary shape and BF16 or FP32 dtype.
        reference: Independent FP64 CUDA reference of the same shape.
    """
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
    error = (actual.double() - reference.double()).norm()
    norm = reference.double().norm()
    # 0.4% relative L2 is the pre-existing BF16 attention validation bound.
    assert error <= 0.004 * norm + 1e-8, (float(error), float(norm))


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, routes: torch.Tensor, *, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate dense FP64 GQA with exact route-set membership.

    Args:
        q: FP64 CUDA [batch, queries, query_heads, 256].
        k: FP64 CUDA [batch, keys, kv_heads, 256].
        v: FP64 CUDA tensor with key's layout.
        routes: Signed CUDA [batch, queries, routes] in physical key coordinates.
        scale: QK score multiplier.

    Returns:
        FP64 output [batch, queries, query_heads, 256] and boolean membership
        [batch, queries, keys]. Neither output aliases an input.
    """
    valid = (routes >= 0) & (routes < k.shape[1])
    counts = torch.zeros(*routes.shape[:2], k.shape[1], device=q.device, dtype=torch.int32)
    counts.scatter_add_(-1, routes.long().clamp(0, k.shape[1] - 1), valid.int())
    mask = counts.bool()
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask[:, None],
        enable_gqa=True,
        scale=scale,
    ).transpose(1, 2)
    return out, mask


@pytest.mark.parametrize(
    "seed,batch,sq,sk,width",
    [(11, 1, 1, 1, 1), (19, 1, 17, 31, 17), (23, 2, 33, 65, 41), (29, 1, 127, 263, 129), (31, 1, 256, 257, 2051)],
)
def test_output_and_all_gradients(seed: int, batch: int, sq: int, sk: int, width: int) -> None:
    torch.manual_seed(seed)
    q = torch.randn(batch, sq, 24, 512, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()
    k = torch.randn(batch, sk, 2, 512, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    dy = torch.randn_like(q)
    routes = torch.randint(-3, sk + 4, (batch, sq, width), device="cuda", dtype=torch.int64)
    if sq > 1:
        routes[:, 0] = -1
        routes[:, 1] = sk + 1
        routes[:, 2] = 0
        routes[:, -1, :2] = torch.tensor([2**40, -(2**40)], device="cuda")
    if sq >= 127:
        # Two documents with a physical K/V gap; no query may cross documents.
        routes[:, 3 : sq // 2] = routes[:, 3 : sq // 2].remainder(sk // 3)
        routes[:, sq // 2 :] = routes[:, sq // 2 :].remainder(sk - sk * 2 // 3) + sk * 2 // 3
    scale = 0.125
    out = qsa_gqa_attention(q, k, v, routes, backend="cute", softmax_scale=scale)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    inputs64 = [x.detach().double().requires_grad_() for x in (q, k, v)]
    ref, mask = _reference(*inputs64, routes, scale=scale)
    ref_grads = torch.autograd.grad(ref, inputs64, dy.double())
    for actual, expected in zip((out, *grads), (ref, *ref_grads)):
        _assert_parity(actual, expected)
    empty = ~mask.any(-1)
    assert torch.count_nonzero(out[empty]) == 0
    assert torch.count_nonzero(grads[0][empty]) == 0
    unused = ~mask.any(1)
    assert torch.count_nonzero(grads[1][unused]) == 0
    assert torch.count_nonzero(grads[2][unused]) == 0

    call = functools.partial(fa4_sparse_gqa_attention, softmax_scale=scale)
    checkpointed = checkpoint(call, q, k, v, routes, use_reentrant=False)
    ac_grads = torch.autograd.grad(checkpointed, (q, k, v), dy)
    for actual, expected in zip((checkpointed, *ac_grads), (ref, *ref_grads)):
        _assert_parity(actual, expected)


@pytest.mark.parametrize("empty_tile", [False, True])
def test_full_tiles_and_empty_tiles(empty_tile: bool) -> None:
    torch.manual_seed(51)
    q = torch.randn(1, 256, 24, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 256, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    routes = torch.arange(256, device="cuda", dtype=torch.int32).expand(1, 256, 256).clone()
    if empty_tile:
        routes[:, 128:] = -1
    dy = torch.randn_like(q)
    out = fa4_sparse_gqa_attention(q, k, v, routes)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    inputs64 = [x.detach().double().requires_grad_() for x in (q, k, v)]
    ref, _ = _reference(*inputs64, routes, scale=256**-0.5)
    refs = torch.autograd.grad(ref, inputs64, dy.double())
    for actual, expected in zip((out, *grads), (ref, *refs)):
        _assert_parity(actual, expected)
    if empty_tile:
        assert torch.count_nonzero(out[:, 128:]) == 0
        assert torch.count_nonzero(grads[0][:, 128:]) == 0


def test_local_query_slices_and_current_stream() -> None:
    torch.manual_seed(61)
    q = torch.randn(1, 257, 24, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 271, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    routes = torch.randint(-1, 271, (1, 257, 67), device="cuda", dtype=torch.int32)
    dy = torch.randn_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = fa4_sparse_gqa_attention(q[:, :129], k, v, routes[:, :129])
        second = fa4_sparse_gqa_attention(q[:, 129:], k, v, routes[:, 129:])
        split = torch.cat((first, second), dim=1)
        split_grads = torch.autograd.grad(split, (q, k, v), dy)
    torch.cuda.current_stream().wait_stream(stream)
    inputs64 = [x.detach().double().requires_grad_() for x in (q, k, v)]
    ref, _ = _reference(*inputs64, routes, scale=256**-0.5)
    refs = torch.autograd.grad(ref, inputs64, dy.double())
    for actual, expected in zip((split, *split_grads), (ref, *refs)):
        _assert_parity(actual, expected)


def test_deterministic_mode_rejected() -> None:
    q = torch.zeros(1, 1, 2, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    routes = torch.zeros(1, 1, 1, device="cuda", dtype=torch.int32)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="atomic reductions"):
            fa4_sparse_gqa_attention(q, kv, kv, routes)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_4k_routes_with_sampled_fp64_backward() -> None:
    torch.manual_seed(73)
    q = torch.randn(1, 4096, 24, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4096, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    routes = torch.arange(4096, device="cuda")[:, None] - torch.arange(2051, device="cuda")[None, :]
    routes[:, 2048:] = -1
    routes = routes.unsqueeze(0).int()
    rows = torch.tensor([0, 31, 63, 127, 128, 1023, 2047, 2050, 2051, 3071, 4095], device="cuda")
    dy = torch.zeros_like(q)
    dy[:, rows] = torch.randn_like(dy[:, rows])
    out = fa4_sparse_gqa_attention(q, k, v, routes)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    inputs64 = [
        q[:, rows].detach().double().requires_grad_(),
        k.detach().double().requires_grad_(),
        v.detach().double().requires_grad_(),
    ]
    ref, _ = _reference(*inputs64, routes[:, rows], scale=256**-0.5)
    refs = torch.autograd.grad(ref, inputs64, dy[:, rows].double())
    for actual, expected in zip((out[:, rows], grads[0][:, rows], grads[1], grads[2]), (ref, *refs)):
        _assert_parity(actual, expected)


def test_model_qsa_layer_parameter_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy

    from nemo_automodel.components.models.qwen3_8_flash_next import qsa
    from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
    from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
    from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextQSAAttention

    torch.manual_seed(83)
    config = Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        layer_types=["full_attention"],
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        max_position_embeddings=4096,
        dtype="bfloat16",
        partial_rotary_factor=0.25,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 0.25},
    )
    backend = Qwen3_8_FlashNextBackendConfig(
        attn="cute", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch"
    )
    layer = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=backend).cuda()
    layer.init_weights(buffer_device=torch.device("cuda"))
    reference = copy.deepcopy(layer)
    x = torch.randn(1, 33, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    freqs = torch.cat((torch.ones(1, 33, 32), torch.zeros(1, 33, 32)), dim=-1).cuda()
    actual = layer(x, freqs_cis=freqs)
    with monkeypatch.context() as patch:
        patch.setattr(qsa, "fa4_sparse_gqa_attention", qsa.gathered_qsa_gqa_attention)
        expected = reference(xr, freqs_cis=freqs)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    # Projections add BF16 rounding beyond the standalone attention contraction.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    for name, result, target in [("input", x.grad, xr.grad)] + [
        (name, p.grad, dict(reference.named_parameters())[name].grad) for name, p in layer.named_parameters()
    ]:
        if result is None or target is None:
            assert result is None and target is None, name
            continue
        assert torch.isfinite(result).all(), name
        relative = (result.float() - target.float()).norm() / target.float().norm().clamp_min(1e-12)
        assert relative < 0.015, (name, float(relative))
