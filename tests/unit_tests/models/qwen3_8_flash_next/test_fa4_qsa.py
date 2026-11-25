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

"""CPU coverage of optional FA4 dispatch and tensor validation; CUDA coverage of the Triton preprocessing."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.attention import fa4 as shared_fa4
from nemo_automodel.components.models.qwen3_8_flash_next import fa4_qsa
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    gathered_qsa_gqa_attention,
    qsa_gqa_attention,
    select_qsa_cuda_kernel,
)

_CUDA_AND_TRITON = pytest.mark.skipif(
    not torch.cuda.is_available() or not fa4_qsa.HAVE_TRITON,
    reason="the QSA preprocessing kernels need CUDA and Triton",
)
BLOCK_KEYS = pytest.mark.parametrize("forward_keys, backward_keys", [(80, 64), (128, 128)], ids=["sm90", "sm100"])


def _reference_kinds(membership: torch.Tensor, block_keys: int) -> torch.Tensor:
    """Classify 128-query x block_keys tiles of the membership table eagerly.

    Args:
        membership: Uint8 tensor [batch, queries, keys] of zeros and ones.
        block_keys: Keys per tile.

    Returns:
        Int32 tensor [batch, ceil(queries / 128), ceil(keys / block_keys)]: absent (0), partial (1)
        or full (2); tiles padded past the last query or key are never full.
    """
    batch, queries, keys = membership.shape
    nq, nk = (queries + 127) // 128, (keys + block_keys - 1) // block_keys
    tiles = F.pad(membership, (0, nk * block_keys - keys, 0, nq * 128 - queries)).reshape(batch, nq, 128, nk, -1)
    return (tiles != 0).any(dim=(2, 4)).to(torch.int32) + (tiles == 1).all(dim=(2, 4)).to(torch.int32)


def _reference_lists(kinds: torch.Tensor) -> list[torch.Tensor]:
    """Compact the partial and full columns of every row eagerly.

    Args:
        kinds: Int32 tensor [batch, rows, columns] of absent (0), partial (1) or full (2).

    Returns:
        Int32 partial counts [batch, 1, rows], partial columns [batch, 1, rows, columns] in ascending
        order padded with ``columns``, then full counts and columns with the same layouts.
    """
    columns = torch.arange(kinds.shape[-1], device=kinds.device, dtype=torch.int32)
    lists = []
    for selected in (kinds == 1, kinds == 2):
        lists.append(selected.sum(-1, dtype=torch.int32).unsqueeze(1))
        lists.append(torch.where(selected, columns, kinds.shape[-1]).sort(-1).values.unsqueeze(1))
    return lists


def _reference_preprocess(
    routes: torch.Tensor, kv_length: int, forward_keys: int, backward_keys: int
) -> list[torch.Tensor]:
    """Eager oracle for fa4_qsa._preprocess (the former PyTorch implementation).

    Args:
        routes: Signed int32/int64 tensor [batch, queries, width]; invalid IDs are ignored.
        kv_length: Number of physical keys.
        forward_keys: Keys per forward block.
        backward_keys: Keys per backward block.

    Returns:
        Uint8 membership [batch, queries, kv_length] with rows padded to a multiple of 32 bytes, then
        the forward and the backward lists in the layout of _reference_lists.
    """
    batch, queries, _ = routes.shape
    valid = (routes >= 0) & (routes < kv_length)
    storage = torch.zeros(batch, queries, (kv_length + 1 + 31) // 32 * 32, dtype=torch.uint8, device=routes.device)
    storage.scatter_(-1, torch.where(valid, routes, kv_length).long(), 1)
    membership = storage[..., :kv_length]
    forward = _reference_lists(_reference_kinds(membership, forward_keys))
    backward = _reference_lists(_reference_kinds(membership, backward_keys).transpose(1, 2))
    return [membership, *forward, *backward]


def test_cpu_fa4_dispatch_keeps_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(4)
    inputs = [torch.randn(1, 7, heads, 4, requires_grad=True) for heads in (4, 2, 2)]
    routes = torch.arange(7).view(1, 1, 7).expand(1, 7, 7)
    monkeypatch.setattr(fa4_qsa, "import_fa4_modules", lambda *args: pytest.fail("CPU must not load FA4"))
    actual = qsa_gqa_attention(*inputs, routes, cuda_kernel=select_qsa_cuda_kernel("fa4"))
    expected = gathered_qsa_gqa_attention(*inputs, routes)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, inputs, dy)
    refs = torch.autograd.grad(expected, inputs, dy)
    for result, reference in zip((actual, *grads), (expected, *refs)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fa4_qsa._load_fa4.cache_clear()
    monkeypatch.setattr(shared_fa4, "safe_import", lambda name: (False, None))
    with pytest.raises(ImportError, match="FlashAttention 4.*could not import flash_attn.cute.interface"):
        fa4_qsa._load_fa4()
    fa4_qsa._load_fa4.cache_clear()


@pytest.mark.parametrize(
    "case", ["rank", "batch", "kv_shape", "head_dim", "gqa", "empty", "routes", "route_dtype", "dtype", "cpu"]
)
def test_fa4_contract_rejects_invalid_inputs(case: str) -> None:
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
        fa4_qsa.fa4_sparse_gqa_attention(q, k, v, routes)


@_CUDA_AND_TRITON
def test_byte_membership_preserves_route_set() -> None:
    routes = torch.tensor(
        [[[0, 0, -1, 1 << 40, 31, 32, 64], [-1] * 7], [[1, 2, 3, 65, -3, 2, 0], [64] * 7]],
        dtype=torch.int64,
        device="cuda",
    )
    raw = fa4_qsa._preprocess(routes, 65, 80, 64)
    expected = torch.zeros(2, 2, 65, dtype=torch.uint8, device="cuda")
    expected[0, 0, [0, 31, 32, 64]] = 1
    expected[1, 0, [0, 1, 2, 3]] = 1
    expected[1, 1, 64] = 1
    torch.testing.assert_close(raw[0], expected, rtol=0, atol=0)
    assert raw[0].stride(1) % 16 == 0
    assert all(t.is_contiguous() for t in raw[1:])


@_CUDA_AND_TRITON
@BLOCK_KEYS
def test_block_lists_distinguish_full_partial_and_empty(forward_keys: int, backward_keys: int) -> None:
    # One key past the forward block: rows 0-127 fill the first forward and backward block and touch the next.
    kv_length = forward_keys + 1
    routes = torch.arange(kv_length, device="cuda").expand(1, 129, kv_length).clone()
    routes[:, 128] = -1
    raw = fa4_qsa._preprocess(routes, kv_length, forward_keys, backward_keys)
    mask, pc, pi, fc, fi, rpc, rpi, rfc, rfi = raw
    assert mask[:, 128].count_nonzero() == 0
    assert pc.tolist() == [[[1, 0]]]
    assert fc.tolist() == [[[1, 0]]]
    assert pi[0, 0, 0, 0] == 1
    assert fi[0, 0, 0, 0] == 0
    assert rpc.tolist() == [[[0, 1]]]
    assert rfc.tolist() == [[[1, 0]]]
    assert rpi[0, 0, 1, 0] == 0
    assert rfi[0, 0, 0, 0] == 0


@pytest.mark.parametrize(
    "capability, expected",
    [((9, 0), (80, 64)), ((10, 0), (128, 128)), ((10, 3), (128, 128)), ((10, 7), (128, 128)), ((11, 0), (128, 128))],
)
def test_block_keys_match_fa4_tiles_per_architecture(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: tuple[int, int]
) -> None:
    monkeypatch.setattr(fa4_qsa.torch.cuda, "get_device_capability", lambda device: capability)
    assert fa4_qsa._block_keys(torch.device("cuda", 0)) == expected


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (12, 0)])
def test_block_keys_reject_gpus_without_fa4_qsa(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(fa4_qsa.torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(RuntimeError, match=f"SM90 and SM100-family.*got SM{capability[0]}{capability[1]}"):
        fa4_qsa._block_keys(torch.device("cuda", 0))


def test_preprocess_without_triton_raises_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa4_qsa, "HAVE_TRITON", False)
    with pytest.raises(ImportError, match="requires Triton"):
        fa4_qsa._preprocess(torch.zeros(1, 1, 1, dtype=torch.int32), 1, 80, 64)


@_CUDA_AND_TRITON
@BLOCK_KEYS
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_triton_preprocess_matches_the_eager_reference(
    forward_keys: int, backward_keys: int, dtype: torch.dtype
) -> None:
    """Bit-identical to the former eager implementation, including duplicate, negative and out-of-range IDs,
    strided route rows, queries and keys that end inside a block, and causal rows whose tiles are full."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    cases = []
    for batch, queries, kv_length, width in ((1, 1, 1, 1), (2, 33, 65, 41), (1, 127, 263, 129), (3, 300, 700, 64)):
        shape = (batch, queries, width + 3)
        wide = torch.randint(-3, kv_length + 4, shape, generator=generator, device="cuda", dtype=dtype)
        cases.append((wide[..., :width], kv_length))  # route rows strided like the trimmed indexer output
    causal = torch.arange(600, device="cuda", dtype=dtype)
    cases.append((torch.where(causal[None, :] <= causal[:, None], causal[None, :], -1).unsqueeze(0), 600))
    # Indexer-shaped routes: aligned 4-key blocks (stored as words), an unaligned consecutive group, a
    # duplicated block, and blocks that straddle the last key (stored as bytes).
    blocks = torch.randint(0, 250, (2, 40, 16), generator=generator, device="cuda", dtype=dtype)
    grouped = (blocks[..., None] * 4 + torch.arange(4, device="cuda", dtype=dtype)).flatten(-2)
    grouped[..., 4:8] += 2
    grouped[..., 8:12] = grouped[..., :4]
    cases.append((grouped, 998))
    for routes, kv_length in cases:
        actual = fa4_qsa._preprocess(routes, kv_length, forward_keys, backward_keys)
        expected = _reference_preprocess(routes, kv_length, forward_keys, backward_keys)
        assert actual[0].stride() == expected[0].stride()
        for result, reference in zip(actual, expected, strict=True):
            assert result.shape == reference.shape and result.dtype == reference.dtype
            assert torch.equal(result, reference)
