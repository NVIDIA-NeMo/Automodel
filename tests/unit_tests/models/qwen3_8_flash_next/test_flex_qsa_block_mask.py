# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The BlockMask built directly from the membership table equals torch's create_block_mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from nemo_automodel.components.models.qwen3_8_flash_next.flex_qsa import (
    _routes_to_membership,
    build_flex_qsa_mask,
)

_BLOCK_MASK_TENSORS = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)


def _reference_block_mask(membership: torch.Tensor) -> BlockMask:
    """torch's own path: evaluate an element-wise mask over ``[B, S_q, KV]`` and reduce it."""
    batch_size, query_length, kv_length = membership.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return membership[b, q_idx, kv_idx]

    return create_block_mask(mask_mod, B=batch_size, H=None, Q_LEN=query_length, KV_LEN=kv_length, device="cpu")


def _random_routes(batch: int, query_length: int, kv_length: int, width: int, seed: int) -> torch.Tensor:
    """Route IDs ``[B, S_q, width]`` in ``[-1, kv_length)``: causal prefix rows dense, later rows random, some empty."""
    g = torch.Generator().manual_seed(seed)
    routes = torch.randint(-1, kv_length, (batch, query_length, width), generator=g, dtype=torch.int32)
    # Dense causal prefix: the first rows see every earlier token, producing full and partial tiles.
    for q in range(min(query_length, width)):
        routes[:, q, : q + 1] = torch.arange(q + 1, dtype=torch.int32)
        routes[:, q, q + 1 :] = -1
    routes[:, -3:] = -1  # empty rows fall back to K/V row zero
    return routes


@pytest.mark.runtime_budget(
    20,
    hard_timeout=120,
    reason="the reference create_block_mask evaluates mask_mod through vmap over the full [B, S_q, KV] grid on CPU",
)
@pytest.mark.parametrize(
    ("batch", "query_length", "kv_length"),
    [(2, 256, 256), (1, 200, 300), (2, 130, 512), (1, 64, 64)],
)
def test_direct_block_mask_matches_create_block_mask(batch: int, query_length: int, kv_length: int) -> None:
    routes = _random_routes(batch, query_length, kv_length, width=160, seed=query_length + kv_length)
    membership, _ = _routes_to_membership(routes, kv_length)
    expected = _reference_block_mask(membership)

    actual = build_flex_qsa_mask(routes, kv_length=kv_length, device=torch.device("cpu")).block_mask

    assert actual.BLOCK_SIZE == expected.BLOCK_SIZE == (128, 128)
    assert actual.seq_lengths == expected.seq_lengths == (query_length, kv_length)
    for name in _BLOCK_MASK_TENSORS:
        got, want = getattr(actual, name), getattr(expected, name)
        assert (got is None) == (want is None), name
        if want is not None:
            assert got.dtype == want.dtype, name
            assert torch.equal(got, want), name
    # The attached mask_mod must agree with the membership table inside the valid range.
    q = torch.tensor([0, query_length - 1, query_length // 2])
    kv = torch.tensor([0, kv_length - 1, kv_length // 3])
    assert torch.equal(actual.mask_mod(torch.tensor(0), torch.tensor(0), q, kv), membership[0, q, kv])


def test_direct_block_mask_separates_full_and_partial_tiles() -> None:
    """A fully dense 128x128 tile is reported as full, a half-covered tile as partial."""
    kv_length = 256
    routes = torch.full((1, 256, 256), -1, dtype=torch.int32)
    routes[0, :128, :128] = torch.arange(128, dtype=torch.int32)  # q tile 0 x kv tile 0: full
    routes[0, 128:, :64] = torch.arange(128, 192, dtype=torch.int32)  # q tile 1 x kv tile 1: partial
    mask = build_flex_qsa_mask(routes, kv_length=kv_length, device=torch.device("cpu")).block_mask
    assert mask.full_kv_num_blocks.tolist() == [[[1, 0]]]
    assert mask.kv_num_blocks.tolist() == [[[0, 1]]]
    assert mask.full_kv_indices[0, 0, 0, 0].item() == 0
    assert mask.kv_indices[0, 0, 1, 0].item() == 1
