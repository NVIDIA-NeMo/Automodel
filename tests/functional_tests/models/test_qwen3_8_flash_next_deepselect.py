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

"""CUDA parity for the optional DeepSelect QSA selector."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.flex_qsa import flex_sparse_gqa_attention
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    Qwen3_8_FlashNextQSAIndexer,
    select_qsa_token_ids,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _require_deepselect() -> None:
    pytest.importorskip("deep_select")


@pytest.mark.parametrize("global_length,offset,query_length,tied", [(4096, 2040, 72, False), (2055, 2048, 7, True)])
def test_deepselect_routes_match_torch_with_padding_offset_and_tail(
    global_length: int, offset: int, query_length: int, tied: bool
) -> None:
    torch.manual_seed(123)
    q = torch.randn(2, query_length, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, global_length // 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    if tied:
        q.zero_()
    lengths = torch.tensor([global_length, offset + 3], device="cuda")
    kwargs = dict(
        token_budget=2048,
        compress_ratio=4,
        query_position_offset=offset,
        global_sequence_length=global_length,
        query_chunk_size=17,
    )
    reference = select_qsa_token_ids(q, k, lengths, **kwargs)
    result = select_qsa_token_ids(q, k, lengths, topk_backend="deepselect", **kwargs)
    if not tied:
        torch.testing.assert_close(result.sort(-1).values, reference.sort(-1).values, rtol=0, atol=0)
    # Tied scores may choose different valid top-k sets; neither backend
    # promises a common tie-breaking order.
    for row in range(query_length):
        ids = result[0, row]
        ids = ids[ids >= 0]
        assert ids.unique().numel() == ids.numel()
        assert bool((ids <= offset + row).all())
        assert ids.numel() == min((offset + row + 1) // 4, 512) * 4 + (offset + row + 1) % 4
    assert bool((result[1, 3:] == -1).all())
    for row in range(query_length):
        pos = offset + row
        if (pos + 1) // 4 <= 512:
            torch.testing.assert_close(result[0, row], reference[0, row], rtol=0, atol=0)
        tail = (pos + 1) % 4
        if tail:
            slot = min((pos + 1) // 4, 512) * 4
            torch.testing.assert_close(result[0, row, slot : slot + tail], reference[0, row, slot : slot + tail])
    repeated = select_qsa_token_ids(q, k, lengths, topk_backend="deepselect", **kwargs)
    torch.testing.assert_close(repeated.sort(-1).values, result.sort(-1).values, rtol=0, atol=0)


def test_deepselect_packed_indexer_propagates_backend() -> None:
    cfg = SimpleNamespace(
        hidden_size=32,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        qsa_indexer_query_chunk_size=128,
        torch_dtype=torch.bfloat16,
        rms_norm_eps=1e-6,
    )
    torch.manual_seed(17)
    model = Qwen3_8_FlashNextQSAIndexer(
        cfg, Qwen3_8_FlashNextBackendConfig(linear="torch", attn="flex", qsa_topk="deepselect")
    ).cuda()
    model.init_weights()
    hidden = torch.randn(1, 2072, 32, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([0, 2055, 2072], device="cuda", dtype=torch.int32)
    freqs = torch.cat((torch.ones(1, 2072, 8), torch.zeros(1, 2072, 8)), -1).cuda()
    selected = model(hidden, freqs_cis=freqs, cu_seqlens=boundaries)
    model.topk_backend = "torch"
    reference = model(hidden, freqs_cis=freqs, cu_seqlens=boundaries)
    torch.testing.assert_close(selected.sort(-1).values, reference.sort(-1).values, rtol=0, atol=0)
    valid = selected[0, 2055:] >= 0
    assert bool((selected[0, 2055:][valid] >= 2055).all())


def test_deepselect_flex_output_gradients_and_checkpoint() -> None:
    torch.manual_seed(31)
    iq = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(1, 1024, 1, 128, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([2060], device="cuda")
    opts = dict(token_budget=2048, compress_ratio=4, query_position_offset=2040, global_sequence_length=4096)
    q = torch.randn(1, 32, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4096, 2, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    upstream = torch.randn_like(q)
    records = []
    outputs = []
    gradients = []
    for backend, use_ac in [("torch", False), ("deepselect", True)]:
        inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]

        def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            """Evaluate selected attention and record routes across recomputation.

            Args:
                query: BF16 CUDA [1, 32, 24, 256].
                key: BF16 CUDA [1, 4096, 2, 256].
                value: BF16 CUDA with key's layout.

            Returns:
                BF16 CUDA output with query's layout.
            """
            routes = select_qsa_token_ids(iq, ik, lengths, topk_backend=backend, **opts)
            records.append(routes.detach().clone())
            return flex_sparse_gqa_attention(query, key, value, routes)

        output = checkpoint(run, *inputs, use_reentrant=False) if use_ac else run(*inputs)
        (output.float() * upstream.float()).sum().backward()
        outputs.append(output.detach())
        gradients.append([x.grad.detach().clone() for x in inputs])
    assert len(records) >= 3
    for route in records[1:]:
        torch.testing.assert_close(route.sort(-1).values, records[0].sort(-1).values, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    assert bool((outputs[1][:, 20:] == 0).all())
    assert bool((gradients[1][0][:, 20:] == 0).all())
    assert bool((gradients[1][1][:, 2060:] == 0).all())
    assert bool((gradients[1][2][:, 2060:] == 0).all())
