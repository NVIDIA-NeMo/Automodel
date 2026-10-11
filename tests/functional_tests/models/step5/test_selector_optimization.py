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

"""CUDA selector checks with individual documents beyond the CSA budget."""

import copy
from collections.abc import Iterator

import pytest
import torch

from nemo_automodel.components.models.step5.sparse import region_metadata
from tests.unit_tests.models.step5.test_sparse_metadata import _indexer, _oracle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def ieee_fp32_projections() -> Iterator[None]:
    """Keep CPU/CUDA FP32 projection checks independent of global TF32 defaults."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("offset,length", [(4090, 30), (8064, 128)])
@pytest.mark.parametrize("tied", [False, True])
def test_cuda_eight_k_selection_matches_cpu_stable_sort(offset: int, length: int, tied: bool) -> None:
    module = _indexer(topk=512, region_size=8)
    if tied:
        with torch.no_grad():
            module.w_proj.weight.zero_()
    cuda = copy.deepcopy(module).cuda()
    positions = torch.arange(8192)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 8192, 32)
    keys, logits = module.project_keys(hidden, positions)
    local, local_positions = hidden[:, offset : offset + length], positions[:, offset : offset + length]
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    global_positions, global_padding = positions.cuda(), padding.cuda()
    # Global K/Z projection outputs are held fixed between CPU and CUDA; local
    # Q/weight projections execute on their respective devices in FP32.
    # Native distributed checkpoint loading is covered separately.
    actual = cuda(
        local.cuda(),
        local_positions.cuda(),
        keys.cuda(),
        logits.cuda(),
        global_positions,
        global_padding,
        query_start=offset,
    )
    torch.testing.assert_close(actual.region_ids.cpu(), expected, rtol=0, atol=0)
    active = actual.region_ids[0, 0].cpu()
    assert (active < actual.region_count).sum(-1).max() == 513


def test_cuda_128k_packed_short_documents_use_shared_prefix_routes() -> None:
    module = _indexer(topk=512, region_size=8).cuda()
    positions = torch.arange(4096, device="cuda").repeat(32)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    offset, queries = 17 * 4096, 4096
    meta = region_metadata(positions, padding, query_start=offset, queries=queries, region_size=8, topk=512)
    assert not meta.needs_projection and not meta.score_tiles
    hidden = torch.randn(1, queries, 32, device="cuda")
    actual = module(hidden, positions[:, offset : offset + queries], None, None, positions, padding, metadata=meta)
    assert actual.region_ids.stride(1) == 0
    assert actual.region_ids[0, 0, 0, 0].item() == 17 * 512
    assert actual.region_ids[0, 0, -1, :512].tolist() == list(range(17 * 512, 18 * 512))
    assert actual.region_ids[0, 0, -1, 512].item() == meta.region_count
