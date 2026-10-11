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

"""Long packed inputs must reuse CSA metadata kernels across document boundaries."""

import pytest
import torch
from torch._dynamo.utils import counters

from nemo_automodel.components.models.step5.fa4 import _region_tiles


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("queries", [1024, 1001])
def test_region_tiles_reuse_graph_across_packed_region_counts(queries: int) -> None:
    """Check real 128K shapes against independent sorted-set membership."""
    torch._dynamo.reset()
    counters.clear()
    generator = torch.Generator().manual_seed(719)
    first_graphs = None
    for region_count in (16386, 16539, 16481):
        # One selected set per 128-query tile; preserve the production CP4K strides.
        sets = torch.randint(region_count, (1, 4, 32, 513), generator=generator, dtype=torch.int32)
        sets[..., -1] = region_count  # Unused-slot sentinel must not select a real key.
        ids = sets.repeat_interleave(128, dim=2).cuda()[:, :, :queries]
        token_regions = (torch.arange(131072, dtype=torch.int64) * region_count // 131072)[None]
        expected = []
        for width in (112, 96):
            starts = torch.arange(0, 131072, width)
            first = token_regions[0, starts]
            last = token_regions[0, (starts + width - 1).clamp(max=131071)]
            grid = torch.empty(1, 4, 8, len(starts), dtype=torch.bool)
            for group in range(4):
                for row in range(8):
                    selected = sets[0, group, row].unique(sorted=True).long()
                    selected = selected[selected < region_count]
                    left = torch.searchsorted(selected, first)
                    right = torch.searchsorted(selected, last, right=True)
                    grid[0, group, row] = right > left
            expected.append(grid)
        actual = _region_tiles(ids, token_regions.cuda(), region_count, 112, 96)
        torch.cuda.synchronize()
        for result, oracle in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), oracle, rtol=0, atol=0)
        graphs = counters["stats"]["unique_graphs"]
        print(f"regions={region_count}, compiled_graphs={graphs}", flush=True)
        if first_graphs is None:
            first_graphs = graphs
            assert first_graphs > 0
        else:
            assert graphs == first_graphs, (region_count, first_graphs, graphs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_region_bitmap_reuses_kernel_across_region_counts() -> None:
    """Changing packed region counts must not specialize the bitmap packer."""
    from nemo_automodel.components.models.step5.kernels import _pack_regions, pack_region_bitmap

    generator = torch.Generator().manual_seed(829)
    first_kernels = None
    for regions in (16386, 16411, 16390, 2049, 4097):
        ids = torch.randint(regions, (1, 2, 17, 9), generator=generator, dtype=torch.int32)
        ids[..., 0] = 31  # Exercise the sign bit of each int32 word.
        ids[..., -2] = regions - 1
        ids[..., -1] = regions  # Unused-slot sentinel.
        words = (regions + 31) // 32
        expected = torch.zeros((1, 2, 17, words), dtype=torch.int32)
        for group in range(2):
            for query in range(17):
                packed = [0] * words
                for region in ids[0, group, query].tolist():
                    if region < regions:
                        packed[region // 32] |= 1 << (region % 32)
                expected[0, group, query] = torch.tensor(
                    [value if value < 2**31 else value - 2**32 for value in packed], dtype=torch.int32
                )
        actual = pack_region_bitmap(ids.cuda(), regions)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        kernels = len(_pack_regions.device_caches[torch.cuda.current_device()][0])
        print(f"bitmap_regions={regions}, words={words}, compiled_kernels={kernels}", flush=True)
        if first_kernels is None:
            first_kernels = kernels
            assert first_kernels > 0
        else:
            assert kernels == first_kernels, (regions, first_kernels, kernels)
