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

"""Gloo CP2 selection components; no checkpoint loading or FSDP coverage."""

import copy
import datetime
import json

import torch
import torch.distributed as dist

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5 import layers
from nemo_automodel.components.models.step5.layers import Step5Attention
from nemo_automodel.components.models.step5.sparse import region_metadata
from tests.unit_tests.models.step5.test_model import tiny_config
from tests.unit_tests.models.step5.test_sparse_metadata import _indexer


def main() -> None:
    """Compare contiguous CP2 routes with CP1 and inspect projection symmetry."""
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    backend = BackendConfig(attn="sdpa", linear="torch", experts="torch_mm", dispatcher="torch")
    attention = Step5Attention(tiny_config(), 3, backend=backend, dtype=torch.float32)
    attention.indexer = _indexer(topk=512, region_size=8)
    reference = copy.deepcopy(attention)
    attention.cp_group = dist.group.WORLD
    original_gather = layers.gather_metadata
    reports = []
    for lengths in ((8192,), (3073, 5119), (2051, 4096, 2045)):
        torch.manual_seed(792)
        positions = torch.cat([torch.arange(length) for length in lengths])[None]
        padding = torch.zeros_like(positions, dtype=torch.bool)
        hidden = torch.randn(1, 8192, 32)
        offset, local = rank * 4096, 4096
        meta = region_metadata(positions, padding, query_start=offset, queries=local, region_size=8, topk=512)
        gathers = []
        norm_calls = []

        def gather(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
            gathers.append(tuple(tensor.shape))
            return original_gather(tensor, group)

        def norm_hook(module: torch.nn.Module, arguments: tuple[torch.Tensor]) -> None:
            norm_calls.append(arguments[0].shape[1])

        hook = attention.indexer.q_norm.register_forward_pre_hook(norm_hook)
        layers.gather_metadata = gather
        try:
            selected = attention._select_regions(
                hidden[:, offset : offset + local],
                positions[:, offset : offset + local],
                positions,
                padding,
                query_start=offset,
                sparse_metadata=meta,
            )
        finally:
            hook.remove()
            layers.gather_metadata = original_gather
        expected = reference._select_regions(
            hidden,
            positions,
            positions,
            padding,
            query_start=0,
            sparse_metadata=None,
        )
        torch.testing.assert_close(
            selected.region_ids, expected.region_ids[:, :, offset : offset + local], rtol=0, atol=0
        )
        assert len(gathers) == int(meta.needs_projection)
        assert norm_calls == [local if meta.score_tiles else 0]
        assert selected.all_regions_selected == (not meta.score_tiles)
        reports.append(
            {
                "rank": rank,
                "document_lengths": lengths,
                "global_needs_projection": meta.needs_projection,
                "local_sparse_queries": int(meta.sparse_queries.sum()),
                "all_regions_selected": selected.all_regions_selected,
                "indexer_gathers": gathers,
                "q_norm_tokens": norm_calls,
                "exact_cp1_routes": True,
            }
        )
        dist.barrier()
    all_reports = [None, None]
    dist.all_gather_object(all_reports, reports)
    if rank == 0:
        print(
            json.dumps(
                {
                    "scope": "CPU Gloo CP2 component, random weights; no FSDP or checkpoint loading",
                    "results": all_reports,
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
