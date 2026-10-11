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

"""CPU coverage of indexed-attention eligibility and physical region bounds."""

import pytest
import torch

from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.sparse import region_metadata


def test_indexed_backend_is_opt_in_and_fa4_only() -> None:
    assert not Step5BackendConfig().indexed_csa
    assert Step5BackendConfig(attn="fa4", indexed_csa=True).indexed_csa
    with pytest.raises(TypeError, match="indexed_csa"):
        Step5BackendConfig(attn="fa4", indexed_csa=1)
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="flex", indexed_csa=True)


@pytest.mark.parametrize("length,enabled", [(8192, False), (16384, False), (32768, False), (40960, True)])
def test_indexed_layout_requires_long_effective_history(length: int, enabled: bool) -> None:
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions,
        padding,
        query_start=length - 4096,
        queries=4096,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert (metadata.indexed_layout is not None) == enabled
    assert (
        region_metadata(
            positions, padding, query_start=length - 4096, queries=4096, region_size=8, topk=512
        ).indexed_layout
        is None
    )
    if enabled:
        layout = metadata.indexed_layout
        assert layout.starts.dtype == layout.ends.dtype == torch.int32
        torch.testing.assert_close(layout.starts[0], torch.arange(0, length, 8, dtype=torch.int32))
        torch.testing.assert_close(layout.ends[0], torch.arange(8, length + 1, 8, dtype=torch.int32))


def test_physical_bounds_preserve_short_document_before_long_padding() -> None:
    # Last valid document has three tokens; all 15 padding positions are zero.
    # Reusing compression slots would overwrite its first valid key with padding.
    positions = torch.cat((torch.arange(98304), torch.arange(3), torch.zeros(15, dtype=torch.long)))[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[:, -15:] = True
    metadata = region_metadata(
        positions,
        padding,
        query_start=65536,
        queries=positions.shape[1] - 65536,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert metadata.indexed_layout is not None
    layout = metadata.indexed_layout
    assert layout.starts[0, -1].item() == 98304
    assert layout.ends[0, -1].item() == 98322
    torch.testing.assert_close(layout.starts[0, :-1], torch.arange(0, 98304, 8, dtype=torch.int32))
    torch.testing.assert_close(layout.ends[0, :-1], torch.arange(8, 98305, 8, dtype=torch.int32))


@pytest.mark.parametrize("defect", ["internal_padding", "repeated_position", "negative_position"])
def test_irregular_layout_keeps_existing_attention(defect: str) -> None:
    positions = torch.arange(98304)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    if defect == "internal_padding":
        padding[:, 100] = True
    elif defect == "repeated_position":
        positions[:, 100] = 99
    else:
        positions[:, 100] = -1
    metadata = region_metadata(
        positions, padding, query_start=65536, queries=4096, region_size=8, topk=512, indexed_csa=True
    )
    assert metadata.indexed_layout is None


def test_long_pack_of_short_documents_does_not_enable_indexed_attention() -> None:
    positions = torch.arange(1024).repeat(128)[None]
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=98304,
        queries=4096,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert not metadata.score_tiles
    assert metadata.indexed_layout is None


def test_indexed_layout_supports_batch_and_empty_query_fallback() -> None:
    positions = torch.arange(98304)[None].expand(2, -1).contiguous()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions, padding, query_start=65536, queries=4096, region_size=8, topk=512, indexed_csa=True
    )
    assert metadata.indexed_layout is not None
    torch.testing.assert_close(metadata.indexed_layout.starts[0], metadata.indexed_layout.starts[1])
    empty = region_metadata(positions, padding, query_start=0, queries=0, region_size=8, topk=512, indexed_csa=True)
    assert empty.indexed_layout is None
