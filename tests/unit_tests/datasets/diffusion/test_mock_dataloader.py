# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for mock_dataloader.py: MockWanDataset, mock_collate_fn, build_mock_dataloader."""

import pytest
import torch
from torch.utils.data import DistributedSampler

from nemo_automodel.components.datasets.diffusion.mock_dataloader import (
    MockWanDataset,
    build_mock_dataloader,
    mock_collate_fn,
)


# =============================================================================
# TestMockWanDataset
# =============================================================================


class TestMockWanDataset:
    """Tests for MockWanDataset."""

    def test_len_default(self):
        ds = MockWanDataset()
        assert len(ds) == 1024

    def test_len_custom(self):
        ds = MockWanDataset(length=50)
        assert len(ds) == 50

    def test_len_min_clamping(self):
        ds = MockWanDataset(length=0)
        assert len(ds) == 1

        ds2 = MockWanDataset(length=-5)
        assert len(ds2) == 1

    def test_getitem_keys(self):
        ds = MockWanDataset(length=4)
        item = ds[0]
        assert "text_embeddings" in item
        assert "video_latents" in item
        assert "metadata" in item
        assert "file_info" in item

    def test_getitem_shapes_default(self):
        ds = MockWanDataset()
        item = ds[0]
        assert item["text_embeddings"].shape == (1, 77, 4096)
        assert item["video_latents"].shape == (1, 16, 16, 30, 52)

    @pytest.mark.parametrize(
        "channels, frames, h, w, seq_len, embed_dim",
        [
            (8, 4, 16, 16, 32, 2048),
            (32, 8, 64, 64, 128, 512),
        ],
    )
    def test_getitem_custom_dims(self, channels, frames, h, w, seq_len, embed_dim):
        ds = MockWanDataset(
            length=2,
            num_channels=channels,
            num_frame_latents=frames,
            spatial_h=h,
            spatial_w=w,
            text_seq_len=seq_len,
            text_embed_dim=embed_dim,
        )
        item = ds[0]
        assert item["video_latents"].shape == (1, channels, frames, h, w)
        assert item["text_embeddings"].shape == (1, seq_len, embed_dim)

    def test_getitem_dtypes(self):
        ds = MockWanDataset(length=2)
        item = ds[0]
        assert item["text_embeddings"].dtype == torch.float32
        assert item["video_latents"].dtype == torch.float32

    def test_getitem_metadata_is_empty_dict(self):
        ds = MockWanDataset(length=2)
        item = ds[0]
        assert item["metadata"] == {}

    def test_file_info_idx_interpolation(self):
        ds = MockWanDataset(length=10)
        for idx in [0, 3, 9]:
            item = ds[idx]
            assert str(idx) in item["file_info"]["meta_filename"]
            assert str(idx) in item["file_info"]["original_filename"]


# =============================================================================
# TestMockCollateFn
# =============================================================================


class TestMockCollateFn:
    """Tests for mock_collate_fn."""

    def test_single_item(self):
        ds = MockWanDataset(length=1)
        batch = mock_collate_fn([ds[0]])
        assert batch["text_embeddings"].shape[0] == 1
        assert batch["video_latents"].shape[0] == 1

    def test_multi_item_concat(self):
        ds = MockWanDataset(length=4)
        items = [ds[i] for i in range(4)]
        batch = mock_collate_fn(items)
        assert batch["text_embeddings"].shape[0] == 4
        assert batch["video_latents"].shape[0] == 4

    def test_metadata_is_list(self):
        ds = MockWanDataset(length=3)
        items = [ds[i] for i in range(3)]
        batch = mock_collate_fn(items)
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == 3

    def test_file_info_is_list(self):
        ds = MockWanDataset(length=2)
        items = [ds[i] for i in range(2)]
        batch = mock_collate_fn(items)
        assert isinstance(batch["file_info"], list)
        assert len(batch["file_info"]) == 2


# =============================================================================
# TestBuildMockDataloader
# =============================================================================


class TestBuildMockDataloader:
    """Tests for build_mock_dataloader."""

    def test_returns_dataloader_and_none_sampler(self):
        dl, sampler = build_mock_dataloader(dp_world_size=1, length=8, batch_size=2, num_workers=0)
        assert sampler is None
        assert dl is not None

    def test_returns_distributed_sampler_when_multi_rank(self):
        dl, sampler = build_mock_dataloader(dp_world_size=2, dp_rank=0, length=8, batch_size=2, num_workers=0)
        assert isinstance(sampler, DistributedSampler)

    def test_iteration_yields_correct_shapes(self):
        dl, _ = build_mock_dataloader(
            dp_world_size=1,
            batch_size=2,
            length=4,
            num_workers=0,
            num_channels=16,
            num_frame_latents=4,
            spatial_h=8,
            spatial_w=8,
            text_seq_len=32,
            text_embed_dim=512,
        )
        batch = next(iter(dl))
        assert batch["video_latents"].shape == (2, 16, 4, 8, 8)
        assert batch["text_embeddings"].shape == (2, 32, 512)

    def test_full_iteration(self):
        dl, _ = build_mock_dataloader(dp_world_size=1, batch_size=4, length=16, num_workers=0)
        count = 0
        for batch in dl:
            assert "video_latents" in batch
            count += 1
        assert count == 4
