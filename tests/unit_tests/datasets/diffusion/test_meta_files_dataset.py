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

"""Unit tests for meta_files_dataset.py: MetaFilesDataset, collate_fn, build_node_parallel_sampler, build_dataloader."""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.datasets.diffusion.meta_files_dataset import (
    MetaFilesDataset,
    build_dataloader,
    build_node_parallel_sampler,
    collate_fn,
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_meta_sample(idx, include_optional=False):
    """Create a single meta sample dict."""
    data = {
        "text_embeddings": torch.randn(1, 77, 4096),
        "video_latents": torch.randn(1, 16, 4, 8, 8),
        "metadata": {"caption": f"test caption {idx}"},
        "original_filename": f"video_{idx}.mp4",
        "original_video_path": f"/data/video_{idx}.mp4",
        "deterministic_latents": True,
        "memory_optimization": False,
        "num_frames": 16,
    }
    if include_optional:
        data["text_mask"] = torch.ones(1, 77)
        data["text_embeddings_2"] = torch.randn(1, 77, 768)
        data["text_mask_2"] = torch.ones(1, 77)
        data["image_embeds"] = torch.randn(1, 512)
    return data


@pytest.fixture
def meta_dir():
    """Create a temp directory with 5 .meta files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"sample_{i:04d}.meta"
            with open(path, "wb") as f:
                pickle.dump(_make_meta_sample(i), f)
        yield tmpdir


@pytest.fixture
def meta_dir_with_optional_fields():
    """Create a temp directory with .meta files that include optional fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"sample_{i:04d}.meta"
            with open(path, "wb") as f:
                pickle.dump(_make_meta_sample(i, include_optional=True), f)
        yield tmpdir


# =============================================================================
# TestMetaFilesDataset
# =============================================================================


class TestMetaFilesDataset:
    """Tests for MetaFilesDataset init and __getitem__."""

    def test_basic_init(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        assert len(ds) == 5

    def test_empty_folder_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No .meta files"):
                MetaFilesDataset(tmpdir)

    def test_max_files(self, meta_dir):
        ds = MetaFilesDataset(meta_dir, max_files=3)
        assert len(ds) == 3

    def test_max_files_env_var(self, meta_dir, monkeypatch):
        monkeypatch.setenv("MAX_META_FILES", "2")
        ds = MetaFilesDataset(meta_dir)
        assert len(ds) == 2

    def test_invalid_env_var(self, meta_dir, monkeypatch):
        monkeypatch.setenv("MAX_META_FILES", "not_a_number")
        # Should warn but not crash; loads all 5 files
        ds = MetaFilesDataset(meta_dir)
        assert len(ds) == 5

    def test_filter_fn_accept(self, meta_dir):
        ds = MetaFilesDataset(meta_dir, filter_fn=lambda m: True)
        assert len(ds) == 5

    def test_filter_fn_reject_all(self, meta_dir):
        # After filtering removes all, dataset has 0 files (source doesn't re-raise)
        ds = MetaFilesDataset(meta_dir, filter_fn=lambda m: False)
        assert len(ds) == 0

    def test_filter_fn_partial(self, meta_dir):
        # Only accept samples with even index in caption
        def _filter(meta):
            caption = meta.get("caption", "")
            idx_str = caption.split()[-1] if caption else ""
            return idx_str.isdigit() and int(idx_str) % 2 == 0

        ds = MetaFilesDataset(meta_dir, filter_fn=_filter)
        assert 1 <= len(ds) <= 5

    def test_getitem_keys(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        item = ds[0]
        assert "text_embeddings" in item
        assert "video_latents" in item
        assert "metadata" in item
        assert "file_info" in item

    def test_getitem_shapes(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        item = ds[0]
        assert item["text_embeddings"].shape == (1, 77, 4096)
        assert item["video_latents"].shape == (1, 16, 4, 8, 8)

    def test_getitem_with_transforms(self, meta_dir):
        ds = MetaFilesDataset(
            meta_dir,
            transform_text=lambda x: x * 2,
            transform_video=lambda x: x + 1,
        )
        item = ds[0]
        # Just check it ran without error and shapes are preserved
        assert item["text_embeddings"].shape == (1, 77, 4096)
        assert item["video_latents"].shape == (1, 16, 4, 8, 8)

    def test_getitem_optional_fields_present(self, meta_dir_with_optional_fields):
        ds = MetaFilesDataset(meta_dir_with_optional_fields)
        item = ds[0]
        assert "text_mask" in item
        assert "text_embeddings_2" in item
        assert "text_mask_2" in item
        assert "image_embeds" in item

    def test_getitem_optional_fields_absent(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        item = ds[0]
        assert "text_mask" not in item
        assert "text_embeddings_2" not in item
        assert "text_mask_2" not in item
        assert "image_embeds" not in item


# =============================================================================
# TestCollateFn
# =============================================================================


class TestCollateFn:
    """Tests for collate_fn."""

    def test_basic_collation(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        items = [ds[i] for i in range(3)]
        batch = collate_fn(items)
        assert batch["text_embeddings"].shape[0] == 3
        assert batch["video_latents"].shape[0] == 3
        assert isinstance(batch["metadata"], list)
        assert len(batch["metadata"]) == 3

    def test_optional_fields_collation(self, meta_dir_with_optional_fields):
        ds = MetaFilesDataset(meta_dir_with_optional_fields)
        items = [ds[i] for i in range(2)]
        batch = collate_fn(items)
        assert "text_mask" in batch
        assert batch["text_mask"].shape[0] == 2
        assert "text_embeddings_2" in batch
        assert "text_mask_2" in batch
        assert "image_embeds" in batch

    def test_ndim_assertion(self):
        """collate_fn checks text_embeddings.ndim == 3 and video_latents.ndim == 5."""
        bad_item = {
            "text_embeddings": torch.randn(77, 4096),  # 2D, not 3D
            "video_latents": torch.randn(1, 16, 4, 8, 8),
            "metadata": {},
            "file_info": {},
        }
        with pytest.raises(AssertionError, match="text_embeddings"):
            collate_fn([bad_item])


# =============================================================================
# TestBuildNodeParallelSampler
# =============================================================================


class TestBuildNodeParallelSampler:
    """Tests for build_node_parallel_sampler."""

    def test_returns_none_when_not_initialized(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        with patch("nemo_automodel.components.datasets.diffusion.meta_files_dataset.dist.is_initialized", return_value=False):
            sampler = build_node_parallel_sampler(ds, dp_rank=0, dp_world_size=1)
        assert sampler is None

    def test_returns_sampler_when_initialized(self, meta_dir):
        ds = MetaFilesDataset(meta_dir)
        with patch("nemo_automodel.components.datasets.diffusion.meta_files_dataset.dist.is_initialized", return_value=True):
            sampler = build_node_parallel_sampler(ds, dp_rank=0, dp_world_size=2)
        assert sampler is not None
        from torch.utils.data import DistributedSampler

        assert isinstance(sampler, DistributedSampler)


# =============================================================================
# TestBuildDataloader
# =============================================================================


class TestBuildDataloader:
    """Tests for build_dataloader."""

    def test_basic_build(self, meta_dir):
        with patch("nemo_automodel.components.datasets.diffusion.meta_files_dataset.dist.is_initialized", return_value=False):
            dl, sampler = build_dataloader(
                meta_folder=meta_dir,
                batch_size=2,
                dp_rank=0,
                dp_world_size=1,
                num_workers=0,
            )
        assert dl is not None
        assert sampler is None

    def test_iteration(self, meta_dir):
        with patch("nemo_automodel.components.datasets.diffusion.meta_files_dataset.dist.is_initialized", return_value=False):
            dl, _ = build_dataloader(
                meta_folder=meta_dir,
                batch_size=2,
                dp_rank=0,
                dp_world_size=1,
                num_workers=0,
            )
        batch = next(iter(dl))
        assert "text_embeddings" in batch
        assert "video_latents" in batch
        assert batch["text_embeddings"].shape[0] == 2
