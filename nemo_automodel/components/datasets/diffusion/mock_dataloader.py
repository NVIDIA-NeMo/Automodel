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

"""Mock dataloader for automodel WAN training tests.

This module provides a mock dataset and dataloader that generates random
tensors with the correct shapes for WAN 2.1 training, allowing functional
tests to run without requiring real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild


@dataclass
class MockWanDatasetConfig:
    """Construction-time configuration for :class:`MockWanDataset`."""

    length: int = 1024
    """Number of samples in the dataset."""
    num_channels: int = 16
    """Number of latent channels."""
    num_frame_latents: int = 16
    """Number of temporal latent frames."""
    spatial_h: int = 30
    """Height of spatial latents."""
    spatial_w: int = 52
    """Width of spatial latents."""
    text_seq_len: int = 77
    """Length of text sequence."""
    text_embed_dim: int = 4096
    """Dimension of text embeddings."""
    device: str = "cpu"
    """Device to place tensors on."""

    def build(self) -> "MockWanDataset":
        """Build a :class:`MockWanDataset` from this :class:`MockWanDatasetConfig`."""
        return MockWanDataset(
            length=self.length,
            num_channels=self.num_channels,
            num_frame_latents=self.num_frame_latents,
            spatial_h=self.spatial_h,
            spatial_w=self.spatial_w,
            text_seq_len=self.text_seq_len,
            text_embed_dim=self.text_embed_dim,
            device=self.device,
        )


class MockWanDataset(Dataset):
    """Mock dataset that generates random data matching WAN 2.1 expected format.

    Args:
        length: Number of samples in the dataset.
        num_channels: Number of latent channels (default: 16 for WAN).
        num_frame_latents: Number of temporal latent frames.
        spatial_h: Height of spatial latents.
        spatial_w: Width of spatial latents.
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings (default: 4096 for UMT5).
        device: Device to place tensors on.
    """

    def __init__(
        self,
        length: int = 1024,
        num_channels: int = 16,
        num_frame_latents: int = 16,
        spatial_h: int = 30,
        spatial_w: int = 52,
        text_seq_len: int = 77,
        text_embed_dim: int = 4096,
        device: str = "cpu",
    ) -> None:
        self.length = max(int(length), 1)
        self.num_channels = num_channels
        self.num_frame_latents = num_frame_latents
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.text_seq_len = text_seq_len
        self.text_embed_dim = text_embed_dim
        self.device = device

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a mock sample with random data.

        Returns:
            Dict containing:
                - text_embeddings: [1, text_seq_len, text_embed_dim]
                - video_latents: [1, num_channels, num_frame_latents, spatial_h, spatial_w]
                - metadata: empty dict
                - file_info: mock file info
        """
        # Generate random video latents: (1, C, T, H, W)
        video_latents = torch.randn(
            1,
            self.num_channels,
            self.num_frame_latents,
            self.spatial_h,
            self.spatial_w,
            dtype=torch.float32,
            device=self.device,
        )

        # Generate random text embeddings: (1, seq_len, embed_dim)
        text_embeddings = torch.randn(
            1,
            self.text_seq_len,
            self.text_embed_dim,
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "text_embeddings": text_embeddings,
            "video_latents": video_latents,
            "metadata": {},
            "file_info": {
                "meta_filename": f"mock_sample_{idx}.meta",
                "original_filename": f"mock_video_{idx}.mp4",
                "original_video_path": f"/mock/path/video_{idx}.mp4",
                "deterministic_latents": True,
                "memory_optimization": False,
                "num_frames": self.num_frame_latents * 4,  # Approximate original frames
            },
        }


def mock_collate_fn(batch):
    """Collate function for mock dataset, matching the real collate_fn behavior."""
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)
    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)

    return {
        "text_embeddings": text_embeddings,
        "video_latents": video_latents,
        "metadata": [item["metadata"] for item in batch],
        "file_info": [item["file_info"] for item in batch],
    }


@dataclass
class MockWanDataloaderConfig:
    """Construction-time configuration for a mock WAN dataloader."""

    num_workers: int = 0
    device: str = "cpu"
    length: int = 1024
    num_channels: int = 16
    num_frame_latents: int = 16
    spatial_h: int = 30
    spatial_w: int = 52
    text_seq_len: int = 77
    text_embed_dim: int = 4096
    shuffle: bool = True

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        """Build the configured mock dataset, sampler, and dataloader."""
        dataset = MockWanDatasetConfig(
            length=self.length,
            num_channels=self.num_channels,
            num_frame_latents=self.num_frame_latents,
            spatial_h=self.spatial_h,
            spatial_w=self.spatial_w,
            text_seq_len=self.text_seq_len,
            text_embed_dim=self.text_embed_dim,
            device=self.device,
        ).build()
        sampler = None
        if dp_world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=self.shuffle,
                drop_last=False,
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None and self.shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=mock_collate_fn,
            pin_memory=self.device == "cpu",
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)


def build_mock_dataloader(
    *,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
    length: int = 1024,
    num_channels: int = 16,
    num_frame_latents: int = 16,
    spatial_h: int = 30,
    spatial_w: int = 52,
    text_seq_len: int = 77,
    text_embed_dim: int = 4096,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Build a mock dataloader for WAN training tests.

    This function follows the same interface as build_dataloader but generates
    random data instead of loading from .meta files.

    Args:
        dp_rank: Data parallel rank.
        dp_world_size: Data parallel world size.
        batch_size: Batch size per GPU.
        num_workers: Number of dataloader workers.
        device: Device to place tensors on.
        length: Number of samples in mock dataset.
        num_channels: Number of latent channels (default: 16).
        num_frame_latents: Number of temporal latent frames.
        spatial_h: Height of spatial latents.
        spatial_w: Width of spatial latents.
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings.
        shuffle: Whether to shuffle data.

    Returns:
        Tuple of (DataLoader, DistributedSampler or None).
    """
    result = MockWanDataloaderConfig(
        num_workers=num_workers,
        device=device,
        length=length,
        num_channels=num_channels,
        num_frame_latents=num_frame_latents,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        text_seq_len=text_seq_len,
        text_embed_dim=text_embed_dim,
        shuffle=shuffle,
    ).build(
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    return result.dataloader, result.sampler
