# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


logger = logging.getLogger(__name__)


class MetaFilesDataset(Dataset):
    """PyTorch dataset for WAN2.1 `.meta` files."""

    def __init__(
        self,
        meta_folder: str,
        transform_text: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_video: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        device: str = "cpu",
        max_files: Optional[int] = None,
    ) -> None:
        self.meta_folder = Path(meta_folder)
        self.transform_text = transform_text
        self.transform_video = transform_video
        self.filter_fn = filter_fn
        self.device = device

        self.meta_files = sorted(self.meta_folder.glob("*.meta"))
        if max_files is None:
            max_files_env = os.environ.get("MAX_META_FILES")
            if max_files_env is not None:
                try:
                    max_files = int(max_files_env)
                except ValueError:
                    logger.warning("Invalid MAX_META_FILES=%s", max_files_env)

        if max_files is not None and max_files > 0:
            self.meta_files = self.meta_files[:max_files]
            logger.info("Limited to first %d meta files", len(self.meta_files))

        if not self.meta_files:
            raise ValueError(f"No .meta files found in {meta_folder}")

        if self.filter_fn:
            filtered = []
            for path in self.meta_files:
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logger.warning("Failed to load %s during filtering: %s", path, exc)
                    continue
                if self.filter_fn(data.get("metadata", {})):
                    filtered.append(path)
            self.meta_files = filtered
            logger.info("Filtered meta files count: %d", len(self.meta_files))

        self._log_dataset_stats()

    def _log_dataset_stats(self) -> None:
        sample_paths = self.meta_files[: min(5, len(self.meta_files))]
        stats: List[Tuple[torch.Size, torch.Size, str]] = []
        for path in sample_paths:
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                stats.append(
                    (
                        data["text_embeddings"].shape,
                        data["video_latents"].shape,
                        str(data.get("deterministic_latents", "unknown")),
                    )
                )
            except Exception as exc:  # pragma: no cover - stats only
                logger.debug("Failed to sample %s: %s", path, exc)

        if stats:
            text_shapes, video_shapes, modes = zip(*stats, strict=False)
            logger.info("Sample text embeddings: %s", text_shapes)
            logger.info("Sample video latents: %s", video_shapes)
            logger.info("Sample encoding modes: %s", set(modes))

    def __len__(self) -> int:
        return len(self.meta_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        path = self.meta_files[index]
        with open(path, "rb") as f:
            data = pickle.load(f)

        text_embeddings: torch.Tensor = data["text_embeddings"].to(self.device)
        video_latents: torch.Tensor = data["video_latents"].to(self.device)

        # Load text_mask if available (backwards compatible)
        text_mask = data.get("text_mask")
        text_embeddings_2 = data.get("text_embeddings_2")
        text_mask_2 = data.get("text_mask_2")
        image_embeds = data.get("image_embeds")
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
        if text_embeddings_2 is not None:
            text_embeddings_2 = text_embeddings_2.to(self.device)
        if text_mask_2 is not None:
            text_mask_2 = text_mask_2.to(self.device)
        if image_embeds is not None:
            image_embeds = image_embeds.to(self.device)

        if self.transform_text is not None:
            text_embeddings = self.transform_text(text_embeddings)
        if self.transform_video is not None:
            video_latents = self.transform_video(video_latents)

        file_info = {
            "meta_filename": Path(path).name,
            "original_filename": data.get("original_filename", "unknown"),
            "original_video_path": data.get("original_video_path", "unknown"),
            "deterministic_latents": data.get("deterministic_latents", "unknown"),
            "memory_optimization": data.get("memory_optimization", "unknown"),
            "num_frames": data.get("num_frames", "unknown"),
        }

        result = {
            "text_embeddings": text_embeddings,
            "video_latents": video_latents,
            "metadata": data.get("metadata", {}),
            "file_info": file_info,
        }

        # Add text_mask if available (backwards compatible)
        if text_mask is not None:
            result["text_mask"] = text_mask
        if text_embeddings_2 is not None:
            result["text_embeddings_2"] = text_embeddings_2
        if text_mask_2 is not None:
            result["text_mask_2"] = text_mask_2
        if image_embeds is not None:
            result["image_embeds"] = image_embeds

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) > 0:
        assert batch[0]["text_embeddings"].ndim == 3, "Expected text_embeddings.ndim to be 3"
        assert batch[0]["video_latents"].ndim == 5, "Expected video_latents.ndim to be 5"
    # use cat to stack the tensors in the batch
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)
    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)

    result = {
        "text_embeddings": text_embeddings,
        "video_latents": video_latents,
        "metadata": [item["metadata"] for item in batch],
        "file_info": [item["file_info"] for item in batch],
    }

    # Collate text_mask if available (backwards compatible)
    if len(batch) > 0 and "text_mask" in batch[0]:
        text_mask = torch.cat([item["text_mask"] for item in batch], dim=0)
        result["text_mask"] = text_mask
    if len(batch) > 0 and "text_embeddings_2" in batch[0]:
        text_embeddings_2 = torch.cat([item["text_embeddings_2"] for item in batch], dim=0)
        result["text_embeddings_2"] = text_embeddings_2
    if len(batch) > 0 and "text_mask_2" in batch[0]:
        text_mask_2 = torch.cat([item["text_mask_2"] for item in batch], dim=0)
        result["text_mask_2"] = text_mask_2
    if len(batch) > 0 and "image_embeds" in batch[0]:
        image_embeds = torch.cat([item["image_embeds"] for item in batch], dim=0)
        result["image_embeds"] = image_embeds

    return result


def build_node_parallel_sampler(
    dataset: "Dataset",
    dp_rank: int,
    dp_world_size: int,
    shuffle: bool = True,
) -> Optional["DistributedSampler"]:
    if not dist.is_initialized():
        return None

    return DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=shuffle,
        drop_last=False,
    )


def build_dataloader(
    *,
    meta_folder: str,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    device: str = "cpu",
    transform_text: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    transform_video: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
    max_files: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = MetaFilesDataset(
        meta_folder=meta_folder,
        transform_text=transform_text,
        transform_video=transform_video,
        filter_fn=filter_fn,
        device=device,
        max_files=max_files,
    )

    sampler = build_node_parallel_sampler(dataset, dp_rank, dp_world_size, shuffle=shuffle)

    use_pin_memory = device == "cpu"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    return dataloader, sampler


def create_dataloader(
    meta_folder: str,
    batch_size: int,
    num_nodes: int,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    return build_dataloader(meta_folder=meta_folder, batch_size=batch_size, num_nodes=num_nodes)
