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

"""Training windows from recorded single-arm episodes (``.npz`` files).

An episode file holds ``canvas`` (frames, height, width, 3) uint8, ``obs8`` and ``cmd8`` (frames, 8)
float32 flange poses ``[xyz, quat_wxyz, grip]`` and ``prompt``. A window is ``window_frames``
consecutive frames; its actions are the commanded poses relative to the observed pose at the start of
each chunk, and its proprio the observed pose at each chunk start, the same layout as the reference
DROID windows. The LIBERO tutorial's ``convert_libero_demos.py`` writes such files.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from praxis.train.data import TrainingWindow

    from nemo_automodel.components.models.praxis.config import JsonValue


def _mapping(config: Mapping[str, JsonValue], name: str) -> Mapping[str, JsonValue]:
    section = config[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"checkpoint config: {name!r} is not a mapping")
    return section


class EpisodeWindowDataset(IterableDataset):
    """Endless random windows over a directory of recorded episodes.

    Args:
        root: Directory with ``*.npz`` episodes.
        model_config: The checkpoint's ``config.json``; its ``chunking`` section gives the chunk geometry
            and its ``video`` section the canvas size the episodes must have been rendered at.
        seed: Base seed; the draws also depend on the rank and the loader worker.
        rank: This data-parallel rank; episodes are split round-robin across ranks.
        world_size: Number of data-parallel ranks.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        model_config: Mapping[str, JsonValue],
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        chunking = _mapping(model_config, "chunking")
        video = _mapping(model_config, "video")
        self.files = sorted(Path(root).glob("*.npz"))[rank::world_size]
        if not self.files:
            raise ValueError(f"no episodes for rank {rank} under {root}")
        from praxis.robot_io import contiguous_chunk_geometry

        self.geometry = contiguous_chunk_geometry(int(chunking["action_chunk"]), int(chunking["n_chunks"]))
        self.window_frames = sum(self.geometry[-1])
        self.seed, self.rank = seed, rank
        self.canvas_hw = (int(video["height"]), int(video["width"]))

    def window(self, file: Path, start: int) -> TrainingWindow:
        from praxis.robot_io import pose8_to_10d
        from praxis.train.data import TrainingWindow

        """The window of ``file`` starting at frame ``start``."""
        ep = np.load(file, allow_pickle=False)
        canvas, obs8, cmd8 = ep["canvas"], ep["obs8"], ep["cmd8"]
        if tuple(canvas.shape[1:3]) != self.canvas_hw:
            raise ValueError(
                f"{file.name}: canvas is {canvas.shape[1]}x{canvas.shape[2]}, the checkpoint's video canvas is "
                f"{self.canvas_hw[0]}x{self.canvas_hw[1]}; convert the episodes at the checkpoint's tile size"
            )
        stop = start + self.window_frames
        frames = torch.from_numpy(np.ascontiguousarray(canvas[start:stop])).permute(0, 3, 1, 2).contiguous()
        starts = [start + off for off, _ in self.geometry]
        actions = [
            pose8_to_10d(cmd8[sk : sk + ln], anchor8=obs8[sk])
            for sk, (_, ln) in zip(starts, self.geometry, strict=True)
        ]
        return TrainingWindow(
            frames=frames,
            prompt=str(ep["prompt"]),
            actions_rel=np.concatenate(actions, axis=0).astype(np.float32),
            proprio_abs=pose8_to_10d(obs8[starts]).astype(np.float32),
            meta={"episode": file.stem, "anchor": start, "source": str(file.parent)},
        )

    def __iter__(self) -> Iterator[TrainingWindow]:
        info = get_worker_info()
        worker = 0 if info is None else int(info.id)
        rng = np.random.default_rng((self.seed, self.rank, worker))
        while True:
            file = self.files[int(rng.integers(len(self.files)))]
            n = int(np.load(file, allow_pickle=False)["obs8"].shape[0])
            if n < self.window_frames:
                continue
            yield self.window(file, int(rng.integers(0, n - self.window_frames + 1)))
