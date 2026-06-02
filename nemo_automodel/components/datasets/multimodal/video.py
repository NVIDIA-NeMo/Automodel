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
#
# Derived from BAGEL (Apache-2.0):
#   https://github.com/bytedance-seed/BAGEL
#   data/video_utils.py
# Includes frame sampling code derived from OpenGVLab/InternVL, MIT License.
# MIT upstream copyright notice: Copyright (c) 2023 OpenGVLab.
# BAGEL upstream copyright notice: Copyright (c) 2025 Bytedance Ltd. and/or
# its affiliates.

"""Frame sampling for video-containing VLM samples."""

from __future__ import annotations

import os
import random
import re

import numpy as np
from PIL import Image

# ``decord`` is optional: only required when a dataset actually contains
# videos. Import lazily so that pure-image VLM SFT runs work in environments
# without decord installed.
try:  # pragma: no cover - optional dep
    import decord  # type: ignore
except ImportError:  # pragma: no cover
    decord = None


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    """Select frame indices from a video according to BAGEL sampling mode."""
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(video_path, num_frames, sample="rand", fix_start=None, clip=None, min_num_frames=4):
    """Read sampled frames from a video file with decord."""
    if decord is None:
        raise RuntimeError("decord is required to read video files; install decord to enable video sampling.")
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def extract_frame_number(filename):
    """Extract the numeric frame suffix used by BAGEL frame-folder samples."""
    match = re.search(r"_(\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    """Sort frame paths by their numeric frame suffix."""
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(video_path, num_frames, sample="rand", fix_start=None, min_num_frames=4):
    """Read sampled frames from a directory of extracted frame images."""
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert("RGB")
        frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames


class FrameSampler:
    """Callable that returns a list of PIL frames for a given file path."""

    def __init__(self, max_num_frames=-1, min_num_frames=8, sample="rand"):
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.sample = sample

    def __call__(self, file_name):
        fn = read_frames_folder if file_name.endswith("/") else read_frames_decord
        frames = fn(
            file_name,
            num_frames=self.max_num_frames,
            min_num_frames=self.min_num_frames,
            sample=self.sample,
        )
        return frames
