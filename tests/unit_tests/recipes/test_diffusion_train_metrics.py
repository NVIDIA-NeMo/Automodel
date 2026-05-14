# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_automodel.recipes.diffusion.train import (
    _calculate_throughput_metrics,
    _count_local_batch_group_samples,
    _get_diffusion_microbatch_size,
)


def test_get_diffusion_microbatch_size_prefers_video_latents():
    batch = {
        "video_latents": torch.zeros(3, 16, 2, 4, 4),
        "text_embeddings": torch.zeros(9, 5, 64),
    }

    assert _get_diffusion_microbatch_size(batch) == 3


def test_get_diffusion_microbatch_size_uses_image_latents():
    assert _get_diffusion_microbatch_size({"image_latents": torch.zeros(2, 16, 8, 8)}) == 2


def test_count_local_batch_group_samples_sums_microbatches():
    batch_group = [
        {"image_latents": torch.zeros(2, 16, 8, 8)},
        {"video_latents": torch.zeros(3, 16, 1, 4, 4)},
    ]

    assert _count_local_batch_group_samples(batch_group) == 5


def test_calculate_throughput_metrics_uses_measured_counts():
    metrics = _calculate_throughput_metrics(
        elapsed_seconds=2.0,
        optimizer_steps=4,
        global_samples=32,
        world_size=8,
    )

    assert metrics["step_time"] == pytest.approx(0.5)
    assert metrics["optimizer_steps_per_sec"] == pytest.approx(2.0)
    assert metrics["samples_per_sec"] == pytest.approx(16.0)
    assert metrics["samples_per_sec_per_gpu"] == pytest.approx(2.0)
    assert metrics["samples_per_step"] == pytest.approx(8.0)
    assert metrics["log_window_seconds"] == pytest.approx(2.0)
    assert metrics["log_window_steps"] == pytest.approx(4.0)
    assert metrics["log_window_samples"] == pytest.approx(32.0)
