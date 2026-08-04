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

"""Unit tests for collating LTX-2 audio cache fields through collate_fn_video."""

import torch

from nemo_automodel.components.datasets.diffusion.collate_fns import collate_fn_video
from nemo_automodel.components.datasets.diffusion.text_to_video_dataset import (
    VIDEO_OPTIONAL_FIELDS,
    load_optional_video_fields,
)


def _make_sample(with_audio: bool = True):
    sample = {
        "video_latents": torch.randn(1, 128, 4, 8, 8),
        "text_embeddings": torch.randn(1, 32, 16),
        "bucket_resolution": torch.tensor([256, 256]),
        "text_mask": torch.ones(1, 32, dtype=torch.long),
    }
    if with_audio:
        sample["audio_latents"] = torch.randn(1, 8, 126, 16)
        sample["audio_text_embeddings"] = torch.randn(1, 32, 12)
    return sample


class TestAudioOptionalFields:
    def test_audio_keys_registered(self):
        assert "audio_latents" in VIDEO_OPTIONAL_FIELDS
        assert "audio_text_embeddings" in VIDEO_OPTIONAL_FIELDS

    def test_load_optional_fields_includes_audio(self):
        loaded = load_optional_video_fields(_make_sample())
        assert loaded["audio_latents"].shape == (1, 8, 126, 16)
        assert loaded["audio_text_embeddings"].shape == (1, 32, 12)

    def test_collate_with_audio(self):
        batch = [_make_sample(), _make_sample()]
        result = collate_fn_video(batch, model_type="ltx2")

        assert result["video_latents"].shape == (2, 128, 4, 8, 8)
        assert result["audio_latents"].shape == (2, 8, 126, 16)
        assert result["audio_text_embeddings"].shape == (2, 32, 12)
        assert result["text_mask"].shape == (2, 32)
        assert result["data_type"] == "video"

    def test_collate_without_audio_unchanged(self):
        batch = [_make_sample(with_audio=False), _make_sample(with_audio=False)]
        result = collate_fn_video(batch, model_type="wan")
        assert "audio_latents" not in result
        assert "audio_text_embeddings" not in result
        assert result["video_latents"].shape == (2, 128, 4, 8, 8)
