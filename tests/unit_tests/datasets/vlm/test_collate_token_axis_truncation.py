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

"""Token-axis truncation must leave patch-indexed media tensors alone."""

import pytest
import torch

from nemo_automodel.components.datasets.vlm.collate_fns import _truncate_token_axis


def _batch(seq: int, patches: int, media_span: slice | None = None) -> dict[str, torch.Tensor]:
    """Build a VLM batch with text- and patch-indexed entries.

    Args:
        seq: Text sequence length; token-aligned tensors get shape [1, seq].
        patches: Number of image patches; ``pixel_values`` is [1, patches, 8] and
            ``image_position_ids`` is [1, patches, 2].
        media_span: Positions along the token axis marked as multimodal in
            ``mm_token_type_ids``; ``None`` marks no multimodal tokens.

    Returns:
        Batch dict mixing token-aligned and patch-aligned tensors.
    """
    mm = torch.zeros(1, seq, dtype=torch.long)
    if media_span is not None:
        mm[:, media_span] = 1
    return {
        "input_ids": torch.arange(seq).unsqueeze(0),
        "attention_mask": torch.ones(1, seq, dtype=torch.long),
        "labels": torch.arange(seq).unsqueeze(0),
        "mm_token_type_ids": mm,
        "pixel_values": torch.randn(1, patches, 8),
        "image_position_ids": torch.zeros(1, patches, 2, dtype=torch.long),
    }


def test_media_tensors_keep_their_patch_axis():
    """pixel_values and image_position_ids must stay full length."""
    batch = _batch(seq=100, patches=64, media_span=slice(0, 10))

    _truncate_token_axis(batch, max_length=50)

    assert batch["input_ids"].shape == (1, 50)
    assert batch["labels"].shape == (1, 50)
    assert batch["mm_token_type_ids"].shape == (1, 50)
    # Previously image_position_ids was clipped to 50 while pixel_values kept 64,
    # so the vision tower failed on hidden_states + position_embeddings.
    assert batch["pixel_values"].shape == (1, 64, 8)
    assert batch["image_position_ids"].shape == (1, 64, 2)


def test_truncating_into_image_tokens_raises():
    """Dropping placeholder tokens would orphan the image features."""
    batch = _batch(seq=100, patches=64, media_span=slice(60, 90))

    with pytest.raises(ValueError, match="cuts into multimodal tokens"):
        _truncate_token_axis(batch, max_length=50)


def test_no_truncation_when_already_short():
    """A batch shorter than max_length is left untouched."""
    batch = _batch(seq=32, patches=64, media_span=slice(0, 4))

    _truncate_token_axis(batch, max_length=50)

    assert batch["input_ids"].shape == (1, 32)
    assert batch["pixel_values"].shape == (1, 64, 8)
