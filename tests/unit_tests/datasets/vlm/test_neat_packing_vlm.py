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

import torch

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import (
    _build_packed_vlm_sample,
    _compute_mrope_position_ids,
    _shift_sample,
    neat_pack_dataset_vlm,
)


class _FakeDataset:
    """Mimics PreTokenizedDatasetWrapper for testing."""

    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def _make_vlm_sample(seq_len, has_image=False, has_video=False):
    """Create a fake VLM sample (unshifted, as from PreTokenizedDatasetWrapper)."""
    sample = {
        "input_ids": torch.arange(1, seq_len + 1),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.arange(101, 101 + seq_len),
    }
    if has_image:
        n_images = 2
        sample["pixel_values"] = torch.randn(n_images, 3, 224, 224)
        sample["image_grid_thw"] = torch.tensor([[1, 224, 224]] * n_images)
    if has_video:
        n_videos = 1
        sample["pixel_values_videos"] = torch.randn(n_videos, 3, 224, 224)
        sample["video_grid_thw"] = torch.tensor([[1, 224, 224]] * n_videos)
    return sample


class TestShiftSample:
    def test_shift(self):
        sample = _make_vlm_sample(5)
        shifted = _shift_sample(sample)

        # input_ids[:-1] -> length 4
        assert shifted["input_ids"].shape == (4,)
        assert shifted["input_ids"].tolist() == [1, 2, 3, 4]

        # labels[1:] -> length 4
        assert shifted["labels"].shape == (4,)
        assert shifted["labels"].tolist() == [102, 103, 104, 105]

        # attention_mask[:-1] -> length 4
        assert shifted["attention_mask"].shape == (4,)

    def test_shift_preserves_media(self):
        sample = _make_vlm_sample(5, has_image=True)
        shifted = _shift_sample(sample)

        assert "pixel_values" in shifted
        assert shifted["pixel_values"].shape == sample["pixel_values"].shape


class TestBuildPackedVlmSample:
    def test_basic(self):
        # Pre-shifted samples
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([102, 103, 104]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([105, 106]),
            },
        ]
        result = _build_packed_vlm_sample(samples, pack_size=8, padding_idx=0)

        # No pre-padding: length = sum of samples (3 + 2 = 5)
        assert result["input_ids"].shape == (5,)
        assert result["input_ids"].tolist() == [1, 2, 3, 4, 5]
        assert result["attention_mask"].tolist() == [1, 1, 1, 2, 2]
        assert result["position_ids"].tolist() == [0, 1, 2, 0, 1]

    def test_media_concat(self):
        samples = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([10, 20]),
                "pixel_values": torch.randn(2, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 224, 224], [1, 224, 224]]),
            },
            {
                "input_ids": torch.tensor([3]),
                "labels": torch.tensor([30]),
                "pixel_values": torch.randn(1, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 224, 224]]),
            },
        ]
        result = _build_packed_vlm_sample(samples, pack_size=4, padding_idx=0)

        assert result["pixel_values"].shape[0] == 3  # 2 + 1
        assert result["image_grid_thw"].shape[0] == 3
        assert result["n_images"] == 3
        assert result["n_videos"] == 0


class TestNeatPackDatasetVlm:
    def test_end_to_end(self):
        samples = [
            _make_vlm_sample(5),  # after shift -> 4 tokens
            _make_vlm_sample(4),  # after shift -> 3 tokens
            _make_vlm_sample(3),  # after shift -> 2 tokens
        ]
        ds = _FakeDataset(samples)

        # pack_size=6, shifted lengths: 4, 3, 2 -> total 9 -> 2 packs
        packed = neat_pack_dataset_vlm(ds, pack_size=6, padding_idx=0)

        assert len(packed) >= 1

        for i in range(len(packed)):
            pack_len = len(packed[i]["input_ids"])
            assert pack_len <= 6  # no longer than pack_size
            assert pack_len > 0
            assert len(packed[i]["labels"]) == pack_len
            assert len(packed[i]["attention_mask"]) == pack_len
            pos = packed[i]["position_ids"]
            pos_len = pos.shape[-1] if isinstance(pos, torch.Tensor) else len(pos)
            assert pos_len == pack_len

    def test_autoregressive_shift(self):
        """Verify shift happens before packing."""
        sample = _make_vlm_sample(4)
        ds = _FakeDataset([sample])

        packed = neat_pack_dataset_vlm(ds, pack_size=5, padding_idx=0)

        ids = packed[0]["input_ids"]
        labels = packed[0]["labels"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        # Original: input_ids=[1,2,3,4], labels=[101,102,103,104]
        # After shift: input_ids=[1,2,3], labels=[102,103,104]
        assert ids[:3] == [1, 2, 3]
        assert labels[:3] == [102, 103, 104]

    def test_with_images(self):
        samples = [
            _make_vlm_sample(4, has_image=True),
            _make_vlm_sample(3, has_image=True),
        ]
        ds = _FakeDataset(samples)

        packed = neat_pack_dataset_vlm(ds, pack_size=8, padding_idx=0)

        # Check that pixel_values are present in the packed output
        assert any(packed[i].get("pixel_values") is not None for i in range(len(packed)))

    def test_n_images_count(self):
        samples = [
            _make_vlm_sample(4, has_image=True),  # 2 images
            _make_vlm_sample(3),  # 0 images
        ]
        ds = _FakeDataset(samples)

        packed = neat_pack_dataset_vlm(ds, pack_size=8, padding_idx=0)

        # All should fit in one pack
        assert len(packed) == 1
        assert packed[0]["n_images"] == 2
        assert packed[0]["n_videos"] == 0

    def test_drop_long(self):
        samples = [
            _make_vlm_sample(10),  # after shift -> 9 tokens, exceeds pack_size=6
            _make_vlm_sample(3),  # after shift -> 2 tokens, fits
        ]
        ds = _FakeDataset(samples)

        packed = neat_pack_dataset_vlm(
            ds,
            pack_size=6,
            padding_idx=0,
            drop_long_samples=True,
        )
        assert len(packed) == 1


class TestNeatPackedVlmCollater:
    def test_collater(self):
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0]),
                "labels": torch.tensor([10, 20, 30, -100]),
                "attention_mask": torch.tensor([1, 1, 2, 0]),
                "position_ids": torch.tensor([0, 1, 0, 0]),
                "n_images": 1,
                "n_videos": 0,
                "pixel_values": torch.randn(1, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 224, 224]]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7]),
                "labels": torch.tensor([40, 50, 60, 70]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "position_ids": torch.tensor([0, 1, 2, 3]),
                "n_images": 2,
                "n_videos": 0,
                "pixel_values": torch.randn(2, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 224, 224], [1, 224, 224]]),
            },
        ]
        result = neat_packed_vlm_collater(batch)

        assert result["input_ids"].shape == (2, 4)
        assert result["labels"].shape == (2, 4)
        assert result["position_ids"].shape == (2, 4)
        assert result["attention_mask"].shape == (2, 1, 4, 4)
        assert result["attention_mask"].dtype == torch.bool

        # pixel_values concatenated: 1 + 2 = 3
        assert result["pixel_values"].shape[0] == 3

        # n_images_per_sample
        assert result["n_images_per_sample"].tolist() == [1, 2]


class TestMRoPESupport:
    """Tests for mRoPE (multi-resolution RoPE) position_ids handling."""

    @staticmethod
    def _fake_get_rope_index(input_ids, attention_mask=None, **kwargs):
        """Simulate Qwen-VL's get_rope_index: returns [3, B, S] position_ids."""
        B, S = input_ids.shape
        # Simple mock: temporal=0..S-1, height=all zeros, width=all zeros
        t_pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        h_pos = torch.zeros(B, S, dtype=torch.long)
        w_pos = torch.zeros(B, S, dtype=torch.long)
        position_ids = torch.stack([t_pos, h_pos, w_pos], dim=0)  # [3, B, S]
        return position_ids, torch.zeros(B)

    def test_compute_mrope_position_ids(self):
        """Per-sample mRoPE position_ids are computed correctly."""
        sample = _make_vlm_sample(5)
        pos = _compute_mrope_position_ids(sample, self._fake_get_rope_index)

        assert pos is not None
        assert pos.shape == (3, 5)  # [3, seq_len]
        # temporal dimension should be 0,1,2,3,4
        assert pos[0].tolist() == [0, 1, 2, 3, 4]
        # height and width should be all zeros
        assert pos[1].tolist() == [0, 0, 0, 0, 0]

    def test_shift_with_mrope(self):
        """mRoPE position_ids are shifted correctly: [:, :-1]."""
        sample = _make_vlm_sample(5)
        sample["position_ids"] = torch.stack(
            [
                torch.arange(5),  # temporal
                torch.zeros(5, dtype=torch.long),  # height
                torch.zeros(5, dtype=torch.long),  # width
            ]
        )  # [3, 5]

        shifted = _shift_sample(sample, has_mrope=True)

        assert shifted["position_ids"].shape == (3, 4)  # [3, seq_len-1]
        assert shifted["position_ids"][0].tolist() == [0, 1, 2, 3]

    def test_build_packed_with_mrope(self):
        """Packed sample concatenates 3D position_ids correctly."""
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([10, 20, 30]),
                "position_ids": torch.tensor(
                    [
                        [0, 1, 2],  # temporal
                        [0, 0, 0],  # height
                        [0, 0, 0],  # width
                    ]
                ),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([40, 50]),
                "position_ids": torch.tensor(
                    [
                        [0, 1],  # temporal (reset)
                        [0, 0],  # height
                        [0, 0],  # width
                    ]
                ),
            },
        ]
        result = _build_packed_vlm_sample(samples, pack_size=7, padding_idx=0, has_mrope=True)

        # No pre-padding: [3, 5] (3+2 tokens)
        assert result["position_ids"].shape == (3, 5)
        # temporal: [0,1,2, 0,1] (concat, no pad)
        assert result["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
        # height: [0,0,0, 0,0]
        assert result["position_ids"][1].tolist() == [0, 0, 0, 0, 0]

    def test_end_to_end_with_mrope(self):
        """Full pipeline with get_rope_index produces 3D position_ids."""
        samples = [
            _make_vlm_sample(5),
            _make_vlm_sample(4),
        ]
        ds = _FakeDataset(samples)

        packed = neat_pack_dataset_vlm(
            ds,
            pack_size=8,
            padding_idx=0,
            get_rope_index=self._fake_get_rope_index,
        )

        assert len(packed) >= 1
        pos = packed[0]["position_ids"]
        # With mRoPE, position_ids should be 2D: [3, actual_len]
        if isinstance(pos, torch.Tensor):
            assert pos.ndim == 2
            assert pos.shape[0] == 3
            assert pos.shape[1] <= 8  # no longer than pack_size
            assert pos.shape[1] > 0

    def test_collater_with_mrope(self):
        """Collater stacks 3D mRoPE position_ids to [3, B, S]."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0]),
                "labels": torch.tensor([10, 20, 30, -100]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "position_ids": torch.tensor(
                    [
                        [0, 1, 2, 0],  # temporal
                        [0, 0, 0, 0],  # height
                        [0, 0, 0, 0],  # width
                    ]
                ),  # [3, 4]
                "n_images": 0,
                "n_videos": 0,
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7]),
                "labels": torch.tensor([40, 50, 60, 70]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "position_ids": torch.tensor(
                    [
                        [0, 1, 2, 3],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),  # [3, 4]
                "n_images": 0,
                "n_videos": 0,
            },
        ]
        result = neat_packed_vlm_collater(batch)

        # position_ids should be [3, B=2, S=4]
        assert result["position_ids"].shape == (3, 2, 4)
        assert result["position_ids"][0, 0].tolist() == [0, 1, 2, 0]
        assert result["position_ids"][0, 1].tolist() == [0, 1, 2, 3]
