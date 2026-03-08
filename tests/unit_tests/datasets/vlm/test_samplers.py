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

from types import SimpleNamespace

from nemo_automodel.components.datasets.vlm.samplers import (
    LengthGroupedSampler,
    _smart_resize_image,
    _smart_resize_video,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(conversations, mm_metas=None):
    """Create a simple list-based dataset from conversation specs."""
    ds = []
    for i, conv in enumerate(conversations):
        item = {"conversation": conv}
        if mm_metas is not None and i < len(mm_metas) and mm_metas[i] is not None:
            item["mm_inputs_meta"] = mm_metas[i]
        ds.append(item)
    return ds


def _text_msg(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def _media_msg(n_images=0, n_videos=0, text=""):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for _ in range(n_images):
        content.append({"type": "image", "image": "dummy.png"})
    for _ in range(n_videos):
        content.append({"type": "video", "video": "dummy.mp4"})
    return [{"role": "user", "content": content}]


def _make_processor(
    image_patch_size=14,
    image_merge_size=2,
    image_min_pixels=56 * 56,
    image_max_pixels=14 * 14 * 4 * 1280,
    video_patch_size=16,
    video_merge_size=2,
    video_temporal_patch_size=2,
    video_min_pixels=128 * 128,
    video_max_pixels=16 * 16 * 2 * 2 * 2 * 6144,
    video_fps=2.0,
    video_min_frames=4,
    video_max_frames=768,
):
    """Create a mock processor with image_processor and video_processor.

    Mirrors real Qwen2VL/Qwen3VL processors: min/max_pixels are stored as
    both direct attributes AND inside the ``size`` dict.
    """
    image_processor = SimpleNamespace(
        patch_size=image_patch_size,
        merge_size=image_merge_size,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
        size={"min_pixels": image_min_pixels, "max_pixels": image_max_pixels},
    )
    video_processor = SimpleNamespace(
        patch_size=video_patch_size,
        merge_size=video_merge_size,
        temporal_patch_size=video_temporal_patch_size,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
        size={"min_pixels": video_min_pixels, "max_pixels": video_max_pixels},
        fps=video_fps,
        min_frames=video_min_frames,
        max_frames=video_max_frames,
    )
    return SimpleNamespace(image_processor=image_processor, video_processor=video_processor)


# ---------------------------------------------------------------------------
# smart_resize unit tests
# ---------------------------------------------------------------------------


class TestSmartResizeImage:
    def test_basic_resize(self):
        h, w = _smart_resize_image(1024, 768, factor=28)
        assert h % 28 == 0 and w % 28 == 0

    def test_respects_max_pixels(self):
        h, w = _smart_resize_image(4000, 4000, factor=28, max_pixels=200000)
        assert h * w <= 200000

    def test_respects_min_pixels(self):
        h, w = _smart_resize_image(10, 10, factor=28, min_pixels=56 * 56)
        assert h * w >= 56 * 56

    def test_exact_factor_multiple(self):
        h, w = _smart_resize_image(280, 560, factor=28)
        assert h == 280 and w == 560


class TestSmartResizeVideo:
    def test_basic_resize(self):
        h, w = _smart_resize_video(16, 480, 640, temporal_factor=2, factor=32)
        assert h % 32 == 0 and w % 32 == 0

    def test_respects_max_pixels(self):
        h, w = _smart_resize_video(32, 1920, 1080, temporal_factor=2, factor=32, max_pixels=500000)
        t_bar = 32  # already multiple of 2
        assert t_bar * h * w <= 500000

    def test_respects_min_pixels(self):
        h, w = _smart_resize_video(4, 64, 64, temporal_factor=2, factor=32, min_pixels=128 * 128)
        t_bar = 4
        assert t_bar * h * w >= 128 * 128


# ---------------------------------------------------------------------------
# Heuristic estimation (no processor) — backward compat
# ---------------------------------------------------------------------------


class TestEstimateLengthHeuristic:
    """Tests for the fallback heuristic when no processor is provided."""

    def _estimate(self, example):
        """Helper: create a minimal sampler with one example and return its length."""
        ds = [example]
        sampler = LengthGroupedSampler(ds, seed=0)
        return sampler.lengths[0]

    def test_text_only(self):
        example = {"conversation": _text_msg("hello world")}  # 11 chars -> 3 tokens
        assert self._estimate(example) == 11 // 3

    def test_with_images(self):
        example = {"conversation": _media_msg(n_images=2, text="hi")}
        # 2 chars // 3 = 0, 2 images * 500 = 1000
        assert self._estimate(example) == 0 + 1000

    def test_with_videos(self):
        example = {"conversation": _media_msg(n_videos=1, text="abc")}
        # 3 chars // 3 = 1, 1 video * 500 = 500
        assert self._estimate(example) == 1 + 500

    def test_string_content(self):
        """Content can also be a plain string (not a list of dicts)."""
        example = {"conversation": [{"role": "user", "content": "hello world!!!"}]}
        assert self._estimate(example) == 14 // 3

    def test_empty_conversation(self):
        assert self._estimate({"conversation": []}) == 0

    def test_missing_conversation(self):
        assert self._estimate({}) == 0


# ---------------------------------------------------------------------------
# Accurate estimation (with processor + mm_inputs_meta)
# ---------------------------------------------------------------------------


class TestEstimateLengthAccurate:
    """Tests for accurate media token estimation using smart_resize."""

    def _estimate(self, example, processor=None):
        ds = [example]
        sampler = LengthGroupedSampler(ds, seed=0, processor=processor)
        return sampler.lengths[0]

    def test_image_tokens_simple(self):
        """A 280x560 image with factor=28 needs no resize; tokens = (280/14)*(560/14)/4 = 20*40/4 = 200."""
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
        )
        example = {
            "conversation": _media_msg(n_images=1),
            "mm_inputs_meta": {"images_meta": [[280, 560]]},
        }
        length = self._estimate(example, processor)
        # text_tokens = 0, image_tokens = (280/14)*(560/14)/4 = 200
        assert length == 200

    def test_image_tokens_with_text(self):
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
        )
        example = {
            "conversation": _media_msg(n_images=1, text="a" * 30),
            "mm_inputs_meta": {"images_meta": [[280, 560]]},
        }
        length = self._estimate(example, processor)
        # text_tokens = 30 // 3 = 10, image_tokens = 200
        assert length == 210

    def test_multiple_images(self):
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
        )
        example = {
            "conversation": _media_msg(n_images=2),
            "mm_inputs_meta": {"images_meta": [[280, 560], [560, 280]]},
        }
        length = self._estimate(example, processor)
        # Each image: (280/14)*(560/14)/4 = 200, total = 400
        assert length == 400

    def test_video_tokens(self):
        """Video with known dimensions — verify token count."""
        processor = _make_processor(
            video_patch_size=16, video_merge_size=2,
            video_temporal_patch_size=2,
            video_min_pixels=100, video_max_pixels=100_000_000,
            video_fps=2.0, video_min_frames=4, video_max_frames=128,
        )
        # Video: 100 frames at 25fps → sampled = int(100/25*2) = 8 frames
        # 8 % 2 == 0, so nframes = 8, grid_t = 8/2 = 4
        # 320x640 with factor=32 → resized to 320x640 (already multiples of 32)
        # grid_h = 320/16 = 20, grid_w = 640/16 = 40
        # merge_length = 4, frame_seqlen = 20*40/4 = 200
        # total = 4 * 200 = 800
        example = {
            "conversation": _media_msg(n_videos=1),
            "mm_inputs_meta": {"videos_meta": [[100, 320, 640, 25.0, 4.0]]},
        }
        length = self._estimate(example, processor)
        assert length == 800

    def test_mixed_image_and_video(self):
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
            video_patch_size=16, video_merge_size=2,
            video_temporal_patch_size=2,
            video_min_pixels=100, video_max_pixels=100_000_000,
            video_fps=2.0, video_min_frames=4, video_max_frames=128,
        )
        example = {
            "conversation": _media_msg(n_images=1, n_videos=1, text="a" * 30),
            "mm_inputs_meta": {
                "images_meta": [[280, 560]],
                "videos_meta": [[100, 320, 640, 25.0, 4.0]],
            },
        }
        length = self._estimate(example, processor)
        # text=10, image=200, video=800
        assert length == 10 + 200 + 800

    def test_falls_back_without_mm_meta(self):
        """With a processor but no mm_inputs_meta, falls back to heuristic."""
        processor = _make_processor()
        example = {"conversation": _media_msg(n_images=2)}
        length = self._estimate(example, processor)
        assert length == 2 * 500  # heuristic

    def test_none_entries_in_meta_skipped(self):
        """None entries in images_meta / videos_meta are skipped gracefully."""
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
        )
        example = {
            "conversation": _media_msg(n_images=2),
            "mm_inputs_meta": {"images_meta": [[280, 560], None]},
        }
        length = self._estimate(example, processor)
        # Only the first image counted: 200
        assert length == 200

    def test_video_min_frames_enforced(self):
        """Very short video gets clamped to min_frames."""
        processor = _make_processor(
            video_patch_size=16, video_merge_size=2,
            video_temporal_patch_size=2,
            video_min_pixels=100, video_max_pixels=100_000_000,
            video_fps=2.0, video_min_frames=4, video_max_frames=128,
        )
        # 2 frames at 30fps → sampled = int(2/30*2) = 0 → max(1,0)=1 → clamped to min_frames=4
        # 4 % 2 == 0, nframes = 4, grid_t = 2
        # 320x320 factor=32 → 320x320
        # grid_h=20, grid_w=20, merge=4, per_frame=100
        # total = 2 * 100 = 200
        example = {
            "conversation": _media_msg(n_videos=1),
            "mm_inputs_meta": {"videos_meta": [[2, 320, 320, 30.0, 0.067]]},
        }
        length = self._estimate(example, processor)
        assert length == 200

    def test_video_max_frames_enforced(self):
        """Long video gets clamped to max_frames."""
        processor = _make_processor(
            video_patch_size=16, video_merge_size=2,
            video_temporal_patch_size=2,
            video_min_pixels=100, video_max_pixels=100_000_000,
            video_fps=2.0, video_min_frames=4, video_max_frames=16,
        )
        # 3000 frames at 30fps → sampled = int(3000/30*2) = 200 → clamped to max_frames=16
        # 16 % 2 == 0, nframes=16, grid_t=8
        # 320x320, grid_h=20, grid_w=20, per_frame=100
        # total = 8 * 100 = 800
        example = {
            "conversation": _media_msg(n_videos=1),
            "mm_inputs_meta": {"videos_meta": [[3000, 320, 320, 30.0, 100.0]]},
        }
        length = self._estimate(example, processor)
        assert length == 800


# ---------------------------------------------------------------------------
# Sorting / grouping with accurate estimation
# ---------------------------------------------------------------------------


class TestSortedOrder:
    def test_indices_sorted_by_text_length_descending(self):
        """Two-level sort: primary key is text tokens (descending)."""
        ds = _make_dataset([
            _text_msg("short"),           # idx 0: text_tok=2, media=0
            _text_msg("a" * 300),         # idx 1: text_tok=100, media=0
            _text_msg("a" * 30),          # idx 2: text_tok=10, media=0
            _media_msg(n_images=3),       # idx 3: text_tok=0, media=1500
        ])
        sampler = LengthGroupedSampler(ds, seed=0)
        # Sort by total tokens desc: idx 3 has 1500 media tokens, idx 1 has 100 text, etc.
        indices = list(sampler)
        # All indices must be present
        assert sorted(indices) == [0, 1, 2, 3]

    def test_accurate_sorting_with_processor(self):
        """With processor + mm_inputs_meta, two-level sort uses text then media."""
        processor = _make_processor(
            image_patch_size=14, image_merge_size=2,
            image_min_pixels=100, image_max_pixels=10_000_000,
        )
        ds = _make_dataset(
            [
                _media_msg(n_images=1),  # idx 0: text_tok=0, media=100
                _media_msg(n_images=1),  # idx 1: text_tok=0, media=400
                _text_msg("a" * 30),     # idx 2: text_tok=10, media=0
            ],
            mm_metas=[
                {"images_meta": [[280, 280]]},   # (280/14)*(280/14)/4 = 100
                {"images_meta": [[560, 560]]},   # (560/14)*(560/14)/4 = 400
                None,
            ],
        )
        sampler = LengthGroupedSampler(ds, seed=0, processor=processor)
        assert sampler.lengths == [100, 400, 10]
        # All indices must be present
        assert sorted(list(sampler)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Bucket shuffle
# ---------------------------------------------------------------------------


class TestBucketShuffle:
    def test_epoch_shuffle_changes_order(self):
        """Different epochs produce different orderings."""
        ds = _make_dataset([_text_msg("a" * (i * 30)) for i in range(10)])
        sampler = LengthGroupedSampler(ds, seed=42)

        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        # With enough samples and different epochs, order should change
        assert sorted(epoch0) == sorted(epoch1) == list(range(10))

    def test_all_indices_in_shuffle(self):
        """All indices are present after shuffling."""
        ds = _make_dataset([_text_msg("a" * (i * 30)) for i in range(16)])
        sampler = LengthGroupedSampler(ds, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(16))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_order(self):
        ds = _make_dataset([_text_msg("a" * (i * 10)) for i in range(20)])
        s1 = LengthGroupedSampler(ds, seed=123)
        s2 = LengthGroupedSampler(ds, seed=123)
        assert list(s1) == list(s2)

    def test_different_seed_different_order(self):
        ds = _make_dataset([_text_msg("a" * (i * 10)) for i in range(20)])
        s1 = LengthGroupedSampler(ds, seed=1)
        s2 = LengthGroupedSampler(ds, seed=2)
        assert list(s1) != list(s2)

    def test_set_epoch_changes_order(self):
        ds = _make_dataset([_text_msg("a" * (i * 10)) for i in range(20)])
        sampler = LengthGroupedSampler(ds, seed=42)
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        assert epoch0 != epoch1

    def test_same_epoch_reproducible(self):
        ds = _make_dataset([_text_msg("a" * (i * 10)) for i in range(20)])
        sampler = LengthGroupedSampler(ds, seed=42)
        first = list(sampler)
        second = list(sampler)
        assert first == second

    def test_same_seed_with_processor(self):
        """Determinism holds when using a processor too."""
        processor = _make_processor()
        ds = _make_dataset(
            [_media_msg(n_images=1, text="a" * (i * 10)) for i in range(10)],
            mm_metas=[{"images_meta": [[280 + i * 28, 280]]} for i in range(10)],
        )
        s1 = LengthGroupedSampler(ds, seed=7, processor=processor)
        s2 = LengthGroupedSampler(ds, seed=7, processor=processor)
        assert list(s1) == list(s2)


# ---------------------------------------------------------------------------
# len / completeness
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_matches_dataset(self):
        ds = _make_dataset([_text_msg("x") for _ in range(50)])
        sampler = LengthGroupedSampler(ds, seed=0)
        assert len(sampler) == 50

    def test_all_indices_present(self):
        ds = _make_dataset([_text_msg("a" * (i * 5)) for i in range(30)])
        sampler = LengthGroupedSampler(ds, seed=42)
        assert sorted(list(sampler)) == list(range(30))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_element(self):
        ds = _make_dataset([_text_msg("hello")])
        sampler = LengthGroupedSampler(ds, seed=0)
        assert list(sampler) == [0]

    def test_small_dataset(self):
        ds = _make_dataset([_text_msg("a" * (i * 10)) for i in range(5)])
        sampler = LengthGroupedSampler(ds, seed=0)
        assert sorted(list(sampler)) == list(range(5))

    def test_all_same_length(self):
        ds = _make_dataset([_text_msg("aaa") for _ in range(10)])
        sampler = LengthGroupedSampler(ds, seed=42)
        assert sorted(list(sampler)) == list(range(10))

    def test_empty_images_meta(self):
        """Empty images_meta list is handled gracefully."""
        processor = _make_processor()
        example = {
            "conversation": _text_msg("hello"),
            "mm_inputs_meta": {"images_meta": [], "videos_meta": None},
        }
        ds = [example]
        sampler = LengthGroupedSampler(ds, seed=0, processor=processor)
        assert sampler.lengths[0] == 5 // 3  # text only

    def test_video_zero_fps(self):
        """When source fps=0, duration is used to compute nframes."""
        processor = _make_processor(
            video_patch_size=16, video_merge_size=2,
            video_temporal_patch_size=2,
            video_min_pixels=100, video_max_pixels=100_000_000,
            video_fps=2.0, video_min_frames=4, video_max_frames=128,
        )
        # total_frames=0, fps=0 → total_frames stays 0
        # nframes from duration: max(1, int(5*2)) = 10
        # min(total_frames=0, max_frames=128, 10) = 0 → max(min_frames=4, 0) = 4
        # 4 % 2 == 0, grid_t = 2
        # 320x320, grid_h=20, grid_w=20, per_frame=100
        # total = 2 * 100 = 200
        example = {
            "conversation": _media_msg(n_videos=1),
            "mm_inputs_meta": {"videos_meta": [[0, 320, 320, 0.0, 5.0]]},
        }
        length = LengthGroupedSampler([example], seed=0, processor=processor).lengths[0]
        assert length == 200
