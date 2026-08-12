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

"""CPU unit tests for Wan-Animate-2 triplet cache preprocessing.

These cover the pure-Python geometry, manifest validation, and frame-selection
helpers. Encoding itself needs the frozen conditioning stack on a GPU and is out
of scope for a unit test.

Expected values are derived from the contracts the helpers must satisfy rather
than from their own arithmetic:

* Bucket dimensions are checked against the upstream ``resize_by_area`` rule --
  both axes divisible by 16, area within the budget, and the aspect ratio
  preserved to within one 16-pixel step -- not by recomputing the same formula.
* Frame indices are checked against the physical meaning of resampling: index
  ``i`` must correspond to time ``i / fps`` in the source clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_automodel.components.models.wan_animate2 import preprocessing

# The preprocessing module guards NumPy, OpenCV and Pillow with ``safe_import``,
# so it imports cleanly without them and its pure-logic helpers stay exercisable.
# Skipping the whole module on the strictest of the three would leave bucket
# arithmetic, manifest parsing and frame resampling untested wherever OpenCV is
# absent, which is the common case for a unit-test environment.
requires_pillow = pytest.mark.skipif(
    not preprocessing.PIL_AVAILABLE,
    reason="Reading image headers requires Pillow",
)
requires_opencv = pytest.mark.skipif(
    not (preprocessing.NUMPY_AVAILABLE and preprocessing.CV2_AVAILABLE),
    reason="Letterboxing requires NumPy and OpenCV",
)

_SPATIAL_DIVISOR = 16


def _write_media(directory: Path, name: str, *, size: tuple[int, int] | None = None) -> Path:
    """Create a placeholder media file, optionally a real image.

    Args:
        directory: Directory to create the file in.
        name: File name.
        size: Optional ``(width, height)``. When given, a real RGB PNG is
            written so Pillow can read its header; otherwise the file holds
            arbitrary bytes.

    Returns:
        Path to the created file.
    """
    path = directory / name
    if size is None:
        path.write_bytes(b"placeholder")
        return path
    preprocessing.Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def _write_manifest(directory: Path, rows: list[dict], *, name: str = "manifest.jsonl") -> Path:
    """Write a JSONL manifest.

    Args:
        directory: Directory to write into.
        rows: Manifest rows, serialized one per line.
        name: Manifest file name.

    Returns:
        Path to the written manifest.
    """
    path = directory / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _triplet_row(directory: Path, index: int, *, reference_size: tuple[int, int]) -> dict:
    """Create one valid manifest row with its media files on disk.

    Args:
        directory: Directory to create media in.
        index: Row index, used to name the files.
        reference_size: ``(width, height)`` of the reference image.

    Returns:
        A manifest row referencing the created files by relative path.
    """
    _write_media(directory, f"ref_{index}.png", size=reference_size)
    _write_media(directory, f"drive_{index}.mp4")
    _write_media(directory, f"target_{index}.mp4")
    return {
        "reference_image": f"ref_{index}.png",
        "driving_video": f"drive_{index}.mp4",
        "target_video": f"target_{index}.mp4",
        "caption": f"caption {index}",
    }


class TestBucketDimensions:
    """``_bucket_dimensions`` must reproduce the upstream ``resize_by_area`` contract."""

    @pytest.mark.parametrize(
        ("height", "width"),
        [(1080, 1920), (1920, 1080), (512, 512), (720, 1280), (2720, 1536), (480, 640)],
    )
    @pytest.mark.parametrize("target_area", [256 * 256, 512 * 512, 768 * 768])
    def test_axes_are_16_aligned_and_within_budget(self, height: int, width: int, target_area: int) -> None:
        bucket_height, bucket_width = preprocessing._bucket_dimensions(height, width, target_area=target_area)

        assert bucket_height > 0 and bucket_width > 0
        assert bucket_height % _SPATIAL_DIVISOR == 0
        assert bucket_width % _SPATIAL_DIVISOR == 0
        assert bucket_height * bucket_width <= target_area

    @pytest.mark.parametrize(("height", "width"), [(1080, 1920), (2720, 1536), (480, 640)])
    def test_aspect_ratio_is_preserved_within_one_alignment_step(self, height: int, width: int) -> None:
        target_area = 512 * 512
        bucket_height, bucket_width = preprocessing._bucket_dimensions(height, width, target_area=target_area)

        source_ratio = width / height
        bucket_ratio = bucket_width / bucket_height
        # Flooring each axis to a multiple of 16 can move the ratio by at most
        # one step on either axis.
        tolerance = source_ratio * (_SPATIAL_DIVISOR / min(bucket_height, bucket_width))
        assert bucket_ratio == pytest.approx(source_ratio, abs=tolerance)

    def test_latent_grid_divides_by_the_transformer_patch_size(self) -> None:
        # 16-alignment exists so the latent grid (8x VAE downsample) stays
        # divisible by the transformer's (2, 2) spatial patch.
        bucket_height, bucket_width = preprocessing._bucket_dimensions(1080, 1920, target_area=512 * 512)

        assert (bucket_height // 8) % 2 == 0
        assert (bucket_width // 8) % 2 == 0

    def test_rejects_a_budget_too_small_to_produce_an_aligned_bucket(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            preprocessing._bucket_dimensions(1080, 1920, target_area=64)


@requires_pillow
class TestResolveSharedBucket:
    """A cache must be structurally single-bucket; mixed aspect ratios must fail fast."""

    def test_returns_the_shared_bucket_for_uniform_reference_images(self, tmp_path: Path) -> None:
        rows = [_triplet_row(tmp_path, index, reference_size=(1920, 1080)) for index in range(3)]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        bucket = preprocessing._resolve_shared_bucket(samples, max_pixels=512 * 512)

        assert bucket == preprocessing._bucket_dimensions(1080, 1920, target_area=512 * 512)

    def test_accepts_differing_sizes_that_share_an_aspect_ratio(self, tmp_path: Path) -> None:
        rows = [
            _triplet_row(tmp_path, 0, reference_size=(1920, 1080)),
            _triplet_row(tmp_path, 1, reference_size=(1280, 720)),
        ]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        bucket = preprocessing._resolve_shared_bucket(samples, max_pixels=512 * 512)

        assert bucket == preprocessing._bucket_dimensions(1080, 1920, target_area=512 * 512)

    def test_rejects_a_manifest_spanning_multiple_buckets_and_names_an_offender(self, tmp_path: Path) -> None:
        rows = [
            _triplet_row(tmp_path, 0, reference_size=(1920, 1080)),
            _triplet_row(tmp_path, 1, reference_size=(1080, 1920)),
        ]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        with pytest.raises(ValueError, match="single resolution bucket") as excinfo:
            preprocessing._resolve_shared_bucket(samples, max_pixels=512 * 512)

        assert "ref_0.png" in str(excinfo.value) or "ref_1.png" in str(excinfo.value)


class TestReadManifest:
    """Manifest validation must reject malformed rows before any GPU work starts."""

    def test_parses_rows_and_resolves_relative_paths(self, tmp_path: Path) -> None:
        rows = [_triplet_row(tmp_path, index, reference_size=(640, 480)) for index in range(2)]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        assert [sample.caption for sample in samples] == ["caption 0", "caption 1"]
        assert [sample.row_index for sample in samples] == [0, 1]
        assert all(sample.reference_path.is_absolute() for sample in samples)
        assert samples[0].reference_path == (tmp_path / "ref_0.png").resolve()

    def test_defaults_the_identifier_to_the_row_index(self, tmp_path: Path) -> None:
        rows = [_triplet_row(tmp_path, index, reference_size=(640, 480)) for index in range(2)]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        assert [sample.identifier for sample in samples] == ["0", "1"]

    def test_honors_an_explicit_identifier(self, tmp_path: Path) -> None:
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480)) | {"id": "clip-a"}
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, [row]))

        assert samples[0].identifier == "clip-a"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480))
        path = tmp_path / "manifest.jsonl"
        path.write_text(f"\n{json.dumps(row)}\n\n", encoding="utf-8")

        assert len(preprocessing._read_manifest(path)) == 1

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            preprocessing._read_manifest(path)

    def test_rejects_a_non_object_row(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text('["not", "an", "object"]\n', encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            preprocessing._read_manifest(path)

    @pytest.mark.parametrize("field", ["reference_image", "driving_video", "target_video"])
    def test_rejects_a_missing_media_field(self, tmp_path: Path, field: str) -> None:
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480))
        del row[field]

        with pytest.raises(ValueError, match=field):
            preprocessing._read_manifest(_write_manifest(tmp_path, [row]))

    def test_rejects_a_non_string_caption(self, tmp_path: Path) -> None:
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480)) | {"caption": 17}

        with pytest.raises(ValueError, match="caption"):
            preprocessing._read_manifest(_write_manifest(tmp_path, [row]))

    def test_rejects_a_boolean_identifier(self, tmp_path: Path) -> None:
        # bool is an int subclass, so this would otherwise slip through.
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480)) | {"id": True}

        with pytest.raises(ValueError, match="id must be"):
            preprocessing._read_manifest(_write_manifest(tmp_path, [row]))

    def test_rejects_a_missing_media_file(self, tmp_path: Path) -> None:
        row = _triplet_row(tmp_path, 0, reference_size=(640, 480)) | {"driving_video": "absent.mp4"}

        with pytest.raises(FileNotFoundError, match="driving_video"):
            preprocessing._read_manifest(_write_manifest(tmp_path, [row]))


class TestResampleFrameIndices:
    """Frame selection must realize a physical frame rate, matching inference."""

    def test_matching_rates_select_consecutive_frames(self) -> None:
        indices = preprocessing._resample_frame_indices(300, 24.0, num_frames=9, fps=24)

        assert indices == list(range(9))

    def test_downsampling_preserves_wall_clock_time(self) -> None:
        source_fps, fps, num_frames = 30.0, 24, 9
        indices = preprocessing._resample_frame_indices(300, source_fps, num_frames=num_frames, fps=fps)

        # Index i must land on the source frame nearest to time i / fps.
        expected = [round(i / fps * source_fps) for i in range(num_frames)]
        assert indices == expected

    def test_upsampling_repeats_source_frames(self) -> None:
        indices = preprocessing._resample_frame_indices(300, 12.0, num_frames=9, fps=24)

        assert indices == [round(i / 24 * 12.0) for i in range(9)]
        assert len(indices) > len(set(indices))

    def test_always_returns_the_requested_count(self) -> None:
        for source_fps in (12.0, 23.976, 24.0, 25.0, 29.97, 60.0):
            indices = preprocessing._resample_frame_indices(1000, source_fps, num_frames=81, fps=24)
            assert len(indices) == 81

    def test_clamps_to_the_last_available_frame_for_short_clips(self) -> None:
        indices = preprocessing._resample_frame_indices(5, 30.0, num_frames=9, fps=24)

        assert len(indices) == 9
        assert max(indices) == 4
        assert indices == sorted(indices)

    def test_indices_are_non_decreasing(self) -> None:
        indices = preprocessing._resample_frame_indices(1000, 29.97, num_frames=81, fps=24)

        assert indices == sorted(indices)
        assert min(indices) >= 0


@requires_opencv
class TestPaddingResize:
    """Letterboxing must hit the exact bucket while preserving the source aspect ratio."""

    @pytest.mark.parametrize(
        ("source_height", "source_width"),
        [(1080, 1920), (1920, 1080), (480, 480), (100, 700)],
    )
    def test_output_matches_the_requested_bucket(self, source_height: int, source_width: int) -> None:
        image = preprocessing.np.zeros((source_height, source_width, 3), dtype=preprocessing.np.uint8)

        resized = preprocessing._padding_resize(
            image, height=256, width=512, interpolation=preprocessing.cv2.INTER_LINEAR
        )

        assert resized.shape == (256, 512, 3)
        assert resized.dtype == preprocessing.np.uint8

    def test_pads_rather_than_stretches_a_mismatched_aspect_ratio(self) -> None:
        # A tall source letterboxed into a wide bucket must leave black bars on
        # the left and right, with content in the middle.
        image = preprocessing.np.full((400, 100, 3), 255, dtype=preprocessing.np.uint8)

        resized = preprocessing._padding_resize(
            image, height=256, width=512, interpolation=preprocessing.cv2.INTER_LINEAR
        )

        assert resized[:, 0, :].max() == 0
        assert resized[:, -1, :].max() == 0
        assert resized[128, 256, :].max() > 0
