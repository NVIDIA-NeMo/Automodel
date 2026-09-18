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
from types import SimpleNamespace

import pytest
import torch

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

        assert bucket == (368, 672)

    def test_accepts_differing_sizes_that_share_an_aspect_ratio(self, tmp_path: Path) -> None:
        rows = [
            _triplet_row(tmp_path, 0, reference_size=(1536, 1024)),
            _triplet_row(tmp_path, 1, reference_size=(768, 512)),
        ]
        samples = preprocessing._read_manifest(_write_manifest(tmp_path, rows))

        bucket = preprocessing._resolve_shared_bucket(samples, max_pixels=512 * 512)

        assert bucket == (416, 624)

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


@requires_pillow
class TestPaddingResize:
    """Letterboxing must hit the exact bucket while preserving the source aspect ratio."""

    @pytest.mark.parametrize(
        ("source_height", "source_width"),
        [(1080, 1920), (1920, 1080), (480, 480), (100, 700)],
    )
    def test_output_matches_the_requested_bucket(self, source_height: int, source_width: int) -> None:
        image = preprocessing.np.zeros((source_height, source_width, 3), dtype=preprocessing.np.uint8)

        resized = preprocessing._padding_resize(image, height=256, width=512, resample="bilinear")

        assert resized.shape == (256, 512, 3)
        assert resized.dtype == preprocessing.np.uint8

    def test_pads_rather_than_stretches_a_mismatched_aspect_ratio(self) -> None:
        # A tall source letterboxed into a wide bucket must leave black bars on
        # the left and right, with content in the middle.
        image = preprocessing.np.full((400, 100, 3), 255, dtype=preprocessing.np.uint8)

        resized = preprocessing._padding_resize(image, height=256, width=512, resample="bilinear")

        assert resized[:, 0, :].max() == 0
        assert resized[:, -1, :].max() == 0
        assert resized[128, 256, :].max() > 0


@pytest.mark.parametrize("shape", [(1080, 1920), (257, 131), (101, 87)])
def test_reference_preprocessing_matches_released_pipeline(shape):
    """Bucket alignment, bicubic pixels, and black padding match Diffusers."""
    pytest.importorskip("diffusers", minversion="0.40.0")
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.wan_animate_2.encoders import WanAnimate2ProcessImagesInputStep
    from diffusers.modular_pipelines.wan_animate_2.video_processor import WanAnimate2VideoProcessor

    rng = preprocessing.np.random.default_rng(17)
    pixels = rng.integers(0, 256, (*shape, 3), dtype=preprocessing.np.uint8)
    processor = WanAnimate2VideoProcessor(resample="bicubic")
    components = SimpleNamespace(
        image_processor=processor, _execution_device=torch.device("cpu"), vae_scale_factor_spatial=8
    )
    state = PipelineState(values={"image": preprocessing.Image.fromarray(pixels), "height": 256, "width": 256})
    _, state = WanAnimate2ProcessImagesInputStep()(components, state)
    actual = preprocessing._frames_to_tensor(
        preprocessing._resize_by_area(pixels, target_area=256**2)[None], device=torch.device("cpu")
    )[:, :, 0]
    expected = state.get("image_pixels")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape[-2:] == preprocessing._bucket_dimensions(*shape, target_area=256**2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clip_features_match_released_encoder(dtype):
    """Match real CLIP outputs, including explicit bf16 input casting."""
    pytest.importorskip("diffusers", minversion="0.40.0")
    from diffusers.modular_pipelines.wan_animate_2.encoders import clip_visual_encode
    from transformers import CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(23)
    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=224, patch_size=14
    )
    encoder = CLIPVisionModel(config).eval().to(dtype=dtype)
    pixels = torch.randn(3, 33, 49)

    # The cache's public shape contract is ViT-H. Use a tiny real CLIP to test
    # its numerical path, retaining 1280 channels via a fixed output projection.
    class ProjectedCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.project = torch.nn.Linear(32, 1280, bias=False).to(dtype=dtype)

        def forward(self, pixel_values, output_hidden_states):
            """Map [batch, 3, 224, 224] pixels to [batch, 257, 1280] features."""
            output = self.encoder(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
            return SimpleNamespace(hidden_states=(self.project(output.hidden_states[-2]), None))

    image_encoder = ProjectedCLIP().eval()
    with torch.no_grad():
        actual = preprocessing._clip_visual_encode(image_encoder, pixels, device=torch.device("cpu"))
        expected = clip_visual_encode(image_encoder, pixels, device=torch.device("cpu"), dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_opencv
@requires_pillow
def test_manifest_encoding_writes_reusable_training_cache(tmp_path, monkeypatch):
    """Decode actual media and persist conditioning with frozen encoder stand-ins."""
    import torch.nn.functional as functional

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode(self, pixels):
            """Map [1, 3, frames, height, width] pixels to [1, 16, latent_frames, h/8, w/8]."""
            pooled = functional.avg_pool3d(pixels[:, :, ::4], (1, 8, 8)).mean(1, keepdim=True).repeat(1, 16, 1, 1, 1)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pooled))

        def decode(self, latents, return_dict):
            """Map [1, 16, frames, h, w] latents to a finite RGB clip."""
            output = functional.interpolate(
                latents[:, :3], size=((latents.shape[2] - 1) * 4 + 1, latents.shape[3] * 8, latents.shape[4] * 8)
            )
            return (output,)

    class TinyCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, pixel_values, output_hidden_states):
            """Map [1, 3, 224, 224] pixels to deterministic [1, 257, 1280] features."""
            return SimpleNamespace(hidden_states=(pixel_values.mean().expand(1, 257, 1280), None))

    class TinyText(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            """Embed [1, tokens] token IDs into [1, tokens, 16]."""
            return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float().expand(-1, -1, 16))

    def tokenizer(prompt, **kwargs):
        return {"input_ids": torch.tensor([[3, 5, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 0, 0]])}

    image = preprocessing.Image.new("RGB", (32, 32), color=(10, 90, 150))
    image.save(tmp_path / "ref.png")
    for name, level in (("drive.avi", 40), ("target.avi", 180)):
        writer = preprocessing.cv2.VideoWriter(
            str(tmp_path / name), preprocessing.cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (32, 32)
        )
        assert writer.isOpened()
        try:
            for frame in range(5):
                writer.write(preprocessing.np.full((32, 32, 3), level + frame, dtype=preprocessing.np.uint8))
        finally:
            writer.release()
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "reference_image": "ref.png",
                "driving_video": "drive.avi",
                "target_video": "target.avi",
                "caption": "moving",
            }
        ],
    )
    models = preprocessing._EncoderModels(
        vae=TinyVAE(),
        text_encoder=TinyText(),
        image_encoder=TinyCLIP(),
        tokenizer=tokenizer,
        latents_mean=torch.full((1, 16, 1, 1, 1), 0.2),
        latents_reciprocal_std=torch.full((1, 16, 1, 1, 1), 2.0),
    )
    encoder = preprocessing.WanAnimate2CacheEncoder(
        model_name="local-fixture", device="cpu", torch_dtype="float32", max_sequence_length=4
    )
    monkeypatch.setattr(encoder, "_load_models", lambda device: models)
    output_dir = tmp_path / "cache"
    metadata_path = encoder.encode_manifest(
        manifest_path=manifest, output_dir=output_dir, max_pixels=1024, num_frames=5, fps=24, verify=True
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["total_items"] == 1
    records = json.loads((output_dir / metadata["shards"][0]).read_text())
    payload = torch.load(records[0]["cache_file"], weights_only=True)
    assert payload["video_latents"].shape == (1, 16, 2, 4, 4)
    assert payload["reference_latents"].shape == (1, 16, 1, 4, 4)
    assert payload["clip_fea"].shape == (1, 257, 1280)
    assert not torch.equal(payload["driving_latents"], payload["video_latents"])
    torch.testing.assert_close(payload["cond_zero_latents"], torch.full((1, 16, 2, 4, 4), -0.4))
    assert torch.count_nonzero(payload["text_embeddings"][:, 2:]) == 0
    assert all(
        value.device.type == "cpu" and not value.requires_grad
        for value in payload.values()
        if isinstance(value, torch.Tensor)
    )
    with pytest.raises(ValueError, match="already holds a cache"):
        encoder.encode_manifest(manifest_path=manifest, output_dir=output_dir, max_pixels=1024, num_frames=5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_frames": 4}, r"4n \+ 1"),
        ({"fps": 0}, "fps must be positive"),
        ({"max_pixels": 64}, "max_pixels must be at least"),
        ({"num_gpus": 0}, "num_gpus must be positive"),
        ({"num_gpus": 2}, "explicit device"),
    ],
)
@requires_opencv
@requires_pillow
def test_invalid_encoding_requests_fail_before_loading_models(tmp_path, kwargs, message):
    encoder = preprocessing.WanAnimate2CacheEncoder(model_name="unused", device="cpu")
    (tmp_path / "unused.jsonl").write_text("")
    arguments = dict(manifest_path=tmp_path / "unused.jsonl", output_dir=tmp_path / "cache", max_pixels=1024)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        encoder.encode_manifest(**arguments)
