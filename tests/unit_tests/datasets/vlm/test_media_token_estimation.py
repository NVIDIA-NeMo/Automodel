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

"""Anti-drift parity tests for VLM media token estimation.

Packing plans bins from estimated token counts; if those estimates drift from
the real processor expansion, whole documents are silently dropped when packs
are materialized.  The tests here assert that every estimation tier of
``MediaTokenEstimator`` produces EXACTLY the number of media placeholder
tokens that a real HuggingFace processor emits — for a matrix of image and
video sizes.  If one of these tests fails, the estimator (most likely the
local Qwen smart_resize replica) has drifted from the installed transformers
version: fix the estimator, do not loosen the assertion.

The processors are constructed offline (default image/video processors plus a
tiny byte-level BPE tokenizer written to a temp dir), so no network or HF
cache is needed.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen3VLVideoProcessor,
)

from nemo_automodel.components.datasets.vlm.media_token_estimation import (
    DEFAULT_TOKENS_PER_MEDIA_ITEM,
    MediaTokenEstimator,
    _extract_image_geometry,
    _extract_video_geometry,
    _smart_resize_image,
    _smart_resize_video,
)
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _estimate_sample_length
from nemo_automodel.components.datasets.vlm.samplers import LengthGroupedSampler

# (height, width) matrix: factor-aligned, tiny (upscaled to min_pixels),
# typical, ragged aspect ratio, and 4K (downscaled to max_pixels).
IMAGE_SIZES = [
    (56, 56),
    (100, 100),
    (224, 224),
    (768, 1024),
    (1024, 768),
    (1234, 371),
    (3840, 2160),
]

# (num_frames, height, width); frame counts are multiples of
# temporal_patch_size so the source-fps metadata keeps every frame.
VIDEO_SIZES = [
    (8, 320, 640),
    (16, 720, 1280),
    (4, 240, 240),
    (12, 1080, 1920),
]


@pytest.fixture(scope="module")
def _offline_tokenizer(tmp_path_factory):
    """Tiny byte-level BPE tokenizer, so no checkpoint download is needed."""
    tokenizer_dir = tmp_path_factory.mktemp("qwen2_tokenizer")
    vocab = {token: idx for idx, token in enumerate(["!", '"', "#", "$", "%", "a", "b", "c", "<|endoftext|>"])}
    (tokenizer_dir / "vocab.json").write_text(json.dumps(vocab))
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]}
    )
    return tokenizer


@pytest.fixture(
    scope="module",
    params=["default_pixel_bounds", "overridden_pixel_bounds"],
)
def qwen2_vl_processor(request, _offline_tokenizer):
    """Real ``Qwen2VLProcessor``, with default and recipe-overridden pixel bounds.

    Recipes routinely pass ``min_pixels``/``max_pixels`` overrides, which change
    the resize target and therefore the token count; an estimator that reads
    stale defaults instead of the processor's live settings has drifted, so both
    variants are exercised.
    """
    image_processor = (
        Qwen2VLImageProcessor()
        if request.param == "default_pixel_bounds"
        else Qwen2VLImageProcessor(min_pixels=262144, max_pixels=4194304)
    )
    return Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=_offline_tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
    )


def _real_image_token_count(processor, height: int, width: int) -> int:
    """Number of image placeholder tokens the processor REALLY produces.

    Runs the full processor (image preprocessing + text expansion) on a
    synthetic image and counts the image placeholder id in the resulting
    ``input_ids``.
    """
    image = Image.new("RGB", (width, height))
    encoded = processor(text=processor.image_token, images=[image])
    input_ids = encoded["input_ids"]
    sequence = input_ids[0] if isinstance(input_ids[0], list) else input_ids
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    return sum(1 for token_id in sequence if token_id == image_token_id)


class _HidingProxy:
    """Forwards to a real processor while hiding selected attributes.

    Hiding ``_get_num_multimodal_tokens`` forces the probe tier; additionally
    hiding ``tokenizer`` forces the Qwen geometry tier; additionally hiding
    ``image_processor`` disables the geometry tier.
    """

    def __init__(self, wrapped, hidden_attrs):
        self._wrapped = wrapped
        self._hidden_attrs = frozenset(hidden_attrs)

    def __getattr__(self, name):
        if name in self._hidden_attrs:
            raise AttributeError(name)
        return getattr(self._wrapped, name)

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Image parity: every tier must equal the processor's real expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestImageParityWithRealProcessor:
    """Every estimation tier == real processor expansion, per image size.

    Each test disables the higher tiers so a silent fall-through cannot mask a
    broken tier: with the remaining tiers disabled, a fall-through would hit
    the flat 500-token default and fail the exact comparison.
    """

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_processor_api_tier_matches_real_expansion(self, qwen2_vl_processor, height, width):
        """Tier 1: counts from processor._get_num_multimodal_tokens."""
        proxy = _HidingProxy(qwen2_vl_processor, ["tokenizer", "image_processor"])
        estimator = MediaTokenEstimator(proxy)
        real = _real_image_token_count(qwen2_vl_processor, height, width)
        assert estimator.estimate_media_tokens(images_meta=[[height, width]]) == real

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_probe_tier_matches_real_expansion(self, qwen2_vl_processor, height, width):
        """Tier 2: counts probed by running the real processor once."""
        proxy = _HidingProxy(qwen2_vl_processor, ["_get_num_multimodal_tokens", "image_processor"])
        estimator = MediaTokenEstimator(proxy)
        real = _real_image_token_count(qwen2_vl_processor, height, width)
        assert estimator.estimate_media_tokens(images_meta=[[height, width]]) == real

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_qwen_geometry_tier_matches_real_expansion(self, qwen2_vl_processor, height, width):
        """Tier 3 (anti-drift): the local smart_resize replica must not drift from HF."""
        proxy = _HidingProxy(qwen2_vl_processor, ["_get_num_multimodal_tokens", "tokenizer"])
        estimator = MediaTokenEstimator(proxy)
        real = _real_image_token_count(qwen2_vl_processor, height, width)
        assert estimator.estimate_media_tokens(images_meta=[[height, width]]) == real


# ═══════════════════════════════════════════════════════════════════════════
# Video parity (geometry math vs the real Qwen3-VL video processor)
# ═══════════════════════════════════════════════════════════════════════════


class TestVideoParityWithRealProcessor:
    """Video estimation == the count the processor expands the video to.

    The video placeholder is expanded to ``video_grid_thw.prod() //
    merge_size**2`` tokens, so the parity target is derived from the real
    video processor's grid output.  ``do_sample_frames=False`` keeps every
    input frame, and the metadata passed to the estimator (source fps equal to
    the processor's target fps) makes its frame-sampling heuristic keep every
    frame too, so the comparison isolates the resize + patch/merge math.
    """

    @pytest.mark.parametrize(("num_frames", "height", "width"), VIDEO_SIZES)
    def test_geometry_matches_real_video_processor(self, num_frames, height, width):
        video_processor = Qwen3VLVideoProcessor()
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        output = video_processor(videos=[frames], do_sample_frames=False)
        grid_t, grid_h, grid_w = output["video_grid_thw"].tolist()[0]
        real = (grid_t * grid_h * grid_w) // video_processor.merge_size**2

        estimator = MediaTokenEstimator(SimpleNamespace(video_processor=video_processor))
        source_fps = float(video_processor.fps)
        estimated = estimator.estimate_media_tokens(
            videos_meta=[[num_frames, height, width, source_fps, num_frames / source_fps]]
        )
        assert estimated == real


# ═══════════════════════════════════════════════════════════════════════════
# Geometry extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestGeometryExtraction:
    """Extraction must read the same values the processor actually uses."""

    def test_image_direct_attrs_take_precedence(self):
        """YAML overrides land as direct attributes and must win over stale size dicts."""
        ip = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            min_pixels=262144,
            max_pixels=4194304,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        geometry = _extract_image_geometry(SimpleNamespace(image_processor=ip))
        assert geometry.min_pixels == 262144
        assert geometry.max_pixels == 4194304
        assert geometry.patch_size == 16
        assert geometry.merge_size == 2
        assert geometry.factor == 32

    def test_image_qwen_style_size_keys(self):
        ip = SimpleNamespace(patch_size=16, merge_size=2, size={"min_pixels": 262144, "max_pixels": 4194304})
        geometry = _extract_image_geometry(SimpleNamespace(image_processor=ip))
        assert geometry.min_pixels == 262144
        assert geometry.max_pixels == 4194304

    def test_image_hf_style_size_keys(self):
        ip = SimpleNamespace(patch_size=14, merge_size=2, size={"shortest_edge": 3136, "longest_edge": 1003520})
        geometry = _extract_image_geometry(SimpleNamespace(image_processor=ip))
        assert geometry.min_pixels == 3136
        assert geometry.max_pixels == 1003520

    def test_image_size_dict_of_real_processor(self):
        """transformers 5.x: direct attrs are None, values live in a SizeDict."""
        image_processor = Qwen2VLImageProcessor()
        geometry = _extract_image_geometry(SimpleNamespace(image_processor=image_processor))
        assert geometry.min_pixels == image_processor.size.get("shortest_edge")
        assert geometry.max_pixels == image_processor.size.get("longest_edge")
        assert geometry.min_pixels is not None
        assert geometry.max_pixels is not None

    def test_video_direct_attrs_take_precedence(self):
        vp = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=131072,
            max_pixels=8388608,
            size={"shortest_edge": 16384, "longest_edge": 100663296},
            fps=2.0,
            min_frames=4,
            max_frames=768,
        )
        geometry = _extract_video_geometry(SimpleNamespace(video_processor=vp))
        assert geometry.min_pixels == 131072
        assert geometry.max_pixels == 8388608
        assert geometry.fps == 2.0
        assert geometry.max_frames == 768

    def test_video_none_valued_attrs_fall_back_to_defaults(self):
        """transformers 5.x Qwen2VLVideoProcessor has fps=None; math must not crash."""
        vp = SimpleNamespace(
            patch_size=None,
            merge_size=None,
            temporal_patch_size=None,
            size=None,
            fps=None,
            min_frames=None,
            max_frames=None,
        )
        geometry = _extract_video_geometry(SimpleNamespace(video_processor=vp))
        assert geometry.fps == 2.0
        assert geometry.patch_size == 16
        assert geometry.merge_size == 2
        assert geometry.temporal_patch_size == 2

    def test_missing_processors_return_none(self):
        assert _extract_image_geometry(None) is None
        assert _extract_video_geometry(None) is None
        assert _extract_image_geometry(SimpleNamespace(image_processor=None)) is None
        assert _extract_video_geometry(SimpleNamespace(video_processor=None)) is None


# ═══════════════════════════════════════════════════════════════════════════
# Estimator fallback and caching behavior
# ═══════════════════════════════════════════════════════════════════════════


class _FakeTokenizer:
    """Tokenizes any text to a single token id."""

    def __call__(self, text):
        return {"input_ids": [3]}

    def convert_tokens_to_ids(self, token):
        return 3


class _CountingProbeProcessor:
    """Probe-only processor: expands the image placeholder to 12 tokens."""

    image_token = "<image>"

    def __init__(self):
        self.calls = 0
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, images):
        self.calls += 1
        return {"input_ids": [[7] * 12]}


class TestEstimatorBehavior:
    def test_without_processor_uses_flat_default(self):
        estimator = MediaTokenEstimator(None)
        assert not estimator.can_estimate
        total = estimator.estimate_media_tokens(
            images_meta=[[100, 100]],
            videos_meta=[[8, 100, 100, 2.0, 4.0]],
        )
        assert total == 2 * DEFAULT_TOKENS_PER_MEDIA_ITEM

    def test_none_meta_entries_are_skipped(self):
        estimator = MediaTokenEstimator(None)
        assert estimator.estimate_media_tokens(images_meta=[None], videos_meta=[None]) == 0

    def test_probe_runs_processor_once_per_unique_size(self):
        processor = _CountingProbeProcessor()
        estimator = MediaTokenEstimator(processor)
        # expanded (12) - base (1) + 1 placeholder = 12 tokens
        assert estimator.estimate_media_tokens(images_meta=[[64, 48]]) == 12
        assert estimator.estimate_media_tokens(images_meta=[[64, 48]]) == 12
        assert processor.calls == 1
        estimator.estimate_media_tokens(images_meta=[[96, 96]])
        assert processor.calls == 2

    def test_processor_api_failure_falls_back_to_geometry(self):
        class _BrokenApiProcessor:
            image_processor = SimpleNamespace(
                patch_size=14, merge_size=2, min_pixels=3136, max_pixels=10_000_000, size={}
            )

            def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
                raise RuntimeError("boom")

        estimator = MediaTokenEstimator(_BrokenApiProcessor())
        # 280x560, factor 28: no resize -> (280/14) * (560/14) / 4 = 200
        assert estimator.estimate_media_tokens(images_meta=[[280, 560]]) == 200


# ═══════════════════════════════════════════════════════════════════════════
# Both planning call sites resolve the same counts
# ═══════════════════════════════════════════════════════════════════════════


class TestCallSitesAgree:
    """The packer and the sampler must plan from identical media counts.

    They used to carry separate copies of the geometry math; a divergence
    means the sampler filters on one length while the packer bins on another.
    """

    def test_sampler_and_packer_agree_with_real_processor(self, qwen2_vl_processor):
        example = {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a" * 30}, {"type": "image", "image": "dummy.png"}],
                }
            ],
            "mm_inputs_meta": {"images_meta": [[1024, 768]]},
        }
        sampler = LengthGroupedSampler([example], seed=0, processor=qwen2_vl_processor)
        _, packer_media_tokens = _estimate_sample_length(example, MediaTokenEstimator(qwen2_vl_processor))

        real = _real_image_token_count(qwen2_vl_processor, 1024, 768)
        assert sampler.media_lengths[0] == real
        assert packer_media_tokens == real


# ═══════════════════════════════════════════════════════════════════════════
# smart_resize invariants
# ═══════════════════════════════════════════════════════════════════════════


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
