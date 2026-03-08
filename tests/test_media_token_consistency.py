#!/usr/bin/env python3
"""Test that sampler media-token estimation is consistent with the actual
HuggingFace processor output.

Covers three levels:
  1. Config extraction — sampler reads the same min/max_pixels as the processor uses.
  2. smart_resize parity — our local _smart_resize_image matches the HF implementation.
  3. End-to-end token count — estimated image tokens == actual image tokens in input_ids.

Run:
    # Pure-logic tests (no GPU / model download needed):
    python -m pytest tests/test_media_token_consistency.py -v -k "not real_processor"

    # Full end-to-end (needs a local Qwen3-VL checkpoint):
    python -m pytest tests/test_media_token_consistency.py -v \
        --processor-path /root/zhiqil/checkpoints/Qwen3-VL-8B-Instruct
"""

import importlib.util
import math
import os
import sys
from types import SimpleNamespace

import pytest

# Import directly from source files to avoid __init__.py pulling in heavy
# dependencies (datasets, transformers) that may not be installed.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_samplers = _import_file(
    "samplers", os.path.join(_REPO, "nemo_automodel/components/datasets/vlm/samplers.py"),
)
LengthGroupedSampler = _samplers.LengthGroupedSampler
_smart_resize_image = _samplers._smart_resize_image
_smart_resize_video = _samplers._smart_resize_video

try:
    _collate_fns = _import_file(
        "collate_fns", os.path.join(_REPO, "nemo_automodel/components/datasets/vlm/collate_fns.py"),
    )
    collate_extract_image_config = _collate_fns._extract_image_config
except ImportError:
    collate_extract_image_config = None


# ── pytest custom option ─────────────────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption(
        "--processor-path", default=None,
        help="Path to a HF processor (e.g. Qwen3-VL-8B-Instruct) for e2e tests",
    )


@pytest.fixture
def processor_path(request):
    return request.config.getoption("--processor-path")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Config Extraction Tests
#    Verify _extract_image_config / _extract_video_config reads the correct
#    values under all representations a HF processor might use.
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigExtraction:
    """Config extraction must match processor's actual attributes, regardless
    of how they are stored (direct attrs, Qwen-style size keys, HF-style size keys)."""

    # ── Image config ──────────────────────────────────────────────────────

    def test_image_direct_attrs_take_precedence(self):
        """When ip.min_pixels / ip.max_pixels exist as direct attributes,
        they must be used — even if ip.size has different values."""
        ip = SimpleNamespace(
            patch_size=16, merge_size=2,
            min_pixels=262144, max_pixels=4194304,
            # size has STALE values that should be ignored
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=None)
        cfg = LengthGroupedSampler._extract_image_config(proc)

        assert cfg["min_pixels"] == 262144,  f"Expected 262144, got {cfg['min_pixels']}"
        assert cfg["max_pixels"] == 4194304, f"Expected 4194304, got {cfg['max_pixels']}"
        assert cfg["patch_size"] == 16
        assert cfg["merge_size"] == 2
        assert cfg["factor"] == 32  # 16 * 2

    def test_image_qwen_style_size_keys(self):
        """If direct attrs are missing, read ip.size["min_pixels"] / ip.size["max_pixels"]."""
        ip = SimpleNamespace(
            patch_size=16, merge_size=2,
            size={"min_pixels": 262144, "max_pixels": 4194304},
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=None)
        cfg = LengthGroupedSampler._extract_image_config(proc)

        assert cfg["min_pixels"] == 262144
        assert cfg["max_pixels"] == 4194304

    def test_image_hf_style_size_keys(self):
        """If direct attrs and Qwen keys are missing, read shortest_edge / longest_edge."""
        ip = SimpleNamespace(
            patch_size=14, merge_size=2,
            size={"shortest_edge": 3136, "longest_edge": 1003520},
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=None)
        cfg = LengthGroupedSampler._extract_image_config(proc)

        assert cfg["min_pixels"] == 3136
        assert cfg["max_pixels"] == 1003520

    def test_image_falls_back_to_defaults(self):
        """If nothing is set, use hardcoded defaults."""
        ip = SimpleNamespace(patch_size=14, merge_size=2, size={})
        proc = SimpleNamespace(image_processor=ip, video_processor=None)
        cfg = LengthGroupedSampler._extract_image_config(proc)

        assert cfg["min_pixels"] == 56 * 56
        assert cfg["max_pixels"] == 14 * 14 * 4 * 1280

    @pytest.mark.skipif(collate_extract_image_config is None,
                        reason="transformers not installed (collate_fns cannot be imported)")
    def test_image_collate_matches_sampler(self):
        """collate_fns._extract_image_config must return identical values."""
        ip = SimpleNamespace(
            patch_size=16, merge_size=2,
            min_pixels=262144, max_pixels=4194304,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=None)

        sampler_cfg = LengthGroupedSampler._extract_image_config(proc)
        collate_cfg = collate_extract_image_config(proc)

        assert sampler_cfg == collate_cfg, (
            f"Sampler and collate configs differ!\n"
            f"  sampler: {sampler_cfg}\n"
            f"  collate: {collate_cfg}"
        )

    # ── Video config ──────────────────────────────────────────────────────

    def test_video_direct_attrs_take_precedence(self):
        """When vp.min_pixels / vp.max_pixels exist, they must be used."""
        vp = SimpleNamespace(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            min_pixels=131072, max_pixels=8388608,
            size={"shortest_edge": 16384, "longest_edge": 100663296},
            fps=2.0, min_frames=4, max_frames=768,
        )
        proc = SimpleNamespace(image_processor=None, video_processor=vp)
        cfg = LengthGroupedSampler._extract_video_config(proc)

        assert cfg["min_pixels"] == 131072,  f"Expected 131072, got {cfg['min_pixels']}"
        assert cfg["max_pixels"] == 8388608, f"Expected 8388608, got {cfg['max_pixels']}"
        assert cfg["fps"] == 2.0
        assert cfg["max_frames"] == 768

    def test_video_qwen_style_size_keys(self):
        ip_size = {"min_pixels": 131072, "max_pixels": 8388608}
        vp = SimpleNamespace(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            size=ip_size, fps=2.0, min_frames=4, max_frames=768,
        )
        proc = SimpleNamespace(image_processor=None, video_processor=vp)
        cfg = LengthGroupedSampler._extract_video_config(proc)

        assert cfg["min_pixels"] == 131072
        assert cfg["max_pixels"] == 8388608

    # ── Qwen3-VL-8B-Instruct exact YAML config ────────────────────────────

    def test_qwen3_vl_8b_yaml_config(self):
        """Simulate the exact processor state after loading with YAML overrides:
            min_pixels: 262144
            max_pixels: 4194304
            video_min_pixels: 131072
            video_max_pixels: 8388608
            max_frames: 16
        The preprocessor_config.json has different size dict values."""
        ip = SimpleNamespace(
            patch_size=16, merge_size=2,
            # Direct attrs set by YAML overrides
            min_pixels=262144, max_pixels=4194304,
            # size dict from preprocessor_config.json (NOT updated by YAML)
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
        vp = SimpleNamespace(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            # Direct attrs set by YAML overrides
            min_pixels=131072, max_pixels=8388608,
            # size dict from preprocessor_config.json (NOT updated by YAML)
            size={"shortest_edge": 65536, "longest_edge": 16777216},
            fps=2.0, min_frames=4, max_frames=16,
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=vp)

        img_cfg = LengthGroupedSampler._extract_image_config(proc)
        vid_cfg = LengthGroupedSampler._extract_video_config(proc)

        # Image: must use YAML values, NOT preprocessor_config.json defaults
        assert img_cfg["min_pixels"] == 262144, (
            f"Image min_pixels wrong: got {img_cfg['min_pixels']}, "
            f"expected 262144 (YAML), not 65536 (preprocessor_config.json)"
        )
        assert img_cfg["max_pixels"] == 4194304, (
            f"Image max_pixels wrong: got {img_cfg['max_pixels']}, "
            f"expected 4194304 (YAML), not 16777216 (preprocessor_config.json)"
        )

        # Video: must use YAML values
        assert vid_cfg["min_pixels"] == 131072
        assert vid_cfg["max_pixels"] == 8388608
        assert vid_cfg["max_frames"] == 16


# ═══════════════════════════════════════════════════════════════════════════
# 2. smart_resize Parity Tests
#    Our _smart_resize_image must match HF's smart_resize for the same params.
# ═══════════════════════════════════════════════════════════════════════════


class TestSmartResizeParity:
    """Verify _smart_resize_image matches the reference implementation."""

    @staticmethod
    def _reference_smart_resize(height, width, factor, min_pixels, max_pixels):
        """Reference implementation from HF transformers
        (qwen2_vl.image_processing_qwen2_vl.smart_resize)."""
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    # Test with Qwen3-VL-8B YAML config: patch_size=16, merge_size=2, factor=32
    FACTOR = 32
    MIN_PIX = 262144
    MAX_PIX = 4194304

    @pytest.mark.parametrize("height,width", [
        (100, 100),      # tiny → upscale to min_pixels
        (256, 256),      # small
        (512, 512),      # medium
        (1024, 768),     # typical photo
        (1920, 1080),    # FHD
        (3840, 2160),    # 4K → downscale to max_pixels
        (4096, 4096),    # large square
        (8000, 6000),    # very large → heavy downscale
        (32, 32),        # edge: very tiny
        (1, 10000),      # edge: extreme aspect ratio
        (10000, 1),      # edge: extreme aspect ratio (other direction)
        (2048, 2048),    # exactly at max_pixels boundary
    ])
    def test_matches_reference(self, height, width):
        ours = _smart_resize_image(
            height, width,
            factor=self.FACTOR,
            min_pixels=self.MIN_PIX,
            max_pixels=self.MAX_PIX,
        )
        ref = self._reference_smart_resize(
            height, width,
            factor=self.FACTOR,
            min_pixels=self.MIN_PIX,
            max_pixels=self.MAX_PIX,
        )
        assert ours == ref, f"Mismatch for ({height}, {width}): ours={ours}, ref={ref}"

    @pytest.mark.parametrize("height,width", [
        (100, 100),
        (1024, 768),
        (3840, 2160),
    ])
    def test_output_respects_constraints(self, height, width):
        h, w = _smart_resize_image(
            height, width,
            factor=self.FACTOR,
            min_pixels=self.MIN_PIX,
            max_pixels=self.MAX_PIX,
        )
        assert h % self.FACTOR == 0, f"h={h} not aligned to factor={self.FACTOR}"
        assert w % self.FACTOR == 0, f"w={w} not aligned to factor={self.FACTOR}"
        assert h * w <= self.MAX_PIX, f"h*w={h*w} exceeds max_pixels={self.MAX_PIX}"
        # min_pixels is a soft guarantee (small images may not reach it exactly)


# ═══════════════════════════════════════════════════════════════════════════
# 3. End-to-End Token Count Tests
#    Estimated image tokens == actual image tokens from the HF processor.
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenCountEndToEnd:
    """Compare sampler's estimated token count against actual processor output.

    Token count from image_grid_thw: for each image grid [t, h, w],
    actual_tokens = t * (h // merge_size) * (w // merge_size).
    """

    @staticmethod
    def _compute_actual_image_tokens(grid_thw, merge_size=2):
        """Compute actual vision tokens from image_grid_thw tensor."""
        total = 0
        for row in grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            total += t * (h // merge_size) * (w // merge_size)
        return total

    @staticmethod
    def _estimate_image_tokens(height, width, cfg):
        """Replicate the sampler's estimation logic."""
        resized_h, resized_w = _smart_resize_image(
            height, width,
            factor=cfg["factor"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
        )
        merge_length = cfg["merge_size"] ** 2
        return (resized_h // cfg["patch_size"]) * (resized_w // cfg["patch_size"]) // merge_length

    @pytest.mark.parametrize("height,width", [
        (256, 256),
        (512, 512),
        (1024, 768),
        (1920, 1080),
        (3840, 2160),
        (100, 100),
        (4096, 4096),
    ])
    def test_with_real_processor(self, processor_path, height, width):
        """Process a synthetic image through the real HF processor and
        verify estimated tokens match actual tokens."""
        if processor_path is None:
            pytest.skip("No --processor-path provided")

        from PIL import Image
        from transformers import AutoProcessor

        # Load processor with YAML overrides
        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=262144,
            max_pixels=4194304,
        )

        # Extract config the same way the sampler does
        img_cfg = LengthGroupedSampler._extract_image_config(processor)

        # Verify config matches processor's actual attributes
        ip = processor.image_processor
        assert img_cfg["min_pixels"] == ip.min_pixels, (
            f"Config mismatch: extracted min_pixels={img_cfg['min_pixels']}, "
            f"processor.min_pixels={ip.min_pixels}"
        )
        assert img_cfg["max_pixels"] == ip.max_pixels, (
            f"Config mismatch: extracted max_pixels={img_cfg['max_pixels']}, "
            f"processor.max_pixels={ip.max_pixels}"
        )

        # Estimate tokens
        estimated = self._estimate_image_tokens(height, width, img_cfg)

        # Create synthetic image and process
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        text = "<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n"
        batch = processor(images=[img], text=[text], return_tensors="pt")

        # Actual tokens from grid_thw
        grid_thw = batch.get("image_grid_thw")
        assert grid_thw is not None, "Processor did not return image_grid_thw"
        actual = self._compute_actual_image_tokens(grid_thw, merge_size=img_cfg["merge_size"])

        assert estimated == actual, (
            f"Token count mismatch for image ({height}x{width}):\n"
            f"  estimated (sampler) = {estimated}\n"
            f"  actual (processor)  = {actual}\n"
            f"  config: min_pixels={img_cfg['min_pixels']}, "
            f"max_pixels={img_cfg['max_pixels']}, "
            f"patch_size={img_cfg['patch_size']}, "
            f"merge_size={img_cfg['merge_size']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Regression: old bug would silently use wrong defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionWrongDefaults:
    """Before the fix, _extract_image_config always fell back to hardcoded
    defaults (min_pixels=3136, max_pixels=1003520) for Qwen processors
    because it looked for "shortest_edge"/"longest_edge" keys in ip.size,
    but Qwen stores them as "min_pixels"/"max_pixels" or as direct attrs."""

    def test_not_using_hardcoded_defaults_when_attrs_available(self):
        """If the processor has min_pixels=262144, we must NOT get 3136."""
        ip = SimpleNamespace(
            patch_size=16, merge_size=2,
            min_pixels=262144, max_pixels=4194304,
            size={"min_pixels": 262144, "max_pixels": 4194304},
        )
        proc = SimpleNamespace(image_processor=ip, video_processor=None)
        cfg = LengthGroupedSampler._extract_image_config(proc)

        # These are the OLD wrong defaults — must NOT match
        assert cfg["min_pixels"] != 56 * 56, "Still using old hardcoded default 3136!"
        assert cfg["max_pixels"] != 14 * 14 * 4 * 1280, "Still using old hardcoded default 1003520!"

        # Must match actual processor values
        assert cfg["min_pixels"] == 262144
        assert cfg["max_pixels"] == 4194304

    def test_token_estimate_changes_with_correct_config(self):
        """Show the impact: a 4096x4096 image gives very different token
        counts with wrong vs correct max_pixels."""
        height, width = 4096, 4096
        factor = 32  # patch_size=16 * merge_size=2

        # Wrong defaults (old bug): max_pixels=1003520
        h_wrong, w_wrong = _smart_resize_image(height, width, factor=factor,
                                                min_pixels=3136, max_pixels=1003520)
        tokens_wrong = (h_wrong // 16) * (w_wrong // 16) // 4

        # Correct config: max_pixels=4194304
        h_correct, w_correct = _smart_resize_image(height, width, factor=factor,
                                                     min_pixels=262144, max_pixels=4194304)
        tokens_correct = (h_correct // 16) * (w_correct // 16) // 4

        ratio = tokens_correct / tokens_wrong
        print("\n4096x4096 image token impact:")
        print(f"  Wrong config (old bug):  {h_wrong}x{w_wrong} → {tokens_wrong} tokens")
        print(f"  Correct config (fixed):  {h_correct}x{w_correct} → {tokens_correct} tokens")
        print(f"  Ratio: {ratio:.1f}x")

        # With 4x larger max_pixels, large images get ~4x more tokens
        assert ratio > 2.0, f"Expected significant difference, got only {ratio:.1f}x"


if __name__ == "__main__":
    # Allow running as a script for quick manual validation
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
