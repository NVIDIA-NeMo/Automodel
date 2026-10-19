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

"""Unit tests for the MiMo-V2 sglang-aligned image processor.

Reference-geometry pairs and pixel expectations in this file were produced by
the official serving stack's ``MiMoProcessor`` (sglang, the preprocessing the
MiMo-V2.6 RL checkpoints were rolled out with); see the module docstring of
``nemo_automodel.components.models.mimo_v2_flash.processing``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_automodel.components.models.mimo_v2_flash.processing import (
    MimoV2ImageProcessor,
    build_mimo_v2_processor,
    smart_resize,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Minimal Qwen2.5-VL-style chat template: one <|image_pad|> placeholder per
# image content block, <|im_start|>/<|im_end|> turn markers.
_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message.role + '\n' }}"
    "{% for item in message.content %}{% if item.type == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% elif item.type == 'text' %}"
    "{{ item.text }}{% endif %}{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
)


def _make_qwen25_processor(**ip_kwargs):
    """Build a real ``Qwen2_5_VLProcessor`` around ``MimoV2ImageProcessor``, fully offline.

    The tokenizer is a tiny ``WordLevel`` vocab with the Qwen special tokens
    isolated by the pre-tokenizer; it is only used for structural assertions
    (placeholder expansion, label masking), never for text content.
    """
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2VLVideoProcessor
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    vocab = {
        "<unk>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|image_pad|>": 3,
        "<|vision_start|>": 4,
        "<|vision_end|>": 5,
        "<|video_pad|>": 6,
        "<|endoftext|>": 7,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"<\|[^|]+\|>"), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<|endoftext|>")
    return Qwen2_5_VLProcessor(
        image_processor=MimoV2ImageProcessor(**ip_kwargs),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
        chat_template=_CHAT_TEMPLATE,
    )


def _conversations(*images):
    return [
        [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
        for img in images
    ]


def _oracle_standardized(array: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 standardized with 0-255-space stats."""
    mean255 = np.array(MEAN, dtype=np.float32) * np.float32(255.0)
    std255 = np.array(STD, dtype=np.float32) * np.float32(255.0)
    return (array.astype(np.float32).transpose(2, 0, 1) - mean255[:, None, None]) / std255[:, None, None]


def _oracle_patches(standardized: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """(3, H, W) standardized -> [grid_h*grid_w, 3*2*16*16] in merge-block order.

    Patch index walks 2x2 merge blocks row-major over the grid; inside a block
    the 2x2 sub-patches are row-major; each patch vector is (channel, temporal,
    16, 16) flattened with the temporal frames duplicated.
    """
    n_blocks = (grid_h // 2) * (grid_w // 2)
    out = np.empty((n_blocks * 4, 3, 2, 16, 16), dtype=np.float32)
    for i in range(n_blocks * 4):
        block, within = divmod(i, 4)
        by, bx = divmod(block, grid_w // 2)
        my, mx = divmod(within, 2)
        ys = slice((by * 2 + my) * 16, (by * 2 + my + 1) * 16)
        xs = slice((bx * 2 + mx) * 16, (bx * 2 + mx + 1) * 16)
        out[i] = standardized[:, ys, xs][:, None, :, :]
    return out.reshape(n_blocks * 4, -1)


def _coord_image(height: int, width: int) -> np.ndarray:
    """uint8 (H, W, 3) image whose channels encode (y, x, y+x) coordinates."""
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack([yy % 256, xx % 256, (yy + xx) % 256], axis=-1).astype(np.uint8)


# (height, width) -> (h_bar, w_bar) from the reference sglang MiMoProcessor with
# the released MiMo-V2.6 bounds (factor=32, min_pixels=8192, max_pixels=8388608),
# covering the upscale, shrink, round-to-factor and short-side-below-factor branches.
REFERENCE_GEOMETRY = [
    ((20, 44), (64, 160)),
    ((64, 64), (96, 96)),
    ((300, 300), (288, 288)),
    ((600, 800), (608, 800)),
    ((389, 517), (384, 512)),
    ((200, 320), (192, 320)),
    ((2160, 3840), (2176, 3840)),
    ((3000, 4000), (2496, 3328)),
    ((4000, 48), (4000, 64)),
]


@pytest.mark.parametrize(("hw", "expected"), REFERENCE_GEOMETRY)
def test_smart_resize_matches_reference_geometry(hw, expected):
    h_bar, w_bar = smart_resize(hw[0], hw[1], factor=32, min_pixels=8192, max_pixels=8388608)
    assert (h_bar, w_bar) == expected


def test_smart_resize_matches_hf_on_supported_inputs():
    """HF and sglang agree on resize geometry for every shape in the reference table."""
    hf_smart_resize = pytest.importorskip(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl", reason="transformers Qwen2VL module layout changed"
    ).smart_resize
    for (h, w), expected in REFERENCE_GEOMETRY:
        assert smart_resize(h, w, factor=32, min_pixels=8192, max_pixels=8388608) == expected
        assert hf_smart_resize(h, w, factor=32, min_pixels=8192, max_pixels=8388608) == expected


def test_smart_resize_diverges_from_hf_below_factor():
    """The only deliberate geometry divergence: a short side below ``factor``.

    The reference upscales it to ``factor`` preserving aspect ratio; HF clamps
    each side independently and then applies the min-pixel bound, so tiny
    inputs can produce different grids (16x192: 32x384 vs 32x320).
    """
    hf_smart_resize = pytest.importorskip(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl", reason="transformers Qwen2VL module layout changed"
    ).smart_resize
    assert smart_resize(16, 192, factor=32, min_pixels=8192, max_pixels=8388608) == (32, 384)
    assert hf_smart_resize(16, 192, factor=32, min_pixels=8192, max_pixels=8388608) == (32, 320)


def test_smart_resize_rejects_degenerate_shrink():
    """HF clamps a floored-to-zero side to ``factor``; sglang semantics reject it."""
    with pytest.raises(ValueError, match="smaller than factor"):
        smart_resize(64, 6400, factor=32, min_pixels=64, max_pixels=2048)


def test_identity_resize_exact_pixels():
    """No resampling needed: output must equal hand-computed normalization, patch for patch.

    Pins the normalization scale space (0-255, not 0-1), the channel/temporal
    layout of each patch vector, the merge-block patch order, and the
    token-count invariant ``Qwen2_5_VLProcessor`` relies on.
    """
    processor = MimoV2ImageProcessor(min_pixels=64, max_pixels=16384)
    array = _coord_image(96, 64)
    out = processor(images=[Image.fromarray(array)])

    assert out["image_grid_thw"].tolist() == [[1, 6, 4]]
    assert out["image_grid_thw"].dtype == torch.long
    expected = _oracle_patches(_oracle_standardized(array), grid_h=6, grid_w=4)
    assert out["pixel_values"].shape == expected.shape
    torch.testing.assert_close(out["pixel_values"], torch.from_numpy(expected), rtol=0, atol=1e-6)

    # Invariant behind Qwen2_5_VLProcessor's <|image_pad|> expansion.
    n_patches = processor.get_number_of_image_patches(96, 64)
    assert n_patches == 6 * 4 == out["pixel_values"].shape[0]
    assert out["image_grid_thw"][0].prod().item() == n_patches


def test_downscale_2x_is_exact_block_average():
    """A 2:1 shrink must be the mean of each 2x2 block (bilinear, no antialias).

    Fails for the HF path (bicubic + antialias on uint8), which deviates from
    the exact block average by several pixel values.
    """
    rng = np.random.RandomState(0)
    array = rng.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    processor = MimoV2ImageProcessor(min_pixels=64, max_pixels=4096)
    out = processor(images=[Image.fromarray(array)])

    assert out["image_grid_thw"].tolist() == [[1, 4, 4]]
    source = _oracle_standardized(array)
    expected = source.reshape(3, 64, 2, 64, 2).mean(axis=(2, 4))
    expected = _oracle_patches(expected, grid_h=4, grid_w=4)
    torch.testing.assert_close(out["pixel_values"], torch.from_numpy(expected), rtol=0, atol=1e-5)


def test_rgba_flattened_on_white():
    """Fully transparent RGBA must normalize as white, not as its raw RGB channels."""
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 3] = 200, 0  # invisible red
    processor = MimoV2ImageProcessor(min_pixels=64, max_pixels=16384)
    out = processor(images=[Image.fromarray(rgba, mode="RGBA")])

    assert out["image_grid_thw"].tolist() == [[1, 2, 2]]
    white = _oracle_standardized(np.full((32, 32, 3), 255, dtype=np.uint8))
    expected = _oracle_patches(white, grid_h=2, grid_w=2)
    torch.testing.assert_close(out["pixel_values"], torch.from_numpy(expected), rtol=0, atol=1e-6)


def test_rejects_unknown_image_type():
    processor = MimoV2ImageProcessor()
    with pytest.raises(TypeError, match="expected a PIL image or a local path"):
        processor(images=[123])


def test_fetch_images_normalizes_inputs(tmp_path):
    """ProcessorMixin (transformers >= 5.18) fetches images before __call__."""
    processor = MimoV2ImageProcessor()
    image = Image.fromarray(_coord_image(32, 32))
    path = tmp_path / "img.png"
    image.save(path)

    assert processor.fetch_images(image) is image
    nested = processor.fetch_images([[str(path)], image])
    assert len(nested) == 2
    assert isinstance(nested[0][0], Image.Image) and nested[1] is image
    with pytest.raises(TypeError, match="expected a PIL image or a local path"):
        processor.fetch_images(123)


def test_per_call_pixel_bound_override():
    """Per-call bounds shrink the image without mutating the processor state."""
    processor = MimoV2ImageProcessor(min_pixels=64, max_pixels=16384)
    array = _coord_image(128, 128)
    out = processor(images=[Image.fromarray(array)], max_pixels=4096)
    assert out["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert processor.max_pixels == 16384
    assert processor.get_number_of_image_patches(128, 128, {"max_pixels": 4096}) == 4 * 4
    assert processor.get_number_of_image_patches(128, 128) == 8 * 8


def test_build_mimo_v2_processor_swaps_image_processor(monkeypatch):
    """The factory keeps the stock processor, forwards extra kwargs verbatim,
    and replaces only the image path."""
    import nemo_automodel.components.models.mimo_v2_flash.processing as module

    class _StubHFImageProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2
        size = {"shortest_edge": 8192, "longest_edge": 8388608}
        image_mean = list(MEAN)
        image_std = list(STD)

    captured = {}

    class _StubProcessor:
        image_processor = _StubHFImageProcessor()

    def _fake_from_pretrained(path, **kwargs):
        captured.update(path=path, **kwargs)
        return _StubProcessor()

    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", staticmethod(_fake_from_pretrained))
    processor = module.build_mimo_v2_processor("dummy/repo", trust_remote_code=False, max_pixels=1048576)

    assert captured == {"path": "dummy/repo", "trust_remote_code": False, "max_pixels": 1048576}
    ip = processor.image_processor
    assert isinstance(ip, MimoV2ImageProcessor)
    assert (ip.min_pixels, ip.max_pixels) == (8192, 8388608)
    assert ip.image_mean == MEAN and ip.image_std == STD
    assert ip.size == {"shortest_edge": 8192, "longest_edge": 8388608}


def test_nested_image_lists_match_flat():
    """Per-sample nested input (as ``ProcessorMixin.apply_chat_template``
    produces for a batch of conversations) must flatten one level and match
    the flat call pixel for pixel."""
    processor = MimoV2ImageProcessor(min_pixels=64, max_pixels=16384)
    img1 = Image.fromarray(_coord_image(96, 64))
    img2 = Image.fromarray(_coord_image(128, 128))

    nested = processor(images=[[img1], [img2]])
    flat = processor(images=[img1, img2])
    assert torch.equal(nested["pixel_values"], flat["pixel_values"])
    assert torch.equal(nested["image_grid_thw"], flat["image_grid_thw"])


def test_apply_chat_template_end_to_end():
    """The real training entry: ``default_collate_fn`` reaches the image
    processor through ``ProcessorMixin.apply_chat_template(tokenize=True)``,
    which hands over a nested per-sample image list. The call must succeed and
    expand exactly ``prod(grid_thw) // merge_size**2`` placeholder tokens."""
    processor = _make_qwen25_processor(min_pixels=64, max_pixels=16384)
    img1 = Image.fromarray(_coord_image(96, 64))
    img2 = Image.fromarray(_coord_image(128, 128))

    batch = processor.apply_chat_template(
        _conversations(img1, img2),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )

    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_placeholders = int((batch["input_ids"] == image_pad_id).sum())
    assert n_placeholders == int(batch["image_grid_thw"].prod(dim=1).sum() // processor.image_processor.merge_size**2)
    assert batch["pixel_values"].shape[0] == int(batch["image_grid_thw"].prod(dim=1).sum())

    flat = processor.image_processor([img1, img2])
    assert torch.equal(batch["pixel_values"], flat["pixel_values"])
    assert torch.equal(batch["image_grid_thw"], flat["image_grid_thw"])


def test_runtime_max_pixels_routed_through_outer_processor():
    """``ProcessorMixin`` only forwards kwargs declared in the image
    processor's ``valid_kwargs``; a per-call ``max_pixels`` must reach
    ``MimoV2ImageProcessor`` instead of being dropped with a warning."""
    processor = _make_qwen25_processor(min_pixels=64, max_pixels=16384)
    image = Image.fromarray(_coord_image(128, 128))

    out = processor(text=["hello"], images=[image], max_pixels=4096, return_tensors="pt")
    assert out["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert processor.image_processor.max_pixels == 16384  # per-call override must not mutate state


def test_processor_save_and_reload_round_trip(tmp_path):
    """``ProcessorMixin.save_pretrained`` (the checkpoint-save path) must
    serialize the swapped-in image processor, and the saved directory must
    reload as a complete processor through both ``AutoProcessor`` and the
    recipe factory, pixel-exact."""
    from transformers import AutoProcessor

    processor = _make_qwen25_processor(min_pixels=64, max_pixels=4096)
    image = Image.fromarray(_coord_image(96, 64))

    processor.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "processor_config.json").read_text())
    assert saved["image_processor"]["image_processor_type"] == "MimoV2ImageProcessor"
    assert saved["image_processor"]["max_pixels"] == 4096

    processor.image_processor.save_pretrained(tmp_path / "image_processor")
    reloaded = MimoV2ImageProcessor.from_pretrained(tmp_path / "image_processor")
    assert (reloaded.min_pixels, reloaded.max_pixels) == (64, 4096)
    assert tuple(reloaded.image_mean) == MEAN and tuple(reloaded.image_std) == STD
    assert torch.equal(
        reloaded([image])["pixel_values"],
        processor.image_processor([image])["pixel_values"],
    )

    # Full reload: the saved processor_config.json names MimoV2ImageProcessor, so
    # it must resolve through the AutoImageProcessor registration (done on import).
    full = AutoProcessor.from_pretrained(tmp_path)
    assert isinstance(full.image_processor, MimoV2ImageProcessor)
    assert (full.image_processor.min_pixels, full.image_processor.max_pixels) == (64, 4096)
    assert torch.equal(
        full.image_processor([image])["pixel_values"],
        processor.image_processor([image])["pixel_values"],
    )

    rebuilt = build_mimo_v2_processor(str(tmp_path))
    assert isinstance(rebuilt.image_processor, MimoV2ImageProcessor)
    assert (rebuilt.image_processor.min_pixels, rebuilt.image_processor.max_pixels) == (64, 4096)


def test_saved_processor_reloads_in_fresh_process(tmp_path):
    """A saved processor directory must reload through ``AutoProcessor`` in a
    fresh interpreter that only imports the model package.

    The in-process round trip above cannot catch a missing
    ``AutoImageProcessor`` registration because this module already imported
    the image processor; the real consumer path is ``import
    nemo_automodel.components.models.mimo_v2_flash`` (or the NeMo auto model
    stack, which imports the package through its config) followed by
    ``AutoProcessor.from_pretrained``.
    """
    processor = _make_qwen25_processor(min_pixels=64, max_pixels=4096)
    processor.save_pretrained(tmp_path)

    array = np.random.RandomState(0).randint(0, 256, size=(96, 64, 3), dtype=np.uint8)
    checksum = float(processor.image_processor([Image.fromarray(array)])["pixel_values"].sum())

    script = (
        "import nemo_automodel.components.models.mimo_v2_flash\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "from transformers import AutoProcessor\n"
        f"p = AutoProcessor.from_pretrained({str(tmp_path)!r})\n"
        "assert type(p.image_processor).__name__ == 'MimoV2ImageProcessor'\n"
        "img = Image.fromarray(np.random.RandomState(0).randint(0, 256, size=(96, 64, 3), dtype=np.uint8))\n"
        f"assert float(p.image_processor([img])['pixel_values'].sum()) == {checksum!r}\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr


def test_default_collate_fn_preserve_alpha():
    """``default_collate_fn`` strips alpha via ``_ensure_rgb`` by default
    (matching stock HF processors); ``preserve_alpha=True`` keeps RGBA intact
    so the MiMo processor's white-flattening decides the final pixels."""
    from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

    processor = _make_qwen25_processor(min_pixels=64, max_pixels=16384)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 3] = 200, 0  # fully transparent red
    image = Image.fromarray(rgba, mode="RGBA")

    kept = default_collate_fn([{"conversation": c} for c in _conversations(image)], processor, preserve_alpha=True)
    dropped = default_collate_fn([{"conversation": c} for c in _conversations(image)], processor)

    # default_collate_fn casts pixel_values to bfloat16.
    white = processor.image_processor([Image.new("RGB", (64, 64), (255, 255, 255))])["pixel_values"].bfloat16()
    alpha_dropped = processor.image_processor([image.convert("RGB")])["pixel_values"].bfloat16()
    assert torch.equal(kept["pixel_values"], white)
    assert torch.equal(dropped["pixel_values"], alpha_dropped)
    assert not torch.equal(kept["pixel_values"], dropped["pixel_values"])
