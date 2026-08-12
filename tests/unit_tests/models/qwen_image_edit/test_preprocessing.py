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

"""Tests for the offline Qwen image-edit cache encoder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from nemo_automodel.components.datasets.diffusion.image_edit_dataset import ImageEditDataset
from nemo_automodel.components.models.qwen_image_edit.preprocessing import (
    QwenImageEditCacheEncoder,
    _resize_condition_image,
    _resize_vae_image,
    _validate_output_directory,
)


class _FakeLatentDistribution:
    def __init__(self, latent: torch.Tensor) -> None:
        """Store a deterministic latent tensor.

        Args:
            latent: Latent tensor of shape ``[batch, channels, frames, height,
                width]``.
        """
        self.latent = latent

    def mode(self) -> torch.Tensor:
        """Return the latent mode.

        Returns:
            Tensor of shape ``[batch, channels, frames, height, width]``.
        """
        return self.latent


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            z_dim=4,
            latents_mean=[0.0, 0.0, 0.0, 0.0],
            latents_std=[1.0, 1.0, 1.0, 1.0],
        )

    def encode(self, image: torch.Tensor) -> SimpleNamespace:
        """Encode a test image into a deterministic latent distribution.

        Args:
            image: Tensor of shape ``[batch, 3, 1, height, width]``.

        Returns:
            Namespace whose latent distribution contains a tensor of shape
            ``[batch, 4, 1, height / 8, width / 8]``.
        """
        downsampled = F.interpolate(image[:, :, 0].float(), scale_factor=1 / 8, mode="area")
        fourth_channel = downsampled.mean(dim=1, keepdim=True)
        latent = torch.cat([downsampled, fourth_channel], dim=1).unsqueeze(2) * self.scale
        return SimpleNamespace(latent_dist=_FakeLatentDistribution(latent))

    def decode(self, latent: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
        """Decode a test latent into a finite image tensor.

        Args:
            latent: Tensor of shape ``[batch, 4, 1, height, width]``.
            return_dict: Must be false to request the tuple output.

        Returns:
            One-element tuple containing a tensor of shape
            ``[batch, 3, 1, height * 8, width * 8]``.
        """
        assert not return_dict
        image = F.interpolate(latent[:, :3, 0].float(), scale_factor=8, mode="nearest").unsqueeze(2)
        return (image,)


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVAE()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.condition_sizes: list[list[tuple[int, int]]] = []

    def encode_prompt(
        self,
        *,
        prompt: list[str],
        image: list[Image.Image],
        device: torch.device,
        num_images_per_prompt: int,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B, S, D] prompt embeddings and [B, S] masks."""
        assert prompt == ["make the sky warmer"]
        assert num_images_per_prompt == 1
        assert max_sequence_length == 5
        self.condition_sizes.append([condition.size for condition in image])
        embeddings = torch.arange(8 * 6, dtype=torch.float32, device=device).reshape(1, 8, 6)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device)
        return embeddings, mask


def _write_manifest(manifest_path: Path) -> None:
    """Write one generic target/context/condition image-edit row."""
    row = {
        "id": "dev:00000000",
        "prompt": "make the sky warmer",
        "media": [
            {"role": "target", "file_name": "media/target.png"},
            {"role": "context", "file_name": "media/source.png"},
            {"role": "condition", "file_name": "media/source.png"},
        ],
        "metadata": {
            "dataset_name": "osunlp/MagicBrush",
            "dataset_config_name": "magicbrush",
            "dataset_split": "dev",
            "row_index": 0,
            "row": {"mask_img": None},
        },
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _make_materialized_manifest(output_dir: Path) -> Path:
    """Create a CLI-shaped materialization tree inside the cache root."""
    export_dir = output_dir / "_hf_dataset" / "image_edit"
    media_dir = export_dir / "media"
    media_dir.mkdir(parents=True)
    Image.new("RGB", (160, 80), color=(200, 100, 20)).save(media_dir / "target.png")
    Image.new("RGB", (80, 160), color=(20, 100, 200)).save(media_dir / "source.png")
    manifest_path = export_dir / "manifest.jsonl"
    _write_manifest(manifest_path)
    return manifest_path


def test_encoder_writes_dataset_compatible_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip fake VAE/text outputs through the generic cached dataset."""
    output_dir = tmp_path / "cache"
    manifest_path = _make_materialized_manifest(output_dir)
    pipeline = _FakePipeline()
    encoder = QwenImageEditCacheEncoder(
        max_sequence_length=5,
        device="cpu",
        torch_dtype="float32",
    )
    load_calls = []

    def fake_load_pipeline(device):
        load_calls.append(device)
        return pipeline

    monkeypatch.setattr(encoder, "_load_pipeline", fake_load_pipeline)

    metadata_path = encoder.encode_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        max_pixels=64 * 64,
        resolution_preset=None,
        num_gpus=1,
        verify=True,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_name"] == "osunlp/MagicBrush"
    assert metadata["dataset_config_name"] == "magicbrush"
    assert metadata["split"] == "dev"
    assert metadata["row_limit"] == 1
    assert metadata["preprocessing_config"] == {
        "condition_image_pixels": 384 * 384,
        "max_pixels": 64 * 64,
        "max_sequence_length": 5,
        "model_name": "Qwen/Qwen-Image-Edit-2511",
        "num_gpus": 1,
        "processor_target": (
            "nemo_automodel.components.models.qwen_image_edit.preprocessing.QwenImageEditCacheEncoder"
        ),
        "resize_mode": "aspect_preserving_max_pixels",
        "resolution_preset": None,
        "spatial_alignment": 32,
        "torch_dtype": "float32",
        "vae_latent_sampling": "mode",
        "verify": True,
    }
    assert load_calls == [torch.device("cpu")]

    dataset = ImageEditDataset(cache_dir=str(output_dir), quantization=32)
    sample = dataset[0]
    assert sample["target_latent"].shape == (4, 4, 8)
    assert [tuple(latent.shape) for latent in sample["context_latents"]] == [(4, 8, 4)]
    assert sample["prompt_embeddings"].shape == (5, 6)
    torch.testing.assert_close(sample["prompt_attention_mask"], torch.tensor([1, 1, 1, 1, 0]))
    assert sample["conditioning_tensors"] == {}
    assert sample["metadata"]["target_token_length"] == 8
    assert sample["metadata"]["context_token_lengths"] == [8]
    assert sample["metadata"]["text_token_length"] == 4
    assert sample["metadata"]["manifest_metadata"]["row"]["mask_img"] is None
    assert pipeline.condition_sizes == [[(256, 544)]]


def test_resize_modes_separate_aspect_buckets_from_fixed_square_benchmark() -> None:
    """Keep normal aspect ratio while making named presets exact square crops."""
    image = Image.new("RGB", (160, 80))

    multiresolution = _resize_vae_image(image, max_pixels=64 * 64, resolution_preset=None)
    benchmark = _resize_vae_image(image, max_pixels=256 * 256, resolution_preset="256p")

    assert multiresolution.size == (64, 32)
    assert multiresolution.width * multiresolution.height <= 64 * 64
    assert multiresolution.width / multiresolution.height == image.width / image.height
    assert benchmark.size == (256, 256)


def test_condition_resize_matches_upstream_qwen_edit_dimensions() -> None:
    """Keep Qwen2.5-VL image grids identical to the Diffusers pipeline."""
    qwen_pipeline = pytest.importorskip("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
    image = Image.new("RGB", (300, 200))

    resized = _resize_condition_image(image)
    expected = qwen_pipeline.calculate_dimensions(384 * 384, image.width / image.height)

    assert resized.size == expected == (480, 320)


def test_encoder_refuses_existing_cache_artifacts(tmp_path: Path) -> None:
    """Retain source materialization but never overwrite a prior cache."""
    output_dir = tmp_path / "cache"
    manifest_path = _make_materialized_manifest(output_dir)
    (output_dir / "metadata.json").write_text("{}", encoding="utf-8")
    encoder = QwenImageEditCacheEncoder(torch_dtype="float32")

    with pytest.raises(ValueError, match="existing cache artifacts"):
        encoder.encode_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            max_pixels=64 * 64,
            resolution_preset=None,
            num_gpus=1,
            verify=False,
        )


def test_output_validation_allows_configured_materialization_tree(tmp_path: Path) -> None:
    """A custom --dataset_dir nested under the cache root remains valid source input."""
    output_dir = tmp_path / "cache"
    manifest_path = output_dir / "custom_source" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("", encoding="utf-8")

    _validate_output_directory(output_dir, manifest_path=manifest_path)


def test_load_pipeline_raises_when_diffusers_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy diffusers import surfaces a clear error when the extra is missing."""
    from nemo_automodel.components.models.qwen_image_edit import preprocessing as preprocessing_module

    encoder = QwenImageEditCacheEncoder(model_name="stub/qwen-image-edit", torch_dtype="float32")
    monkeypatch.setattr(preprocessing_module, "safe_import", lambda name, msg=None: (False, None))

    with pytest.raises(ImportError, match="requires diffusers"):
        encoder._load_pipeline(torch.device("cpu"))


def test_load_pipeline_loads_frozen_conditioning_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_pipeline lazily imports diffusers and returns a frozen eval-mode VAE/text-encoder pair."""
    diffusers = pytest.importorskip("diffusers")

    fake_pipeline = SimpleNamespace(vae=torch.nn.Linear(2, 2), text_encoder=torch.nn.Linear(2, 2))
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_name, transformer=None, torch_dtype=None):
        captured.update(model_name=model_name, transformer=transformer, torch_dtype=torch_dtype)
        return fake_pipeline

    monkeypatch.setattr(diffusers.QwenImageEditPlusPipeline, "from_pretrained", fake_from_pretrained)

    encoder = QwenImageEditCacheEncoder(model_name="stub/qwen-image-edit", torch_dtype="float32")
    pipeline = encoder._load_pipeline(torch.device("cpu"))

    assert pipeline is fake_pipeline
    assert captured == {"model_name": "stub/qwen-image-edit", "transformer": None, "torch_dtype": torch.float32}
    for module in (pipeline.vae, pipeline.text_encoder):
        assert not module.training
        parameters = list(module.parameters())
        assert parameters
        assert all(not p.requires_grad for p in parameters)
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in parameters)
