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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.flux3.text_encoding import (
    resolve_text_encoder_out_layers as _resolve_text_encoder_out_layers,
)
from tools.diffusion.processors.flux3 import Flux3Processor

# Layers stacked by the released FLUX.3 checkpoint (Qwen3-VL-4B geometry).
_CACHED_SEQUENCE_LENGTH = 80
_DEFAULT_OUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


@pytest.fixture
def processor():
    return Flux3Processor()


def _make_vae_models(latent_channels=32, h_patch=4, w_patch=4, bn_mean_val=0.0, bn_std_val=1.0):
    """Build a mocked models dict for encode_image / verify_latent.

    The mocked VAE encode returns a raw latent of shape
    [1, latent_channels, h_patch*2, w_patch*2] so patchify yields
    [1, latent_channels*4, h_patch, w_patch].
    """
    channels = latent_channels * 4
    raw_latent = torch.randn(1, latent_channels, h_patch * 2, w_patch * 2).float()

    vae = MagicMock()
    vae.config.latent_channels = latent_channels
    mock_dist = MagicMock()
    mock_dist.latent_dist.mode.return_value = raw_latent
    vae.encode.return_value = mock_dist

    models = {
        "vae": vae,
        "bn_mean": torch.full((1, channels, 1, 1), bn_mean_val),
        "bn_std": torch.full((1, channels, 1, 1), bn_std_val),
    }
    return models, raw_latent


def _make_text_models(hidden_dim=16, num_hidden_states=33):
    """Build mocked text_encoder + tokenizer producing deterministic hidden states.

    Each hidden state k is a constant tensor filled with k of shape
    [1, _CACHED_SEQUENCE_LENGTH, hidden_dim], so the stacked layer order is
    observable in the output embedding.
    """
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "templated prompt"
    tokenizer.tokenizer.return_value = {
        "input_ids": torch.ones(1, _CACHED_SEQUENCE_LENGTH, dtype=torch.long),
        "attention_mask": torch.ones(1, _CACHED_SEQUENCE_LENGTH, dtype=torch.long),
    }

    hidden_states = tuple(
        torch.full((1, _CACHED_SEQUENCE_LENGTH, hidden_dim), float(k)) for k in range(num_hidden_states)
    )
    out = MagicMock()
    out.hidden_states = hidden_states
    text_encoder = MagicMock(return_value=out)
    text_encoder.dtype = torch.float32
    text_encoder.device = torch.device("cpu")

    models = {
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "text_encoder_out_layers": _DEFAULT_OUT_LAYERS,
    }
    return models, hidden_states


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestFlux3Properties:
    def test_model_type(self, processor):
        assert processor.model_type == "flux3"

    def test_no_default_model_name(self, processor):
        # No public FLUX.3 checkpoint id yet: --model_name must be explicit.
        with pytest.raises(NotImplementedError):
            _ = processor.default_model_name


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------
class TestPatchifyLatents:
    def test_shape(self):
        latents = torch.randn(2, 32, 8, 10)
        patchified = Flux3Processor._patchify_latents(latents)
        assert patchified.shape == (2, 128, 4, 5)

    def test_values_match_manual_patchify(self):
        # Every 2x2 spatial patch moves into the channel dim: output channel
        # block c*4 + (dy*2 + dx) at (i, j) equals input channel c at (2i+dy, 2j+dx).
        latents = torch.randn(1, 2, 4, 4)
        patchified = Flux3Processor._patchify_latents(latents)
        for c in range(2):
            for dy in range(2):
                for dx in range(2):
                    out_channel = c * 4 + dy * 2 + dx
                    assert torch.equal(patchified[0, out_channel], latents[0, c, dy::2, dx::2])


# ---------------------------------------------------------------------------
# Encode image
# ---------------------------------------------------------------------------
class TestEncodeImage:
    def test_encode_image_shape_and_dtype(self, processor):
        models, _ = _make_vae_models(latent_channels=32, h_patch=4, w_patch=4)
        result = processor.encode_image(torch.randn(1, 3, 64, 64), models, device="cpu")
        assert result.shape == (128, 4, 4)
        assert result.dtype == torch.float32

    def test_encode_image_applies_bn_normalization(self, processor):
        models, raw_latent = _make_vae_models(latent_channels=32, h_patch=4, w_patch=4, bn_mean_val=1.0, bn_std_val=2.0)
        result = processor.encode_image(torch.randn(1, 3, 64, 64), models, device="cpu")
        expected = (Flux3Processor._patchify_latents(raw_latent) - 1.0) / 2.0
        assert torch.allclose(result.float(), expected.squeeze(0), atol=1e-2)


# ---------------------------------------------------------------------------
# Encode text
# ---------------------------------------------------------------------------
class TestEncodeText:
    def test_returns_expected_keys(self, processor):
        models, _ = _make_text_models()
        result = processor.encode_text("a fox", models, device="cpu")
        assert set(result.keys()) == {"prompt_embeds"}

    def test_shape_stacks_configured_layers(self, processor):
        hidden_dim = 16
        models, _ = _make_text_models(hidden_dim=hidden_dim)
        result = processor.encode_text("a fox", models, device="cpu")
        assert result["prompt_embeds"].shape == (
            _CACHED_SEQUENCE_LENGTH,
            len(_DEFAULT_OUT_LAYERS) * hidden_dim,
        )
        assert result["prompt_embeds"].dtype == torch.float16

    def test_layer_stacking_order(self, processor):
        # Hidden state k is constant k, so each hidden_dim-wide slice of the
        # output must equal the corresponding configured layer index, in order.
        hidden_dim = 4
        models, _ = _make_text_models(hidden_dim=hidden_dim)
        result = processor.encode_text("a fox", models, device="cpu")
        embeds = result["prompt_embeds"].float()
        for slot, layer_idx in enumerate(_DEFAULT_OUT_LAYERS):
            block = embeds[:, slot * hidden_dim : (slot + 1) * hidden_dim]
            assert (block == float(layer_idx)).all()

    def test_applies_chat_template_and_max_length(self, processor):
        models, _ = _make_text_models()
        processor.encode_text("a fox", models, device="cpu")
        models["tokenizer"].apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "a fox"}], tokenize=False, add_generation_prompt=True
        )
        tok_kwargs = models["tokenizer"].tokenizer.call_args[1]
        assert tok_kwargs["max_length"] == _CACHED_SEQUENCE_LENGTH
        assert tok_kwargs["padding"] == "max_length"
        assert tok_kwargs["truncation"] is True

    def test_matches_upstream_conditioning(self, processor: Flux3Processor) -> None:
        # The minimum supported Diffusers lacks the FLUX3 reference block.
        # Remove this guard when that block is included in the dependency floor.
        encoders = pytest.importorskip(
            "diffusers.modular_pipelines.flux3.encoders",
            reason="Upstream FLUX3 text-encoding parity requires a Diffusers build with FLUX3.",
            exc_type=ModuleNotFoundError,
        )
        models, hidden_states = _make_text_models()
        generator = torch.Generator().manual_seed(17)
        models["text_encoder"].return_value.hidden_states = tuple(
            torch.randn(state.shape, generator=generator) for state in hidden_states
        )

        actual = processor.encode_text("a fox", models, device="cpu")["prompt_embeds"]
        expected = encoders.Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
            text_encoder=models["text_encoder"],
            tokenizer=models["tokenizer"],
            prompt="a fox",
            device=torch.device("cpu"),
            max_sequence_length=8192,
            pad_multiple=80,
            hidden_states_layers=models["text_encoder_out_layers"],
        )
        torch.testing.assert_close(actual, expected.squeeze(0).to(torch.float16), rtol=0, atol=0)

    def test_requests_hidden_states(self, processor):
        models, _ = _make_text_models()
        processor.encode_text("a fox", models, device="cpu")
        enc_kwargs = models["text_encoder"].call_args[1]
        assert enc_kwargs["output_hidden_states"] is True
        assert enc_kwargs["use_cache"] is False


# ---------------------------------------------------------------------------
# Verify latent
# ---------------------------------------------------------------------------
class TestVerifyLatent:
    def _models(self, latent_channels=32):
        vae = MagicMock()
        vae.config.latent_channels = latent_channels
        return {"vae": vae}

    def test_valid_latent_passes(self, processor):
        assert processor.verify_latent(torch.randn(128, 4, 4), self._models(), device="cpu")

    def test_wrong_ndim_fails(self, processor):
        assert not processor.verify_latent(torch.randn(1, 128, 4, 4), self._models(), device="cpu")

    def test_wrong_channels_fails(self, processor):
        assert not processor.verify_latent(torch.randn(64, 4, 4), self._models(), device="cpu")

    def test_channel_count_follows_vae_config(self, processor):
        # Tiny checkpoints have fewer latent channels; the check must scale.
        assert processor.verify_latent(torch.randn(16, 4, 4), self._models(latent_channels=4), device="cpu")

    def test_nan_latent_fails(self, processor):
        latent = torch.randn(128, 4, 4)
        latent[0, 0, 0] = float("nan")
        assert not processor.verify_latent(latent, self._models(), device="cpu")

    def test_inf_latent_fails(self, processor):
        latent = torch.randn(128, 4, 4)
        latent[0, 0, 0] = float("inf")
        assert not processor.verify_latent(latent, self._models(), device="cpu")


# ---------------------------------------------------------------------------
# Cache data
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def test_cache_data_structure(self, processor):
        latent = torch.randn(128, 4, 4)
        text_encodings = {"prompt_embeds": torch.randn(_CACHED_SEQUENCE_LENGTH, 20480)}
        metadata = {
            "original_resolution": (512, 512),
            "bucket_resolution": (512, 512),
            "crop_offset": (0, 0),
            "prompt": "a fox",
            "image_path": "/data/fox.jpg",
            "bucket_id": "512x512",
            "aspect_ratio": 1.0,
        }
        cache = processor.get_cache_data(latent, text_encodings, metadata)
        assert torch.equal(cache["latent"], latent)
        assert torch.equal(cache["prompt_embeds"], text_encodings["prompt_embeds"])
        assert cache["model_type"] == "flux3"
        assert cache["prompt"] == "a fox"

    def test_no_clip_keys_in_cache(self, processor):
        cache = processor.get_cache_data(
            torch.randn(128, 4, 4),
            {"prompt_embeds": torch.randn(8, 16)},
            {
                "original_resolution": (512, 512),
                "bucket_resolution": (512, 512),
                "crop_offset": (0, 0),
                "prompt": "p",
                "image_path": "i",
                "bucket_id": "b",
                "aspect_ratio": 1.0,
            },
        )
        assert "clip_hidden" not in cache
        assert "pooled_prompt_embeds" not in cache


# ---------------------------------------------------------------------------
# Layer resolution from checkpoint geometry
# ---------------------------------------------------------------------------
class TestResolveTextEncoderOutLayers:
    def test_released_checkpoint_geometry(self):
        # Qwen3-VL-4B: hidden 2560, 36 layers; joint_attention_dim 20480.
        layers = _resolve_text_encoder_out_layers(20480, 2560, 36)
        assert layers == (4, 8, 12, 16, 20, 24, 28, 32)

    def test_tiny_checkpoint_geometry(self):
        # Tiny repo: hidden 16, 2 layers; joint_attention_dim 16 -> single layer.
        layers = _resolve_text_encoder_out_layers(16, 16, 2)
        assert layers == (2,)

    def test_non_multiple_dim_raises(self):
        with pytest.raises(ValueError, match="not a multiple"):
            _resolve_text_encoder_out_layers(20481, 2560, 36)

    def test_too_shallow_encoder_raises(self):
        with pytest.raises(ValueError, match="only has"):
            _resolve_text_encoder_out_layers(64, 16, 2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestFlux3Registry:
    def test_registered_name(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert "flux3" in ProcessorRegistry.list_available()

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert isinstance(ProcessorRegistry.get("flux3"), Flux3Processor)


@pytest.mark.parametrize("sequence_length", [80, 160])
def test_image_edit_cache_metadata_matches_variable_prompt_length(tmp_path, monkeypatch, sequence_length):
    import json

    from PIL import Image

    from nemo_automodel.components.datasets.diffusion.image_edit_dataset import ImageEditDataset
    from nemo_automodel.components.models.flux3 import preprocessing

    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "example",
                "prompt": "keep the image unchanged",
                "media": [
                    {"role": "target", "file_name": "image.png"},
                    {"role": "context", "file_name": "image.png"},
                ],
                "metadata": {"row_index": 0},
            }
        )
        + "\n"
    )
    encoder = preprocessing.Flux3ImageEditCacheEncoder(model_name="test-model", device="cpu")
    monkeypatch.setattr(encoder, "_load_models", lambda device: {})
    monkeypatch.setattr(encoder, "_encode_latent", lambda models, image, device: torch.zeros(128, 2, 2))
    monkeypatch.setattr(
        preprocessing, "encode_text_qwen3vl", lambda prompt, models, device: torch.ones(sequence_length, 16)
    )
    output = tmp_path / "cache"
    encoder.encode_manifest(
        manifest_path=manifest,
        output_dir=output,
        max_pixels=1024,
        resolution_preset=None,
        num_gpus=1,
        verify=True,
    )
    sample = ImageEditDataset(str(output), quantization=16)[0]
    assert sample["metadata"]["text_token_length"] == sequence_length
    assert sample["prompt_attention_mask"].numel() == sequence_length
    index = json.loads((output / "metadata_shard_0000.json").read_text())
    assert index[0]["text_token_length"] == sequence_length
