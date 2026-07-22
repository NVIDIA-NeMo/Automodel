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

from unittest.mock import MagicMock, patch

import pytest
import torch

from tools.diffusion.processors._ltx2_audio import (
    LTX2_AUDIO_HOP_LENGTH,
    LTX2_AUDIO_N_MELS,
    LTX2_AUDIO_SAMPLE_RATE,
    LTX2MelSpectrogram,
    melscale_fbanks_slaney,
)
from tools.diffusion.processors.ltx2 import LTX2Processor
from tools.diffusion.processors.registry import ProcessorRegistry


@pytest.fixture
def processor():
    return LTX2Processor()


# ---------------------------------------------------------------------------
# Registry / property tests
# ---------------------------------------------------------------------------
class TestLTX2Properties:
    def test_registry_lookup(self):
        assert isinstance(ProcessorRegistry.get("ltx2"), LTX2Processor)

    def test_model_type(self, processor):
        assert processor.model_type == "ltx2"

    def test_model_version(self, processor):
        assert processor.model_version == "ltx2.3"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "dg845/LTX-2.3-Diffusers"

    def test_supported_modes(self, processor):
        assert processor.supported_modes == ["video"]

    def test_frame_constraint(self, processor):
        assert processor.frame_constraint == "8n+1"

    def test_quantization(self, processor):
        assert processor.quantization == 32

    def test_audio_constants(self, processor):
        assert processor.AUDIO_SAMPLE_RATE == 16000
        # 16000 / hop 160 / audio-VAE temporal compression 4
        assert processor.AUDIO_LATENT_FPS == 25.0


# ---------------------------------------------------------------------------
# Mel spectrogram (vendored) tests
# ---------------------------------------------------------------------------
class TestLTX2MelSpectrogram:
    def test_output_shape(self):
        mel = LTX2MelSpectrogram()
        samples = LTX2_AUDIO_SAMPLE_RATE  # 1 second
        out = mel(torch.randn(1, 2, samples))
        expected_frames = samples // LTX2_AUDIO_HOP_LENGTH + 1
        assert out.shape == (1, 2, LTX2_AUDIO_N_MELS, expected_frames)

    def test_magnitude_nonnegative_and_finite(self):
        mel = LTX2MelSpectrogram()
        out = mel(torch.randn(2, 16000))
        assert (out >= 0).all()
        assert torch.isfinite(out).all()

    def test_silence_maps_to_zeros(self):
        mel = LTX2MelSpectrogram()
        out = mel(torch.zeros(1, 2, 16000))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_pure_tone_peaks_in_expected_band(self):
        # A 440 Hz tone must put its energy in a low mel band, well below
        # the band a 4 kHz tone lands in.
        mel = LTX2MelSpectrogram()
        t = torch.arange(16000, dtype=torch.float32) / LTX2_AUDIO_SAMPLE_RATE
        low = mel(torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0))
        high = mel(torch.sin(2 * torch.pi * 4000.0 * t).unsqueeze(0))
        assert low.mean(dim=-1).argmax() < high.mean(dim=-1).argmax()

    def test_filterbank_shape_and_coverage(self):
        fb = melscale_fbanks_slaney(n_freqs=513, f_min=0.0, f_max=8000.0, n_mels=64, sample_rate=16000)
        assert fb.shape == (513, 64)
        assert (fb >= 0).all()
        # Every mel band must have at least one non-zero weight.
        assert (fb.max(dim=0).values > 0).all()


# ---------------------------------------------------------------------------
# encode_audio tests (mocked audio VAE)
# ---------------------------------------------------------------------------
def _mock_audio_models(latent_frames: int, mean: float = 0.0, std: float = 1.0):
    audio_vae = MagicMock()
    audio_vae.dtype = torch.float32
    audio_vae.latents_mean = torch.full((8 * 16,), mean)
    audio_vae.latents_std = torch.full((8 * 16,), std)
    dist = MagicMock()
    dist.mode.return_value = torch.randn(1, 8, latent_frames, 16)
    audio_vae.encode.return_value = (dist,)
    return {"audio_vae": audio_vae, "mel_transform": LTX2MelSpectrogram()}


class TestEncodeAudio:
    NUM_FRAMES = 121  # 8n+1; ~5.04 s at 24 fps
    EXPECTED_L = round(121 / 24.0 * 25.0)  # 126

    def _run(self, processor, latent_frames, **model_kwargs):
        models = _mock_audio_models(latent_frames, **model_kwargs)
        with patch.object(LTX2Processor, "load_audio", return_value=torch.zeros(2, round(121 / 24.0 * 16000))):
            return processor.encode_audio("clip.mp4", self.NUM_FRAMES, models, "cpu")

    def test_output_shape_and_dtype(self, processor):
        result = self._run(processor, self.EXPECTED_L)
        assert set(result.keys()) == {"audio_latents"}
        assert result["audio_latents"].shape == (1, 8, self.EXPECTED_L, 16)
        assert result["audio_latents"].dtype == torch.bfloat16

    def test_length_mismatch_raises(self, processor):
        with pytest.raises(ValueError, match="alignment"):
            self._run(processor, self.EXPECTED_L + 10)

    def test_off_by_one_length_tolerated(self, processor):
        result = self._run(processor, self.EXPECTED_L + 1)
        assert result["audio_latents"].shape[2] == self.EXPECTED_L + 1

    def test_normalization_applied(self, processor):
        models = _mock_audio_models(self.EXPECTED_L, mean=2.0, std=4.0)
        raw = models["audio_vae"].encode.return_value[0].mode.return_value
        with patch.object(LTX2Processor, "load_audio", return_value=torch.zeros(2, round(121 / 24.0 * 16000))):
            result = processor.encode_audio("clip.mp4", self.NUM_FRAMES, models, "cpu")
        expected = ((raw - 2.0) / 4.0).to(torch.bfloat16)
        torch.testing.assert_close(result["audio_latents"], expected)


# ---------------------------------------------------------------------------
# get_cache_data tests
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def _text_encodings(self):
        return {
            "text_embeddings": torch.randn(1, 32, 8),
            "audio_text_embeddings": torch.randn(1, 32, 6),
            "text_mask": torch.ones(1, 32, dtype=torch.long),
        }

    def test_cache_keys_and_passthrough(self, processor):
        latent = torch.randn(1, 128, 16, 8, 8)
        audio = torch.randn(1, 8, 126, 16)
        metadata = {
            "audio_latents": audio,
            "bucket_resolution": (256, 256),
            "num_frames": 121,
            "prompt": "a dog barking",
        }
        cache = processor.get_cache_data(latent, self._text_encodings(), metadata)

        assert torch.equal(cache["video_latents"], latent)
        assert torch.equal(cache["audio_latents"], audio)
        for key in ("text_embeddings", "audio_text_embeddings", "text_mask"):
            assert key in cache
        assert cache["model_type"] == "ltx2"
        assert cache["model_version"] == "ltx2.3"
        assert cache["prompt"] == "a dog barking"

    def test_missing_audio_latents_raises(self, processor):
        with pytest.raises(ValueError, match="audio latents"):
            processor.get_cache_data(torch.randn(1, 128, 4, 4, 4), self._text_encodings(), {})


# ---------------------------------------------------------------------------
# Round-trip (encode normalization <-> decode denormalization) tests
# ---------------------------------------------------------------------------
class TestRoundTripHelpers:
    def test_pixels_to_frames_range_mapping(self):
        from tools.diffusion.validate_ltx2_roundtrip import pixels_to_frames

        pixels = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)
        frames = pixels_to_frames(pixels)
        assert frames.shape == (1, 1, 1, 3)
        assert frames.reshape(-1).tolist() == [0, 128, 255]

    def test_audio_decode_denormalization_inverts_encode(self):
        # encode_audio stores (z - mean) / std on the flattened [B, L, C*M]
        # layout; decode_audio_latents must feed z back into the audio VAE.
        from tools.diffusion.validate_ltx2_roundtrip import decode_audio_latents

        torch.manual_seed(0)
        raw = torch.randn(1, 8, 20, 16)
        models = _mock_audio_models(latent_frames=20, mean=1.5, std=2.5)
        models["audio_vae"].encode.return_value[0].mode.return_value = raw
        with patch.object(LTX2Processor, "load_audio", return_value=torch.zeros(2, round(20 / 25 * 16000))):
            normalized = LTX2Processor().encode_audio("clip.mp4", round(20 / 25 * 24), models, "cpu")["audio_latents"]

        vocoder = MagicMock(return_value=torch.zeros(1, 2, 100))
        vocoder.config.output_sampling_rate = 48000
        models["audio_vae"].decode.return_value = (torch.zeros(1, 2, 100, 64),)
        models["audio_vae"].dtype = torch.float32
        waveform, rate = decode_audio_latents(normalized.float(), models["audio_vae"], vocoder, "cpu")

        assert rate == 48000
        assert waveform.shape == (2, 100)
        recovered = models["audio_vae"].decode.call_args.args[0]
        torch.testing.assert_close(recovered, raw, atol=2e-2, rtol=1e-2)  # bf16 cache quantization
