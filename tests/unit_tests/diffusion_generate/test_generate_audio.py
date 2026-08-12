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

"""Unit tests for audio-aware output saving in the diffusion generation script.

Covers the frame/waveform conversion helpers, audio sample-rate resolution, and
the ``run_inference`` save routing: dual-stream pipelines (e.g. LTX-2) return a
waveform alongside frames and must have it muxed into the mp4, while video-only
pipelines stay on the plain ``export_to_video`` path. The muxer itself is patched
out — these tests exercise routing and conversion, not PyAV encoding.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import examples.diffusion.generate.generate as gen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeVideoPipeline:
    """Stand-in video pipeline.

    ``num_frames`` in the call signature is what ``detect_output_type`` keys off
    to classify an unknown pipeline as video-producing.
    """

    def __init__(self, output, vocoder_rate=None):
        self._output = output
        if vocoder_rate is not None:
            self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=vocoder_rate))

    def __call__(self, prompt, generator=None, num_frames=None, **kwargs):
        return self._output


def _make_cfg(output_dir, audio_sample_rate=None):
    """Build the minimal config node ``run_inference`` reads."""
    return SimpleNamespace(
        model=SimpleNamespace(lora_weights=None),
        inference=SimpleNamespace(prompts=["a cat watching the rain"], dtype="float32"),
        output=SimpleNamespace(output_dir=str(output_dir), fps=24, audio_sample_rate=audio_sample_rate),
        seed=42,
    )


def _video_output(with_audio, num_frames=4, height=8, width=8, samples=100):
    """Build a pipeline output holding frames [frames, H, W, 3] and optional audio."""
    frames = np.random.rand(num_frames, height, width, 3).astype(np.float32)
    if not with_audio:
        return SimpleNamespace(frames=[frames])
    return SimpleNamespace(frames=[frames], audio=torch.randn(1, 2, samples).clamp(-1.0, 1.0))


def _run_inference(pipe, cfg):
    """Run ``run_inference`` with the muxer and video writer patched out.

    Returns:
        Tuple of (mux mock, export_to_video mock).
    """
    # run_inference hard-codes a CUDA generator; these tests run on CPU.
    with (
        patch.object(gen.torch, "Generator", return_value=MagicMock()),
        patch.object(gen, "_write_video_with_audio") as mux,
        patch("diffusers.utils.export_to_video") as export_to_video,
    ):
        gen.run_inference(pipe, cfg, True)
    return mux, export_to_video


# ---------------------------------------------------------------------------
# Frame / waveform conversion
# ---------------------------------------------------------------------------
class TestFramesToUint8:
    def test_float_ndarray_scaled_to_uint8(self):
        frames = gen._frames_to_uint8(np.zeros((5, 8, 8, 3), dtype=np.float32) + 1.0)
        assert frames.shape == (5, 8, 8, 3)
        assert frames.dtype == np.uint8
        assert frames.min() == 255

    def test_channels_first_tensor_transposed(self):
        frames = gen._frames_to_uint8(torch.rand(5, 3, 8, 16))
        assert frames.shape == (5, 8, 16, 3)
        assert frames.dtype == np.uint8

    def test_uint8_input_passes_through_unscaled(self):
        source = np.full((3, 4, 4, 3), 200, dtype=np.uint8)
        frames = gen._frames_to_uint8(source)
        assert frames.dtype == np.uint8
        assert np.array_equal(frames, source)

    def test_out_of_range_floats_clipped(self):
        frames = gen._frames_to_uint8(np.array([[[[-5.0, 0.5, 5.0]]]], dtype=np.float32))
        assert frames.reshape(-1).tolist() == [0, 128, 255]

    def test_pil_image_list(self):
        pil = pytest.importorskip("PIL.Image")
        images = [pil.new("RGB", (6, 4), color=(10, 20, 30)) for _ in range(3)]
        frames = gen._frames_to_uint8(images)
        assert frames.shape == (3, 4, 6, 3)  # PIL size is (width, height)
        assert frames[0, 0, 0].tolist() == [10, 20, 30]


class TestWaveformTo2d:
    def test_leading_batch_dim_squeezed(self):
        assert gen._waveform_to_2d(torch.randn(1, 2, 100)).shape == (2, 100)

    def test_mono_1d_gains_channel_dim(self):
        assert gen._waveform_to_2d(torch.randn(100)).shape == (1, 100)

    def test_numpy_input(self):
        wav = gen._waveform_to_2d(np.random.randn(2, 100))
        assert wav.shape == (2, 100)
        assert wav.dtype == torch.float32

    def test_list_output_takes_first_sample(self):
        assert gen._waveform_to_2d([torch.randn(2, 64), torch.randn(2, 64)]).shape == (2, 64)

    def test_values_clamped_to_unit_range(self):
        wav = gen._waveform_to_2d(torch.tensor([[-3.0, 0.25, 3.0]]))
        assert wav.max().item() == 1.0
        assert wav.min().item() == -1.0


# ---------------------------------------------------------------------------
# Muxing
# ---------------------------------------------------------------------------
class TestWriteVideoWithAudio:
    """Round-trip the muxer through a real file; PyAV ships in the optional
    ``diffusion-media`` extra, so skip when it is absent."""

    def test_mp4_carries_both_streams(self, tmp_path):
        av = pytest.importorskip("av")

        num_frames, height, width, sample_rate, fps = 12, 64, 96, 48000, 24
        frames = np.full((num_frames, height, width, 3), 128, dtype=np.uint8)
        ramp = torch.linspace(-1.0, 1.0, sample_rate // 2)
        waveform = torch.stack([ramp, ramp])  # [channels, samples]

        path = tmp_path / "muxed.mp4"
        gen._write_video_with_audio(path, frames, fps, waveform, sample_rate)
        assert path.stat().st_size > 0

        with av.open(str(path)) as container:
            streams = {s.type: s for s in container.streams}
            assert streams["video"].codec_context.name == "h264"
            assert (streams["video"].codec_context.width, streams["video"].codec_context.height) == (width, height)
            assert streams["audio"].codec_context.name == "aac"
            assert streams["audio"].codec_context.sample_rate == sample_rate
            assert streams["audio"].codec_context.channels == 2
            assert sum(1 for _ in container.decode(video=0)) == num_frames

    def test_mono_waveform_is_accepted(self, tmp_path):
        av = pytest.importorskip("av")

        path = tmp_path / "mono.mp4"
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        gen._write_video_with_audio(path, frames, 24, torch.zeros(1, 8000), 16000)

        with av.open(str(path)) as container:
            audio = next(s for s in container.streams if s.type == "audio")
            assert audio.codec_context.channels == 1


# ---------------------------------------------------------------------------
# Sample-rate resolution
# ---------------------------------------------------------------------------
class TestResolveAudioSampleRate:
    def test_config_value_wins_over_vocoder(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True), vocoder_rate=48000)
        assert gen._resolve_audio_sample_rate(pipe, _make_cfg(tmp_path, audio_sample_rate=16000)) == 16000

    def test_falls_back_to_vocoder_rate(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True), vocoder_rate=48000)
        assert gen._resolve_audio_sample_rate(pipe, _make_cfg(tmp_path)) == 48000

    def test_raises_when_rate_is_unavailable(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True))  # no vocoder
        with pytest.raises(ValueError, match="audio_sample_rate"):
            gen._resolve_audio_sample_rate(pipe, _make_cfg(tmp_path))


# ---------------------------------------------------------------------------
# run_inference save routing
# ---------------------------------------------------------------------------
class TestRunInferenceSaving:
    def test_video_only_pipeline_uses_export_to_video(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=False))
        mux, export_to_video = _run_inference(pipe, _make_cfg(tmp_path))

        assert export_to_video.call_count == 1
        mux.assert_not_called()
        assert export_to_video.call_args.kwargs["fps"] == 24
        assert export_to_video.call_args.args[1].endswith(".mp4")

    def test_audio_output_is_muxed_not_dropped(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True, num_frames=6, samples=128), vocoder_rate=48000)
        mux, export_to_video = _run_inference(pipe, _make_cfg(tmp_path))

        assert mux.call_count == 1
        export_to_video.assert_not_called()
        path, frames, fps, waveform, sample_rate = mux.call_args.args
        assert str(path).endswith(".mp4")
        assert frames.shape == (6, 8, 8, 3)
        assert frames.dtype == np.uint8
        assert fps == 24
        assert waveform.shape == (2, 128)
        assert sample_rate == 48000

    def test_config_sample_rate_reaches_the_muxer(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True), vocoder_rate=48000)
        mux, _ = _run_inference(pipe, _make_cfg(tmp_path, audio_sample_rate=16000))
        assert mux.call_args.args[4] == 16000

    def test_audio_without_resolvable_rate_raises(self, tmp_path):
        pipe = _FakeVideoPipeline(_video_output(with_audio=True))  # no vocoder, no config rate
        with pytest.raises(ValueError, match="audio_sample_rate"):
            _run_inference(pipe, _make_cfg(tmp_path))
