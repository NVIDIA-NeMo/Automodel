# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#
# The mel filterbank and spectrogram math below is adapted from torchaudio
# v2.10 (BSD-2-Clause, Copyright (c) 2017 Facebook Inc. (Soumith Chintala)):
#   https://github.com/pytorch/audio/blob/release/2.10/src/torchaudio/functional/functional.py
#   https://github.com/pytorch/audio/blob/release/2.10/src/torchaudio/transforms/_transforms.py
# Vendored so LTX-2 preprocessing does not require a torchaudio dependency,
# while staying numerically identical to the LTX-2 reference pipeline.

"""Pure-torch mel spectrogram for LTX-2 audio preprocessing.

LTX-2's audio VAE consumes log-mel spectrograms computed with a fixed
configuration (16 kHz, 1024-point FFT, hop 160, 64 slaney-scale/slaney-norm
mel bins, magnitude power 1.0). This module provides exactly that transform.
"""

import math

import torch
from torch import Tensor, nn

LTX2_AUDIO_SAMPLE_RATE = 16000
LTX2_AUDIO_N_FFT = 1024
LTX2_AUDIO_WIN_LENGTH = 1024
LTX2_AUDIO_HOP_LENGTH = 160
LTX2_AUDIO_F_MIN = 0.0
LTX2_AUDIO_F_MAX = 8000.0
LTX2_AUDIO_N_MELS = 64


def _hz_to_mel_slaney(freq: float) -> float:
    """Convert a frequency in Hz to the slaney mel scale."""
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: Tensor) -> Tensor:
    """Convert slaney-scale mel bin values to frequencies in Hz.

    Args:
        mels: Mel frequencies of shape [n_mels + 2].

    Returns:
        Frequencies in Hz, same shape as ``mels``.
    """
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def melscale_fbanks_slaney(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
) -> Tensor:
    """Create a slaney-scale, slaney-normalized triangular mel filterbank.

    Args:
        n_freqs: Number of STFT frequency bins (``n_fft // 2 + 1``).
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.
        n_mels: Number of mel bands.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Filterbank matrix of shape [n_freqs, n_mels]; a spectrogram of shape
        [..., n_freqs] is projected to mel via ``spec @ fb``.
    """
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    m_min = _hz_to_mel_slaney(f_min)
    m_max = _hz_to_mel_slaney(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz_slaney(m_pts)

    # Triangular filters (librosa-style).
    f_diff = f_pts[1:] - f_pts[:-1]  # [n_mels + 1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # [n_freqs, n_mels + 2]
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(torch.zeros(1), torch.min(down_slopes, up_slopes))

    # Slaney-style area normalization: approximately constant energy per band.
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    fb *= enorm.unsqueeze(0)
    return fb


class LTX2MelSpectrogram(nn.Module):
    """Magnitude mel spectrogram with the fixed LTX-2 audio configuration.

    Numerically equivalent to ``torchaudio.transforms.MelSpectrogram`` with
    ``sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
    f_min=0.0, f_max=8000.0, n_mels=64, power=1.0, mel_scale="slaney",
    norm="slaney", center=True, pad_mode="reflect"`` and a Hann window.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(LTX2_AUDIO_WIN_LENGTH), persistent=False)
        self.register_buffer(
            "fb",
            melscale_fbanks_slaney(
                n_freqs=LTX2_AUDIO_N_FFT // 2 + 1,
                f_min=LTX2_AUDIO_F_MIN,
                f_max=LTX2_AUDIO_F_MAX,
                n_mels=LTX2_AUDIO_N_MELS,
                sample_rate=LTX2_AUDIO_SAMPLE_RATE,
            ),
            persistent=False,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute the magnitude mel spectrogram.

        Args:
            waveform: Audio of shape [..., time] at 16 kHz.

        Returns:
            Mel spectrogram of shape [..., 64, n_frames] where
            ``n_frames = time // 160 + 1`` (center-padded STFT).
        """
        shape = waveform.size()
        flat = waveform.reshape(-1, shape[-1])
        spec = torch.stft(
            input=flat,
            n_fft=LTX2_AUDIO_N_FFT,
            hop_length=LTX2_AUDIO_HOP_LENGTH,
            win_length=LTX2_AUDIO_WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.reshape(shape[:-1] + spec.shape[-2:]).abs()  # [..., n_freqs, time]
        # Project STFT bins onto the mel basis: [..., time, n_freqs] @ [n_freqs, n_mels].
        return torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)
