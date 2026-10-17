# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waveform encoder for Muse Spark Open audio tokens."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MuseSparkAudioRMSNorm(nn.Module):
    """RMSNorm used by the waveform interaction encoder."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize audio states shaped ``[batch,time,hidden]`` in FP32."""
        output = hidden_states.float()
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).to(hidden_states.dtype)


def patchify_audio(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Fold ``[batch,samples,channels]`` into ``[batch,steps,patch*channels]``."""
    batch, length, channels = hidden_states.shape
    if length % patch_size:
        raise ValueError(f"Audio length {length} must be divisible by patch size {patch_size}")
    return hidden_states.reshape(batch, length // patch_size, patch_size * channels)


class MuseSparkAudioPatchifier(nn.Module):
    """Patch, project, and normalize one waveform stage."""

    def __init__(self, patch_size: int, input_size: int, output_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(input_size * patch_size, output_size, bias=True, dtype=dtype)
        self.norm = MuseSparkAudioRMSNorm(output_size, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform ``[batch,time,input]`` into a shorter projected sequence."""
        return self.norm(self.linear(patchify_audio(hidden_states, self.patch_size)))


class MuseSparkDepthwiseCausalConv(nn.Module):
    """Depthwise causal temporal convolution."""

    def __init__(self, hidden_size: int, kernel_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=True,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convolve audio states shaped ``[batch,time,hidden]`` causally."""
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.pad(hidden_states, (self.kernel_size - 1, 0))
        return self.conv(hidden_states).transpose(1, 2)


class MuseSparkAudioBlock(nn.Module):
    """Causal convolution followed by a GELU MLP residual branch."""

    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.conv = MuseSparkDepthwiseCausalConv(hidden_size, kernel_size, dtype)
        self.norm = MuseSparkAudioRMSNorm(hidden_size, dtype)
        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform audio states with shape ``[batch,time,hidden]``."""
        residual = hidden_states
        hidden_states = self.conv(hidden_states)
        hidden_states = self.linear2(F.gelu(self.linear1(self.norm(hidden_states))))
        return hidden_states + residual


class MuseSparkAudioEncoder(nn.Module):
    """Encode mono 24-kHz waveforms into one token per 1,920 samples."""

    def __init__(
        self,
        output_size: int,
        dtype: torch.dtype,
        *,
        patch_size: int = 1920,
        kernel_size: int = 7,
        input_channels: int = 1,
        patchify_sizes: Sequence[int] = (16, 6, 5, 4),
        hidden_sizes: Sequence[int] = (256, 512, 1024, 2048),
        layers_per_stage: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(patchify_sizes) != len(hidden_sizes) or len(hidden_sizes) != len(layers_per_stage):
            raise ValueError("Muse Spark Open audio stage settings must have equal lengths")
        if math.prod(patchify_sizes) != patch_size:
            raise ValueError("Muse Spark Open audio patch_size must equal the product of stage patch sizes")
        self.output_size = output_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        stages: list[nn.Sequential] = []
        stage_input = input_channels
        for stage_patch, stage_hidden, layer_count in zip(patchify_sizes, hidden_sizes, layers_per_stage):
            modules: list[nn.Module] = [MuseSparkAudioPatchifier(stage_patch, stage_input, stage_hidden, dtype)]
            modules.extend(
                MuseSparkAudioBlock(stage_hidden, stage_hidden * 4, kernel_size, dtype) for _ in range(layer_count)
            )
            stages.append(nn.Sequential(*modules))
            stage_input = stage_hidden
        self.stages = nn.Sequential(*stages)
        self.norm = MuseSparkAudioRMSNorm(stage_input, dtype)
        self.o_proj = nn.Linear(stage_input, output_size, bias=True, dtype=dtype)
        self.dual_norm = MuseSparkAudioRMSNorm(output_size, dtype)

    def _encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode a padded batch shaped ``[batch,samples,channels]``."""
        for stage in self.stages:
            hidden_states = stage(hidden_states)
        return self.dual_norm(self.o_proj(self.norm(hidden_states)))

    def forward(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode variable-length mono waveforms into ``[audio_tokens,output]`` tensors."""
        outputs = []
        for waveform in waveforms:
            waveform = waveform.reshape(-1, self.input_channels)
            padding = (-waveform.shape[0]) % self.patch_size
            waveform = F.pad(waveform, (0, 0, 0, padding)).unsqueeze(0)
            outputs.append(self._encode(waveform).squeeze(0))
        return outputs

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize a checkpoint-free audio encoder on ``buffer_device``."""
        with buffer_device:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, MuseSparkAudioRMSNorm):
                    nn.init.ones_(module.weight)


__all__ = ["MuseSparkAudioEncoder", "patchify_audio"]
