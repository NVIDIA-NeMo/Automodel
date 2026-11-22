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

"""Typed component and cache contracts shared by FLUX.3 preprocessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from nemo_automodel.components.models.flux3.text_encoding import Flux3TextStack

if TYPE_CHECKING:
    from diffusers import AutoencoderKLFlux2


class Flux3Models(Flux3TextStack):
    """Frozen VAE, normalization buffers, and Qwen3-VL text stack."""

    vae: AutoencoderKLFlux2
    bn_mean: torch.Tensor
    bn_std: torch.Tensor


class Flux3CacheMetadata(TypedDict):
    """Image metadata supplied by the native preprocessing worker."""

    original_resolution: tuple[int, int]
    bucket_resolution: tuple[int, int]
    crop_offset: tuple[int, int]
    prompt: str
    image_path: str
    bucket_id: int
    aspect_ratio: float


class Flux3CacheData(Flux3CacheMetadata):
    """Tensor payload and metadata saved for a text-to-image example."""

    latent: torch.Tensor
    prompt_embeds: torch.Tensor
    model_type: str


class Flux3BucketSignature(TypedDict):
    """Latent shapes used to group compatible image-edit examples."""

    target: list[int]
    contexts: list[list[int]]


class Flux3EditRecord(TypedDict):
    """JSON index entry for one cached image-edit example."""

    cache_file: str
    compound_bucket_signature: Flux3BucketSignature
    original_id: str
    row_index: int
    target_token_length: int
    context_token_lengths: list[int]
    text_token_length: int
