# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""HunyuanImage-3.0 (tencent/HunyuanImage-3.0) preprocessing.

The release's VAE and prompt format come from the checkpoint's remote code
(``nemo_automodel.components.models.hunyuan_image3.release``), shared with sampling:

- Latents: the release VAE (``AutoencoderKLConv3D``), sampled from the posterior under fp16 autocast and scaled by
  ``vae.scaling_factor``, as the release encodes images; fp32, shape ``[32, H/16, W/16]``.
- Tokens: the release ``TokenizerWrapper.apply_chat_template`` in ``gen_image`` mode with classifier-free guidance,
  for the bucket resolution of the sample. The cache keeps the ids before the image span (ending in ``<timestep>``)
  for the prompt and for the unconditional ``<cfg>`` prompt, and the ids after it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from nemo_automodel.components.models.hunyuan_image3.release import HunyuanImage3PromptTokenizer, load_release_vae

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("hunyuan_image3")
class HunyuanImage3Processor(BaseModelProcessor):
    """Processor for HunyuanImage-3.0 text-to-image fine-tuning."""

    @property
    def model_type(self) -> str:
        return "hunyuan_image3"

    @property
    def default_model_name(self) -> str:
        return "tencent/HunyuanImage-3.0"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        from transformers import AutoConfig

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        model_dir = resolve_diffusion_model_dir(model_name)
        logger.info("[HunyuanImage-3.0] Loading VAE and tokenizer from %s", model_dir)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Token construction happens in get_cache_data, which does not receive the models dict.
        self._config = config
        self._prompt_tokenizer = HunyuanImage3PromptTokenizer.from_pretrained(model_dir, config)
        return {"vae": load_release_vae(model_dir, config.vae, device)}

    def target_resolution(self, width: int, height: int, bucket: Dict[str, Any]) -> tuple[int, int]:
        """Snap to the release's resolution group; the generic bucket's resolution is not used."""
        return self._prompt_tokenizer.target_size(width, height)

    def encode_image(self, image_tensor: torch.Tensor, models: Dict[str, Any], device: str) -> torch.Tensor:
        vae = models["vae"]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
            latent = vae.encode(image_tensor.to(device, dtype=torch.float32)).latent_dist.sample()
        latent = latent.float() * self.get_vae_scaling_factor(models)
        if latent.ndim == 5:
            if latent.shape[2] != 1:
                raise ValueError(f"Expected one latent frame for an image, got {latent.shape[2]}")
            latent = latent.squeeze(2)
        return latent.squeeze(0).cpu()

    def encode_text(self, prompt: str, models: Dict[str, Any], device: str) -> Dict[str, Any]:
        # The token sequence depends on the bucket resolution; get_cache_data builds it from metadata["prompt"].
        return {}

    def verify_latent(self, latent: torch.Tensor, models: Dict[str, Any], device: str) -> bool:
        channels = int(self._config.vae["latent_channels"])
        return bool(torch.isfinite(latent).all()) and latent.ndim == 3 and latent.shape[0] == channels

    def get_cache_data(
        self, latent: torch.Tensor, text_encodings: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        width, height = (int(v) for v in metadata["bucket_resolution"])
        tokens = self._prompt_tokenizer(metadata["prompt"], height, width)
        if tuple(latent.shape[-2:]) != (height // 16, width // 16):
            raise RuntimeError(f"Latent {tuple(latent.shape)} does not match bucket {width}x{height} at 16x.")
        return {
            "latent": latent,
            **tokens,
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "image_path": metadata["image_path"],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            "model_type": self.model_type,
        }
