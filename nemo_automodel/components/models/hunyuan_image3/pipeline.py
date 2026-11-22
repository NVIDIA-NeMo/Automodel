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

"""HunyuanImage-3.0 text-to-image sampling with the native transformer.

Follows the release ``HunyuanImage3Text2ImagePipeline`` with its default settings: bf16 Gaussian latents, Euler
steps on the sigma schedule ``linspace(1, 0)`` shifted by ``flow_shift``, the model fed ``sigma * 1000``,
classifier-free guidance ``uncond + scale * (cond - uncond)`` over a ``[cond, uncond]`` batch, and the release VAE
decoding under fp16 autocast. The release caches the text keys and values across steps; recomputing them gives the
same result because text tokens never attend to the image.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.models.hunyuan_image3.flow_adapter import (
    PROMPT_IDS_KEY,
    PROMPT_SUFFIX_IDS_KEY,
    UNCOND_PROMPT_IDS_KEY,
)
from nemo_automodel.components.models.hunyuan_image3.release import HunyuanImage3PromptTokenizer, load_release_vae

logger = logging.getLogger(__name__)


@dataclass
class HunyuanImage3PipelineOutput:
    """Generated images, one per prompt."""

    images: list[Image.Image]


def flow_sigmas(num_inference_steps: int, flow_shift: float) -> torch.Tensor:
    """Sigma schedule of the release ``FlowMatchDiscreteScheduler`` (``shift`` set, ``reverse=True``).

    Returns:
        fp32 tensor of shape ``[num_inference_steps + 1]`` running from 1 to 0.
    """
    sigmas = torch.linspace(1, 0, num_inference_steps + 1)
    return flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)


class HunyuanImage3Pipeline:
    """Text-to-image sampler around a ``HunyuanImage3ForCausalMM`` transformer.

    Every rank of a sharded transformer must call it with the same prompt and seed: each denoising step is a
    collective forward pass.

    Args:
        transformer: The (possibly FSDP2 / expert-parallel sharded) ``HunyuanImage3ForCausalMM``.
        vae: The release VAE (``load_release_vae``).
        prompt_tokenizer: Token ids of the release prompt format.
        flow_shift: Shift of the sigma schedule.
        device: Device of the latents and token ids.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        prompt_tokenizer: HunyuanImage3PromptTokenizer,
        flow_shift: float,
        device: torch.device,
    ):
        self.transformer = transformer
        self.vae = vae
        self.prompt_tokenizer = prompt_tokenizer
        self.flow_shift = flow_shift
        self.device = device

    @classmethod
    def from_transformer(cls, transformer: torch.nn.Module, model_dir: str) -> "HunyuanImage3Pipeline":
        """Add the release VAE, prompt format and ``flow_shift`` from the checkpoint in ``model_dir``.

        Args:
            transformer: The loaded ``HunyuanImage3ForCausalMM``.
            model_dir: Local release checkpoint directory (remote code, VAE weights, generation config).
        """
        from transformers import AutoConfig

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        # The first import of the remote code copies it into the HF modules cache; concurrent ranks can read a
        # half-written module, so global rank 0 imports it before the others (as model_init does for remote code).
        with FirstRankPerNode():
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            vae = load_release_vae(model_dir, config.vae, device)
            prompt_tokenizer = HunyuanImage3PromptTokenizer.from_pretrained(model_dir, config)
        with open(os.path.join(model_dir, "generation_config.json")) as f:
            flow_shift = float(json.load(f).get("flow_shift", 3.0))
        return cls(transformer, vae, prompt_tokenizer, flow_shift, device)

    def _input_ids(self, prompt: str, height: int, width: int, with_uncond: bool) -> torch.Tensor:
        """Long ``[1 or 2, sequence]``: the prompt row, then the ``<cfg>`` row when ``with_uncond``."""
        tokens = self.prompt_tokenizer(prompt, height, width)
        image_ids = torch.full(((height // 16) * (width // 16),), self.transformer.config.image_token_id)
        prefixes = [tokens[PROMPT_IDS_KEY], tokens[UNCOND_PROMPT_IDS_KEY]][: 1 + with_uncond]
        rows = [torch.cat([prefix, image_ids, tokens[PROMPT_SUFFIX_IDS_KEY]]) for prefix in prefixes]
        return torch.stack(rows).to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        generator: torch.Generator | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
    ) -> HunyuanImage3PipelineOutput:
        """Generate one image.

        Args:
            prompt: Text prompt.
            generator: Seeds the initial latents (on ``self.device``).
            num_inference_steps: Number of Euler steps.
            guidance_scale: Classifier-free guidance scale; at most 1 disables guidance.
            height: Requested image height, snapped to the release resolution group.
            width: Requested image width, snapped to the release resolution group.

        Returns:
            ``HunyuanImage3PipelineOutput`` holding one PIL image.
        """
        target_width, target_height = self.prompt_tokenizer.target_size(width, height)
        if (target_width, target_height) != (width, height):
            logger.info("Snapped %dx%d to the release resolution %dx%d", width, height, target_width, target_height)
        width, height = target_width, target_height
        with_uncond = guidance_scale > 1.0
        input_ids = self._input_ids(prompt, height, width, with_uncond)

        model_dtype = next(self.transformer.parameters()).dtype
        channels = int(self.transformer.config.vae["latent_channels"])
        latents = torch.randn(
            (1, channels, height // 16, width // 16), generator=generator, device=self.device, dtype=torch.bfloat16
        ).float()
        # fp32 sigmas and timesteps, computed as the release scheduler does.
        sigmas = flow_sigmas(num_inference_steps, self.flow_shift).to(self.device)
        timesteps = sigmas[:-1] * 1000
        for step in range(num_inference_steps):
            batch = latents.expand(input_ids.shape[0], -1, -1, -1).to(model_dtype)
            (velocity,) = self.transformer(input_ids, batch, timesteps[step].repeat(input_ids.shape[0]))
            velocity = velocity.float()
            if with_uncond:
                cond, uncond = velocity.chunk(2)
                velocity = uncond + guidance_scale * (cond - uncond)
            latents = latents + velocity * (sigmas[step + 1] - sigmas[step])
        return HunyuanImage3PipelineOutput(images=[self._decode(latents, generator)])

    def _decode(self, latents: torch.Tensor, generator: torch.Generator | None) -> Image.Image:
        """Decode ``[1, channels, h, w]`` scaled latents to a PIL image, as the release pipeline does."""
        vae_config = self.vae.config
        latents = latents / vae_config.scaling_factor
        if getattr(vae_config, "shift_factor", None):
            latents = latents + vae_config.shift_factor
        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
        if image.ndim == 5:
            image = image.squeeze(2)
        image = (image[0].float() / 2 + 0.5).clamp(0, 1)
        array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array)
