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

"""
HunyuanImage-3.0 processor for preprocessing.

HunyuanImage-3.0 conditions on a token sequence built by its chat template rather than on text-encoder embeddings,
and that sequence depends on the target image size. The processor therefore caches:

- ``latent``: VAE latent ``[32, H/16, W/16]`` with the release scaling applied
- ``conditioning``: the conditional and unconditional (classifier-free guidance) token sequences, image-slot masks,
  timestep-token positions and image-block placement, built by the release's own input builder

Everything is produced by the checkpoint's remote code (VAE encode, chat template, size tokens), so the cache matches
what the released inference pipeline feeds the model.
"""

import logging

import torch

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("hunyuan_image3")
class HunyuanImage3Processor(BaseModelProcessor):
    """Processor for tencent/HunyuanImage-3.0 text-to-image finetuning caches."""

    @property
    def model_type(self) -> str:
        return "hunyuan_image3"

    @property
    def default_model_name(self) -> str:
        return "tencent/HunyuanImage-3.0"

    def load_models(self, model_name: str, device: str) -> dict[str, torch.nn.Module]:
        """Load the release model without decoder layers: only the VAE, tokenizer and input builder are needed."""
        from transformers import AutoTokenizer

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        model_dir = resolve_diffusion_model_dir(model_name)
        logger.info("[HunyuanImage-3.0] Loading VAE and input builder from %s", model_dir)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Use the release classes directly: nemo_automodel registers its own config for this model_type, so the
        # Auto factories would resolve to it, and it lacks the release's image/tokenizer defaults.
        release_cls = get_class_from_dynamic_module("hunyuan.HunyuanImage3ForCausalMM", model_dir)
        config = release_cls.config_class.from_pretrained(model_dir)
        config.num_hidden_layers = 0
        for key in ("moe_topk", "moe_intermediate_size", "num_shared_expert"):
            if isinstance(config.to_dict()[key], list):
                setattr(config, key, [])
        config.moe_impl = "eager"
        model = release_cls.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
        ).eval()
        # Pass a tokenizer object so the release wrapper does not prompt for remote code interactively.
        model.load_tokenizer(AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True))
        return {"model": model, "vae": model.vae}

    def encode_image(self, image_tensor: torch.Tensor, models: dict[str, torch.nn.Module], device: str) -> torch.Tensor:
        """Encode an image in ``[-1, 1]`` (``[1, 3, H, W]``) to a scaled VAE latent ``[32, H/16, W/16]``."""
        model = models["model"]
        with torch.no_grad():
            _, latents = model.vae_encode(image_tensor.to(device=device, dtype=torch.float32))
        return latents.detach().float().cpu().squeeze(0)

    def encode_text(self, prompt: str, models: dict[str, torch.nn.Module], device: str) -> dict[str, str]:
        """The token sequence depends on the image size, so it is built in :meth:`get_cache_data`."""
        self._models = models
        return {"prompt": prompt}

    def build_conditioning(
        self, prompt: str, width: int, height: int, models: dict[str, torch.nn.Module]
    ) -> dict[str, torch.Tensor | int]:
        """Conditional and unconditional token sequences for a ``width x height`` target image."""
        model = models["model"]
        info = model.image_processor.build_image_info((height, width))
        if (info.image_width, info.image_height) != (width, height):
            raise ValueError(
                f"Bucket {width}x{height} is not a HunyuanImage-3.0 resolution (nearest supported: "
                f"{info.image_width}x{info.image_height}); preprocess with a supported preset."
            )
        with torch.no_grad():
            inputs = model.prepare_model_inputs(prompt=prompt, mode="gen_image", image_size=(height, width), seed=0)
        output = inputs["tokenizer_output"]
        conditioning: dict[str, torch.Tensor | int] = {"token_h": info.token_height, "token_w": info.token_width}
        # Row 0 is the conditional prompt, row 1 the unconditional one (classifier-free guidance pair).
        for row, prefix in ((0, ""), (1, "uncond_")):
            image_slices = output.gen_image_slices[row]
            if len(image_slices) != 1:
                raise ValueError(f"Expected one generated image block, got {len(image_slices)}")
            conditioning[prefix + "input_ids"] = inputs["input_ids"][row].detach().cpu().to(torch.int32)
            conditioning[prefix + "image_mask"] = inputs["image_mask"][row].detach().cpu().bool()
            conditioning[prefix + "timestep_scatter_index"] = (
                inputs["gen_timestep_scatter_index"][row].detach().cpu().to(torch.int32)
            )
            conditioning[prefix + "image_start"] = int(image_slices[0].start)
        return conditioning

    def verify_latent(self, latent: torch.Tensor, models: dict[str, torch.nn.Module], device: str) -> bool:
        """Decode the latent back to pixels and check for a finite RGB image."""
        try:
            vae = models["vae"]
            # The release VAE scales latents by scaling_factor and has no shift factor.
            z = latent.unsqueeze(0).to(device=device, dtype=torch.float32) / vae.config.scaling_factor
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                image = vae.decode(z.unsqueeze(2), return_dict=False)[0]
            return image.shape[1] == 3 and bool(torch.isfinite(image).all())
        except Exception as e:  # noqa: BLE001 - verification reports failure instead of aborting preprocessing
            logger.warning("[HunyuanImage-3.0] Verification failed: %s", e)
            return False

    def get_cache_data(
        self, latent: torch.Tensor, text_encodings: dict[str, str], metadata: dict[str, object]
    ) -> dict[str, object]:
        """Assemble the cache entry; ``bucket_resolution`` is ``(width, height)`` in pixels."""
        width, height = metadata["bucket_resolution"]
        return {
            "latent": latent,
            "conditioning": self.build_conditioning(text_encodings["prompt"], width, height, self._models),
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "image_path": metadata["image_path"],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            "model_type": self.model_type,
        }
