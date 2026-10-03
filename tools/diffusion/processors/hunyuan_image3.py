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

The release ships its VAE, tokenizer wrapper and image processor as remote code inside the checkpoint. They are
loaded from the checkpoint at preprocessing time (``trust_remote_code``) and not vendored here:

- Latents: the release VAE (``AutoencoderKLConv3D``), sampled from the posterior under fp16 autocast and scaled by
  ``vae.scaling_factor``, as the release encodes images; shape ``[32, H/16, W/16]``.
- Tokens: the release ``TokenizerWrapper.apply_chat_template`` in ``gen_image`` mode with classifier-free guidance,
  for the bucket resolution of the sample. The cache keeps the ids before the image span (ending in ``<timestep>``)
  for the prompt and for the unconditional ``<cfg>`` prompt, and the ids after it.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from typing import Any, Dict

import torch

from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import (
    PROMPT_IDS_KEY,
    PROMPT_SUFFIX_IDS_KEY,
    UNCOND_PROMPT_IDS_KEY,
)

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


def _load_vae(model_dir: str, config: Any, device: str) -> torch.nn.Module:
    """Build the release VAE from the checkpoint's remote code and load its ``vae.*`` weights."""
    from safetensors import safe_open
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    vae_cls = get_class_from_dynamic_module("autoencoder_kl_3d.AutoencoderKLConv3D", model_dir)
    vae = vae_cls.from_config(config.vae)
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith("vae.")})
    state_dict = {}
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as reader:
            for key in reader.keys():
                if key.startswith("vae."):
                    state_dict[key[len("vae.") :]] = reader.get_tensor(key)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VAE weights do not match: missing={missing[:5]} unexpected={unexpected[:5]}")
    return vae.to(device=device, dtype=torch.float32).eval()


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
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        from nemo_automodel._diffusers._hf_cache import resolve_diffusion_model_dir

        model_dir = resolve_diffusion_model_dir(model_name)
        logger.info("[HunyuanImage-3.0] Loading VAE and tokenizer from %s", model_dir)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        image_processor_cls = get_class_from_dynamic_module("image_processor.HunyuanImage3ImageProcessor", model_dir)
        # Take the wrapper from the package the image processor imported it from: a second dynamic load can create a
        # separate module copy whose ImageInfo fails the wrapper's isinstance checks.
        package = image_processor_cls.__module__.rsplit(".", 1)[0]
        wrapper_cls = importlib.import_module(f"{package}.tokenizer_wrapper").TokenizerWrapper
        with open(os.path.join(model_dir, "generation_config.json")) as f:
            sequence_template = json.load(f).get("sequence_template", "pretrain")
        # Token construction happens in get_cache_data, which does not receive the models dict.
        self._config = config
        self._wrapper = wrapper_cls(tokenizer)
        self._image_processor = image_processor_cls(config)
        self._sequence_template = sequence_template
        return {"vae": _load_vae(model_dir, config, device)}

    def target_resolution(self, width: int, height: int, bucket: Dict[str, Any]) -> tuple[int, int]:
        """Snap to the release's resolution group (33 aspect ratios around ``image_base_size``).

        The release only generates these sizes, and its ``<img_ratio_*>`` token names one of them; the generic
        bucket's resolution is not used.
        """
        target_width, target_height = self._image_processor.reso_group.get_target_size(width, height)
        return int(target_width), int(target_height)

    def encode_image(self, image_tensor: torch.Tensor, models: Dict[str, Any], device: str) -> torch.Tensor:
        vae = models["vae"]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
            latent = vae.encode(image_tensor.to(device, dtype=torch.float32)).latent_dist.sample()
        latent = latent.float() * self.get_vae_scaling_factor(models)
        if latent.ndim == 5:
            if latent.shape[2] != 1:
                raise ValueError(f"Expected one latent frame for an image, got {latent.shape[2]}")
            latent = latent.squeeze(2)
        return latent.squeeze(0).cpu().to(torch.bfloat16)

    def encode_text(self, prompt: str, models: Dict[str, Any], device: str) -> Dict[str, Any]:
        # The token sequence depends on the bucket resolution; get_cache_data builds it from metadata["prompt"].
        return {}

    def build_prompt_tokens(self, prompt: str, height: int, width: int) -> Dict[str, torch.Tensor]:
        """Return the release's token ids around the image span for one prompt and image size.

        Returns:
            ``prompt_input_ids`` / ``uncond_prompt_input_ids``: ids before the image span (ending in ``<timestep>``);
            ``prompt_suffix_ids``: ids after the image span.
        """
        info = self._image_processor.build_image_info(f"{height}x{width}")
        out = self._wrapper.apply_chat_template(
            batch_prompt=[prompt],
            batch_message_list=None,
            mode="gen_image",
            batch_gen_image_info=[info],
            batch_cond_image_info=None,
            batch_system_prompt=None,
            batch_cot_text=None,
            max_length=None,
            bot_task="auto",
            image_base_size=self._config.image_base_size,
            sequence_template=self._sequence_template,
            cfg_factor=2,
            drop_think=False,
        )["output"]
        cond, uncond = out.tokens[0], out.tokens[1]
        span = out.gen_image_slices[0][0]
        if out.gen_image_slices[1][0] != span:
            raise RuntimeError("Conditional and unconditional sequences place the image at different positions.")
        if span.stop - span.start != info.token_height * info.token_width:
            raise RuntimeError(f"Image span {span} does not hold {info.token_height}x{info.token_width} tokens.")
        timestep_index = int(out.gen_timestep_scatter_index[0].reshape(-1)[0])
        if timestep_index != span.start - 1:
            raise RuntimeError(f"Expected <timestep> right before the image, got index {timestep_index} vs {span}.")
        return {
            PROMPT_IDS_KEY: cond[: span.start].clone(),
            UNCOND_PROMPT_IDS_KEY: uncond[: span.start].clone(),
            PROMPT_SUFFIX_IDS_KEY: cond[span.stop :].clone(),
        }

    def verify_latent(self, latent: torch.Tensor, models: Dict[str, Any], device: str) -> bool:
        channels = int(self._config.vae["latent_channels"])
        return bool(torch.isfinite(latent).all()) and latent.ndim == 3 and latent.shape[0] == channels

    def get_cache_data(
        self, latent: torch.Tensor, text_encodings: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        width, height = (int(v) for v in metadata["bucket_resolution"])
        tokens = self.build_prompt_tokens(metadata["prompt"], height, width)
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
