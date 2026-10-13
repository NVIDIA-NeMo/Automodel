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

"""The release's VAE and prompt format, loaded from the checkpoint's remote code.

The release ships its VAE, tokenizer wrapper and image processor as remote code inside the checkpoint
(``trust_remote_code``); they are not vendored here. Preprocessing and sampling both go through this module, so the
latents and token sequences a model is trained on are the ones it is sampled with.
"""

import importlib
import json
import os
from typing import Any

import torch

from nemo_automodel.components.models.hunyuan_image3.flow_adapter import (
    PROMPT_IDS_KEY,
    PROMPT_SUFFIX_IDS_KEY,
    UNCOND_PROMPT_IDS_KEY,
)


def load_release_vae(model_dir: str, vae_config: dict[str, Any], device: str | torch.device) -> torch.nn.Module:
    """Build the release VAE (``AutoencoderKLConv3D``) and load the checkpoint's ``vae.*`` weights.

    Args:
        model_dir: Local checkpoint directory.
        vae_config: The ``vae`` section of the checkpoint config.
        device: Device for the returned VAE.

    Returns:
        The VAE in fp32 and eval mode.
    """
    from safetensors import safe_open
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    vae_cls = get_class_from_dynamic_module("autoencoder_kl_3d.AutoencoderKLConv3D", model_dir)
    vae = vae_cls.from_config(vae_config)
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


class HunyuanImage3PromptTokenizer:
    """Token ids of the release's text-to-image sequence around the image span.

    Args:
        wrapper: The release ``TokenizerWrapper``.
        image_processor: The release ``HunyuanImage3ImageProcessor`` (resolution group and image token grid).
        image_base_size: ``image_base_size`` of the checkpoint config.
        sequence_template: ``sequence_template`` of the checkpoint's generation config.
    """

    def __init__(self, wrapper: Any, image_processor: Any, image_base_size: int, sequence_template: str):
        self.wrapper = wrapper
        self.image_processor = image_processor
        self.image_base_size = image_base_size
        self.sequence_template = sequence_template

    @classmethod
    def from_pretrained(cls, model_dir: str, config: Any) -> "HunyuanImage3PromptTokenizer":
        """Load the release tokenizer wrapper and image processor from ``model_dir``.

        Args:
            model_dir: Local checkpoint directory.
            config: The checkpoint config (needs ``image_base_size``).
        """
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        image_processor_cls = get_class_from_dynamic_module("image_processor.HunyuanImage3ImageProcessor", model_dir)
        # Take the wrapper from the package the image processor imported it from: a second dynamic load can create a
        # separate module copy whose ImageInfo fails the wrapper's isinstance checks.
        package = image_processor_cls.__module__.rsplit(".", 1)[0]
        wrapper_cls = importlib.import_module(f"{package}.tokenizer_wrapper").TokenizerWrapper
        with open(os.path.join(model_dir, "generation_config.json")) as f:
            sequence_template = json.load(f).get("sequence_template", "pretrain")
        return cls(wrapper_cls(tokenizer), image_processor_cls(config), config.image_base_size, sequence_template)

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        """Snap ``width`` x ``height`` to the release's resolution group (33 ratios around ``image_base_size``).

        Returns:
            ``(width, height)`` of the closest size the release generates; its ``<img_ratio_*>`` token names it.
        """
        target_width, target_height = self.image_processor.reso_group.get_target_size(width, height)
        return int(target_width), int(target_height)

    def __call__(self, prompt: str, height: int, width: int) -> dict[str, torch.Tensor]:
        """Return the token ids around the image span for one prompt and image size.

        The sequence is the release's ``gen_image`` chat template with classifier-free guidance (``bot_task`` auto,
        no system prompt), which is also what its ``generate_image`` samples with.

        Returns:
            ``prompt_input_ids`` / ``uncond_prompt_input_ids``: 1D long ids before the image span, ending in
            ``<timestep>`` (the unconditional one holds ``<cfg>`` tokens in place of the prompt);
            ``prompt_suffix_ids``: 1D long ids after the image span.
        """
        info = self.image_processor.build_image_info(f"{height}x{width}")
        out = self.wrapper.apply_chat_template(
            batch_prompt=[prompt],
            batch_message_list=None,
            mode="gen_image",
            batch_gen_image_info=[info],
            batch_cond_image_info=None,
            batch_system_prompt=None,
            batch_cot_text=None,
            max_length=None,
            bot_task="auto",
            image_base_size=self.image_base_size,
            sequence_template=self.sequence_template,
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
