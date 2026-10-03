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
HunyuanImage-3.0 model adapter for FlowMatching Pipeline.

HunyuanImage-3.0 is a unified autoregressive MoE model: the noisy target latents are embedded into image slots of a
token sequence built from the release chat template, and the model predicts the flow velocity for those slots.
The preprocessing cache stores, per sample, the token sequence of the conditional prompt (and of the unconditional
prompt used for classifier-free guidance):

- ``input_ids``: ``[S]`` token ids (image and timestep slots hold placeholder ids)
- ``image_mask``: ``[S]`` True at the target-image slots
- ``timestep_scatter_index``: ``[n]`` positions of the timestep tokens
- ``image_start``, ``token_h``, ``token_w``: placement and size of the image block

and the same keys with an ``uncond_`` prefix. The release samples with sigma running from 1 (noise) to 0 and steps
``x <- x + v * (sigma_next - sigma)``, so the model predicts ``noise - x0``, the flow-matching target used by the
pipeline; its timestep input is ``sigma * 1000``.
"""

import random

import torch
import torch.nn as nn

from nemo_automodel.components.models.hunyuan_image3.rope import build_2d_positions

from .base import FlowMatchingContext, ModelAdapter


def text_causal_image_bidirectional_mask(seq_len: int, image_start: int, image_len: int, device) -> torch.Tensor:
    """``[1, 1, S, S]`` boolean mask: causal everywhere, bidirectional inside the image block."""
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    mask[image_start : image_start + image_len, image_start : image_start + image_len] = True
    return mask[None, None]


class HunyuanImage3Adapter(ModelAdapter):
    """Flow-matching adapter for Automodel's HunyuanImage-3.0 model.

    Every sample is run as its own forward call: samples differ in sequence length, and the call count must equal the
    local batch size on every rank so FSDP/EP collectives stay aligned.
    """

    def prepare_inputs(self, context: FlowMatchingContext) -> dict[str, object]:
        """Collect per-sample token conditioning; ``cfg_dropout_prob`` swaps in the unconditional sequence."""
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"HunyuanImage3Adapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")
        conditioning = context.batch.get("conditioning")
        if conditioning is None or len(conditioning) != noisy_latents.shape[0]:
            raise ValueError(
                "HunyuanImage-3.0 batches need one `conditioning` dict per sample from the preprocessing cache"
            )

        samples = []
        for cond in conditioning:
            prefix = "uncond_" if random.random() < context.cfg_dropout_prob else ""
            samples.append(
                {
                    key: cond[prefix + key]
                    for key in ("input_ids", "image_mask", "timestep_scatter_index", "image_start")
                }
            )
            samples[-1]["token_hw"] = (int(cond["token_h"]), int(cond["token_w"]))
        return {
            "noisy_latents": noisy_latents,
            # The release embeds timesteps on the [0, 1000] sigma scale used by the pipeline.
            "timesteps": context.timesteps.float(),
            "samples": samples,
            "device": context.device,
        }

    def forward(self, model: nn.Module, inputs: dict[str, object]) -> torch.Tensor:
        """Return the velocity prediction ``[B, C, H, W]``."""
        device = inputs["device"]
        preds = []
        for i, sample in enumerate(inputs["samples"]):
            input_ids = sample["input_ids"].to(device=device, dtype=torch.long)[None]
            seq_len = input_ids.shape[1]
            token_h, token_w = sample["token_hw"]
            latents = inputs["noisy_latents"][i : i + 1]
            if tuple(latents.shape[-2:]) != (token_h, token_w):
                raise ValueError(
                    f"Latent size {tuple(latents.shape[-2:])} does not match the cached image block {(token_h, token_w)}"
                )
            image_start = int(sample["image_start"])
            positions = build_2d_positions(seq_len, [(image_start, token_h, token_w)])[None].to(device)
            out = model(
                input_ids,
                rope_positions=positions,
                attention_mask=text_causal_image_bidirectional_mask(seq_len, image_start, token_h * token_w, device),
                mode="gen_image",
                images=latents,
                timestep=inputs["timesteps"][i : i + 1],
                image_mask=sample["image_mask"].to(device=device, dtype=torch.bool)[None],
                timestep_scatter_index=sample["timestep_scatter_index"].to(device=device, dtype=torch.long)[None],
            )
            preds.append(out.diffusion_prediction)
        return torch.cat(preds).to(inputs["noisy_latents"].dtype)
