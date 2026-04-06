# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoRA configuration and per-model target module defaults.

No pipeline class registry — pure PEFT approach means we only
need to know which modules to target, not which pipeline class to use.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.

    Args:
        enabled:        Whether LoRA is active. When False, full fine-tune.
        rank:           Rank of LoRA update matrices (r).
        alpha:          LoRA scaling factor. Effective scale = alpha / rank.
        dropout:        Dropout probability applied to LoRA layers.
        target_modules: List of module name substrings to target.
                        None = use per-model defaults from
                        MODEL_DEFAULT_TARGET_MODULES.
    """

    enabled: bool = False
    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    target_modules: Optional[list] = None


# ── Flux ──────────────────────────────────────────────────────────────────────
# Verified against train_dreambooth_flux.py (diffusers examples).
# FluxTransformer2DModel has two block types:
#   DoubleStreamBlock: joint self-attention for latent + text streams
#     → attn.to_q/k/v/out (latent stream)
#     → attn.add_q/k/v_proj, attn.to_add_out (text stream)
#     → ff + ff_context (feedforward for both streams)
#   SingleStreamBlock: self-attention for merged stream only
#     → attn.to_q/k/v/out only (no add projections)
# FF layers included for richer style/content adaptation.
FLUX_DEFAULT_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

# ── Wan2.1 ────────────────────────────────────────────────────────────────────
# Verified against WanAttention in diffusers Wan transformer.
# WanTransformerBlock has two attention layers per block:
#   attn1: self-attention  → to_q, to_k, to_v, to_out[0]
#   attn2: cross-attention → to_q, to_k, to_v, to_out[0]
#         (I2V also has add_k_proj, add_v_proj for image conditioning)
# IMPORTANT: WanAttention fuses to_q/k/v → to_qkv after loading for speed.
# PEFT cannot find individual projections if fused.
# _wan_pre_inject hook in setup.py calls unfuse_projections() before injection.
WAN_DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]

# ── HunyuanVideo15 ────────────────────────────────────────────────────────────
# Verified against HunyuanVideo15AttnProcessor2_0 and
# HunyuanVideo15TransformerBlock in diffusers.
# HunyuanVideo15TransformerBlock uses diffusers Attention class (no fused proj).
# Attention processor accesses:
#   Latent stream:  attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]
#   Encoder stream: attn.add_q_proj, attn.add_k_proj, attn.add_v_proj,
#                   attn.to_add_out
# No pre-inject hook needed — standard diffusers Attention, no fusing.
# Note: also matches attn in HunyuanVideo15IndividualTokenRefinerBlock
# (to_q/k/v only, no add projections) — intentional, refiner benefits from LoRA.
HUNYUAN_DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
]

# Registry used by inject_lora() in setup.py
MODEL_DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "flux": FLUX_DEFAULT_TARGET_MODULES,
    "wan": WAN_DEFAULT_TARGET_MODULES,
    "hunyuan": HUNYUAN_DEFAULT_TARGET_MODULES,
}
