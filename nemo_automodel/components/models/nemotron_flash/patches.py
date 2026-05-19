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

"""Runtime RoPE patches for legacy Nemotron Flash remote-code models."""

import logging
import types

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def _to_local(t):
    """Unwrap DTensor to its local shard for numeric checks."""
    return t._local_tensor if isinstance(t, DTensor) else t


@torch.no_grad()
def _safe_rope_forward(self, x, position_ids, **kwargs):
    """Drop-in replacement matching Nemotron-Flash-1B's native rotary forward."""
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _compute_flash_inv_freq(cfg, device, dim):
    """Compute ``inv_freq`` using Nemotron-Flash-1B's own NTK/default formula."""
    base = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    rope_type = getattr(cfg, "rope_type", None) or "default"
    if rope_type == "ntk":
        max_pos = getattr(cfg, "max_position_embeddings", None)
        orig_max = getattr(cfg, "orig_max_position_embeddings", None)
        if max_pos is not None and orig_max is not None and orig_max > 0:
            factor = 2
            base = base * ((factor * max_pos / orig_max) - (factor - 1)) ** (dim / (dim - 2))
    indices = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
    return 1.0 / (base ** (indices / dim))


def _is_nemotron_flash_config(cfg) -> bool:
    """Return True when *cfg* identifies a Nemotron Flash remote-code model."""
    if cfg is None:
        return False

    model_type = getattr(cfg, "model_type", None)
    if model_type == "nemotron_flash":
        return True

    architectures = getattr(cfg, "architectures", None) or ()
    if "NemotronFlashForCausalLM" in architectures:
        return True

    name_or_path = getattr(cfg, "name_or_path", "") or ""
    return "nemotron-flash" in name_or_path.lower()


def should_fix_rotary_embeddings(model_parts: list[object]) -> bool:
    """Return True when the legacy rotary workaround should run."""
    for mp in model_parts:
        if isinstance(mp, nn.Module):
            for _, module in mp.named_modules():
                if _is_nemotron_flash_config(getattr(module, "config", None)):
                    return True
        elif _is_nemotron_flash_config(getattr(mp, "config", None)):
            return True

    return False


def fix_rotary_embeddings(model_parts: list[object]) -> int:
    """Install Nemotron-Flash-1B's native NTK ``inv_freq`` deterministically."""
    fixed = 0
    for mp in model_parts:
        for fqn, module in mp.named_modules():
            inv = getattr(module, "inv_freq", None)
            if inv is None or not isinstance(inv, torch.Tensor):
                continue

            cfg = getattr(module, "config", None)
            iv = _to_local(inv)
            dim = iv.shape[-1] * 2
            new_inv = _compute_flash_inv_freq(cfg, iv.device, dim)

            inv.data.copy_(new_inv.to(dtype=inv.dtype, device=inv.device))
            orig = getattr(module, "original_inv_freq", None)
            if orig is not None:
                orig.data.copy_(new_inv.to(dtype=orig.dtype, device=orig.device))

            module.forward = types.MethodType(_safe_rope_forward, module)

            rope_type = getattr(cfg, "rope_type", None) or "default"
            logger.info(f"[fix_rope] {fqn}: installed Flash NTK inv_freq (rope_type={rope_type}, dim={dim})")
            fixed += 1

    logger.info(f"[fix_rope] repaired {fixed} rotary embeddings.")
    return fixed


def should_apply_post_shard_patches(model_parts: list[object]) -> bool:
    """Return True when post-shard compatibility patches should run."""
    return should_fix_rotary_embeddings(model_parts)


def apply_post_shard_patches(model_parts: list[object]) -> int:
    """Apply post-shard compatibility patches for Nemotron Flash models."""
    return fix_rotary_embeddings(model_parts)


__all__ = [
    "_is_nemotron_flash_config",
    "apply_post_shard_patches",
    "fix_rotary_embeddings",
    "should_apply_post_shard_patches",
    "should_fix_rotary_embeddings",
]
