# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Runtime RoPE patches for legacy v4-style remote-code models."""

import types

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def _to_local(t):
    """Unwrap DTensor to its local shard for numeric checks."""
    return t._local_tensor if isinstance(t, DTensor) else t


def _safe_rope_forward(self, x, position_ids, **kwargs):
    """Drop-in replacement for legacy rotary embedding forward methods."""
    inv_freq = self.inv_freq.float()
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _is_nemotron_flash_config(cfg):
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


def should_fix_rotary_embeddings(model_parts):
    """Return True when the legacy rotary workaround should run."""
    for mp in model_parts:
        if isinstance(mp, nn.Module):
            for _, module in mp.named_modules():
                if _is_nemotron_flash_config(getattr(module, "config", None)):
                    return True
        elif _is_nemotron_flash_config(getattr(mp, "config", None)):
            return True

    return False


def fix_rotary_embeddings(model_parts):
    """Patch rotary embeddings to bypass fragile legacy HF runtime behavior."""
    fixed = 0
    for mp in model_parts:
        for fqn, module in mp.named_modules():
            inv = getattr(module, "inv_freq", None)
            if inv is None or not isinstance(inv, torch.Tensor):
                continue

            iv = _to_local(inv)
            bad = bool(torch.isnan(iv).any().item()) or bool(torch.isinf(iv).any().item())

            cfg = getattr(module, "config", None)
            if cfg is not None:
                rope_params = getattr(cfg, "rope_parameters", {}) or {}
                base = rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
                dim = getattr(cfg, "head_dim", None)
                if dim is None:
                    hs = getattr(cfg, "hidden_size", None)
                    nah = getattr(cfg, "num_attention_heads", None)
                    if hs and nah:
                        dim = hs // nah
                if dim is None:
                    dim = iv.shape[-1] * 2
            else:
                base = 10000.0
                dim = iv.shape[-1] * 2

            new_inv = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=iv.device, dtype=torch.float32) / dim)
            )

            inv.data.copy_(new_inv.to(dtype=inv.dtype, device=inv.device))
            orig = getattr(module, "original_inv_freq", None)
            if orig is not None:
                orig.data.copy_(new_inv.to(dtype=orig.dtype, device=orig.device))

            module.forward = types.MethodType(_safe_rope_forward, module)

            print(
                f"[fix_rope] {fqn}: patched forward + inv_freq (bad={bad} base={base} dim={dim})",
                flush=True,
            )
            fixed += 1

    print(f"[fix_rope] patched {fixed} rotary embeddings.", flush=True)
    return fixed


__all__ = ["fix_rotary_embeddings", "should_fix_rotary_embeddings"]
