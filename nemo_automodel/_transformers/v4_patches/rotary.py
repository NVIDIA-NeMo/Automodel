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
    """Drop-in replacement matching Nemotron-Flash-1B's native rotary forward.

    Mirrors ``modeling_nemotron_flash.LlamaRotaryEmbedding.forward`` verbatim
    (incl. ``@torch.no_grad`` + autocast disable for FP32 precision) so that
    running this patched forward is semantically identical to letting Flash's
    native forward run with the same ``inv_freq``.
    """
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
    """Compute ``inv_freq`` using Nemotron-Flash-1B's own NTK/default formula.

    Copy of the relevant init branch from
    ``modeling_nemotron_flash.LlamaRotaryEmbedding.__init__``. Flash's NTK
    differs from transformers' standard:
    - ``factor = 2`` (hardcoded in Flash)
    - Reads ``config.orig_max_position_embeddings`` (not
      ``original_max_position_embeddings``).
    - Scales ``base`` directly (no post-hoc ``attention_scaling``).
    """
    base = float(getattr(cfg, "rope_theta", 10000.0) or 10000.0)
    rope_type = getattr(cfg, "rope_type", None) or "default"
    if rope_type == "ntk":
        max_pos = getattr(cfg, "max_position_embeddings", None)
        orig_max = getattr(cfg, "orig_max_position_embeddings", None)
        if max_pos is not None and orig_max is not None and orig_max > 0:
            factor = 2
            base = base * (
                (factor * max_pos / orig_max) - (factor - 1)
            ) ** (dim / (dim - 2))
    # default / dynamic_ntk / unknown: use (possibly unscaled) ``base``.
    indices = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
    return 1.0 / (base ** (indices / dim))


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
    """Install Nemotron-Flash-1B's native NTK ``inv_freq`` deterministically.

    Flash's own ``LlamaRotaryEmbedding.__init__`` (remote code, under
    trust_remote_code) can land with NaN/Inf ``inv_freq`` buffers under
    transformers 5.x's meta-device init context, and its NTK formula is
    non-standard (``factor=2``, reads ``config.orig_max_position_embeddings``,
    no post-hoc ``attention_scaling``), so transformers' own
    ``ROPE_INIT_FUNCTIONS`` does not match it. The old version of this patch
    sidestepped that by overwriting ``inv_freq`` with a plain-vanilla formula
    (no NTK) and replacing ``forward`` with a vanilla one — but that silently
    downgraded training-time rope semantics relative to Flash's native, which
    vanilla HF uses when reloading the consolidated checkpoint. The result was
    Phase 4 HF KL > 1.0, "fixed" by skipping Phase 4.

    This revised patch computes ``inv_freq`` using Flash's *own* NTK formula
    (copied verbatim from ``modeling_nemotron_flash.LlamaRotaryEmbedding``) and
    installs it on every Flash rotary found, unconditionally. The forward is
    also replaced with ``_safe_rope_forward`` (now semantically identical to
    Flash's native forward), which guards against any init-order oddity in
    the remote-code class. Training, Phase 3 Automodel reload, and Phase 4
    vanilla HF reload all end up computing the same NTK-scaled rope.

    Scope: only touches modules whose ``config`` is recognized as Nemotron-
    Flash (via ``_is_nemotron_flash_config``), so non-Flash models are never
    affected. ``should_fix_rotary_embeddings`` further narrows the call site.
    """
    fixed = 0
    for mp in model_parts:
        for fqn, module in mp.named_modules():
            inv = getattr(module, "inv_freq", None)
            if inv is None or not isinstance(inv, torch.Tensor):
                continue

            cfg = getattr(module, "config", None)
            iv = _to_local(inv)
            # ``inv_freq`` has ``dim/2`` elements → dim = 2 * iv.shape[-1].
            # Verified against ``LlamaRotaryEmbedding.__init__`` which uses
            # ``torch.arange(0, dim, 2)`` of length ``dim/2``.
            dim = iv.shape[-1] * 2
            new_inv = _compute_flash_inv_freq(cfg, iv.device, dim)

            inv.data.copy_(new_inv.to(dtype=inv.dtype, device=inv.device))
            orig = getattr(module, "original_inv_freq", None)
            if orig is not None:
                orig.data.copy_(new_inv.to(dtype=orig.dtype, device=orig.device))

            module.forward = types.MethodType(_safe_rope_forward, module)

            rope_type = getattr(cfg, "rope_type", None) or "default"
            logger.info(
                f"[fix_rope] {fqn}: installed Flash NTK inv_freq (rope_type={rope_type}, dim={dim})"
            )
            fixed += 1

    logger.info(f"[fix_rope] repaired {fixed} rotary embeddings.")
    return fixed


__all__ = ["fix_rotary_embeddings", "should_fix_rotary_embeddings"]
