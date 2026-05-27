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

"""Utilities for dLLM generation.

Contains:
- Sampling utilities (Gumbel noise, transfer schedule, transfer index)
- Model loading helpers (checkpoint resolution, tokenizer setup, compat patches)
- Response trimming
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Automodel utilities — use the installed/editable nemo_automodel package.
# If running from a standalone checkout, set AUTOMODEL_ROOT to the Automodel repo root.
_automodel_root = os.environ.get("AUTOMODEL_ROOT", "/opt/Automodel")
if _automodel_root not in sys.path:
    sys.path.insert(0, _automodel_root)
from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoTokenizer

# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling noise in float64 for numerical stability."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Linear transfer schedule: spread unmasking evenly across steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select which masked positions to unmask based on confidence.

    When ``threshold`` is set, ``num_transfer_tokens`` is overridden to all
    masked positions. Top-1 is always unmasked; positions 2+ only if confidence
    exceeds the threshold.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_checkpoint(path: str) -> str:
    """Resolve a checkpoint path, checking for a ``consolidated/`` subdirectory.

    Accepts any of:
      - ``.../consolidated`` (direct HF-format dir)
      - ``.../model`` (finds ``consolidated/`` inside)
      - ``.../LATEST`` (finds ``model/consolidated/`` inside)
      - ``.../epoch_0_step_312/model/consolidated`` (intermediate steps)
    """
    if os.path.isdir(os.path.join(path, "consolidated")):
        return os.path.join(path, "consolidated")
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    for sub in [
        "LATEST/model/consolidated",
        "LATEST/model",
        "model/consolidated",
        "model",
    ]:
        candidate = os.path.join(path, sub)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(
        f"Could not find a valid HF checkpoint at {path}. Expected a directory with config.json and model safetensors."
    )


def load_model_and_tokenizer(checkpoint_path: str, sampler_name: str = "llada"):
    """Load model and tokenizer from an Automodel checkpoint.

    Args:
        checkpoint_path: Path to the HF-format checkpoint directory.
        sampler_name: ``"llada"`` or ``"nemotron"``. Adjusts tokenizer setup and
            model construction kwargs for the chosen family.

    Returns:
        ``(model, tokenizer, mask_id, eos_id)``.
    """
    from nemo_automodel._transformers.auto_model import _patch_remote_code_compat

    _patch_remote_code_compat()

    tokenizer = NeMoAutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if sampler_name == "llada":
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})

    model_kwargs = dict(
        torch_dtype="bfloat16",
        trust_remote_code=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        attn_implementation="eager",
    )
    if sampler_name == "nemotron":
        # Inference mode for Nemotron-Labs-Diffusion is "bidirectional" (simulates
        # block-wise attention).  SFT may have saved the checkpoint with
        # dlm_paradigm=block_diff baked into config.json; force back to
        # bidirectional for generation.
        model_kwargs["dlm_paradigm"] = "bidirectional"
        model_kwargs["block_size"] = 32

    model = NeMoAutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs).eval()

    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)

    eos_id = tokenizer.eos_token_id
    return model, tokenizer, mask_id, eos_id


# ---------------------------------------------------------------------------
# Response trimming
# ---------------------------------------------------------------------------


def trim_response(tokenizer, seq_ids_list, input_ids_list):
    """Extract generated text after the prompt, truncated at first EOS / EOT."""
    results = []
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    eot_id = getattr(tokenizer, "eot_token_id", None)
    if eot_id is not None:
        stop_ids.add(eot_id)
    for token_str in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        start = len(list(input_ids))
        end = len(full)
        for i in range(start, len(full)):
            if full[i] in stop_ids:
                end = i
                break
        text = tokenizer.decode(full[start:end], skip_special_tokens=True)
        for stop_str in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
            if stop_str in text:
                text = text.split(stop_str)[0]
        results.append(text)
    return results
