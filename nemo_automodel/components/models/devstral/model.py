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

"""FP8-native Mistral3 / Ministral3 custom model.

Covers three checkpoint variants, all of which ship as FP8 safetensors with
per-tensor scalar `weight_scale_inv`:

  * mistralai/Devstral-Small-2-24B-Instruct-2512 — VLM wrapper
    (``Mistral3ForConditionalGeneration``) with ``language_model.`` prefix.
  * mistralai/Devstral-2-123B-Instruct-2512 — dense text-only
    (``Ministral3ForCausalLM``), no prefix.
  * dawn-ridge-medium-3p5-128b (codenamed Mistral-3.5 128B) — VLM wrapper
    with ``model.language_model.`` infix.

HF's ``FineGrainedFP8Config`` loader materializes the full dequantized BF16
model on every rank before PP split, OOM-ing under TP+PP on 80 GB H100. We
avoid it by (a) registering a resolver hook that routes these configs to
this class and (b) attaching a ``Mistral3FP8StateDictAdapter`` that runs
FP8 dequant inside the standard Checkpointer DCP load path.

Layout detection is automatic: at ``__init__`` we peek at the checkpoint's
``model.safetensors.index.json`` to tell the three variants apart by their
top-level key prefix. Fallbacks choose the safest layout that still loads.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

import torch
from transformers import PretrainedConfig

from nemo_automodel.components.models.devstral.state_dict_adapter import (
    Mistral3FP8StateDictAdapter,
)
from nemo_automodel.components.models.mistral3.model import (
    Ministral3Config,
    Ministral3ForCausalLM,
)

logger = logging.getLogger(__name__)


def _resolve_snapshot_dir(name_or_path: str) -> Optional[str]:
    """Return a local dir containing ``model.safetensors.index.json`` if findable.

    Accepts either an absolute path or an HF repo id. For repo ids, scans the
    HF hub cache for a matching snapshot; returns None if nothing looks right.
    """
    if not name_or_path:
        return None
    if os.path.isdir(name_or_path):
        return name_or_path if os.path.exists(
            os.path.join(name_or_path, "model.safetensors.index.json")
        ) else None
    # HF repo id: scan cache. Try the env-exported HF_HOME first.
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    roots = [os.path.join(hf_home, "hub")] if hf_home else []
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    safe = name_or_path.replace("/", "--")
    for root in roots:
        for snap in glob.glob(os.path.join(root, f"models--{safe}", "snapshots", "*")):
            if os.path.exists(os.path.join(snap, "model.safetensors.index.json")):
                return snap
    return None


def _detect_layout(name_or_path: str) -> str:
    """Sample the safetensors weight map to choose the adapter layout.

    Returns one of ``"devstral_vlm"``, ``"dawn_ridge_vlm"``, ``"dense"``.
    If the checkpoint can't be located we default to ``"dense"`` — this is
    the safest fallback for ``Ministral3ForCausalLM`` archs (no remapping).
    """
    snap = _resolve_snapshot_dir(name_or_path)
    if snap is None:
        logger.info(
            "Mistral3FP8: could not resolve snapshot dir for %r — defaulting "
            "to 'dense' layout (no key remapping)",
            name_or_path,
        )
        return "dense"
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    # Sample some representative keys.
    sample = []
    for k in weight_map:
        if ".layers.0." in k or k.endswith("embed_tokens.weight"):
            sample.append(k)
            if len(sample) >= 4:
                break
    if not sample:
        # Fall back to the first 4 keys.
        sample = list(weight_map.keys())[:4]
    # Decide.
    if any(k.startswith("language_model.") for k in sample):
        layout = "devstral_vlm"
    elif any(k.startswith("model.language_model.") for k in sample):
        layout = "dawn_ridge_vlm"
    else:
        layout = "dense"
    logger.info(
        "Mistral3FP8: detected checkpoint layout=%r from %s (sample=%s)",
        layout,
        snap,
        sample[:2],
    )
    return layout


class Mistral3FP8ForCausalLM(Ministral3ForCausalLM):
    """Unified FP8 loader for the Mistral3 / Ministral3 family.

    Accepts:
      - ``Mistral3Config`` (VLM — `text_config` is extracted and coerced to
        a ``Ministral3Config``), or
      - ``Ministral3Config`` directly (dense text-only).

    In either case the outer ``quantization_config`` is preserved so that
    ``apply_model_infrastructure`` (infrastructure.py:419-422) sets
    ``dequantize_base_checkpoint=True`` on the Checkpointer — that's the
    signal for ``_maybe_adapt_state_dict_to_hf`` to call ``adapter.to_hf``
    with ``quantization=True`` and emit the ``_scale_inv`` placeholder keys
    that DCP reads from disk alongside the FP8 weights.
    """

    # See checkpointing.py:initialize_model_weights — gate on this attribute
    # to skip HF's `initialize_weights()` (the upcoming adapter load will
    # populate every tensor anyway, and skipping avoids a stage-divergent
    # DTensor collective inside HF's init on PP setups).
    _skip_init_weights_on_load = True

    def __init__(self, config: PretrainedConfig):
        ministral_cfg, orig_name_or_path = self._coerce_to_ministral_config(config)
        super().__init__(ministral_cfg)
        layout = _detect_layout(orig_name_or_path)
        if layout == "devstral_vlm":
            self.state_dict_adapter = Mistral3FP8StateDictAdapter.for_devstral_vlm()
        elif layout == "dawn_ridge_vlm":
            self.state_dict_adapter = Mistral3FP8StateDictAdapter.for_dawn_ridge_vlm()
        else:
            self.state_dict_adapter = Mistral3FP8StateDictAdapter.for_dense()

    @staticmethod
    def _coerce_to_ministral_config(config: PretrainedConfig) -> tuple[Ministral3Config, str]:
        """Return ``(Ministral3Config, name_or_path)``.

        If ``config`` is a VLM wrapper (has ``text_config``), unwrap it and
        carry the outer ``quantization_config`` across (Ministral3Config would
        otherwise drop it — it's not part of the ministral3 schema, but
        ``apply_model_infrastructure`` reads ``hasattr(config,
        'quantization_config')`` as a signal).
        """
        name_or_path = getattr(config, "name_or_path", "") or getattr(
            config, "_name_or_path", ""
        )
        if isinstance(config, Ministral3Config):
            return config, name_or_path
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            if getattr(config, "model_type", None) != "ministral3":
                raise TypeError(
                    f"{Mistral3FP8ForCausalLM.__name__} expects a Ministral3Config "
                    f"or a VLM config with .text_config, got {type(config).__name__}."
                )
            outer_qc = getattr(config, "quantization_config", None)
            new_cfg = Ministral3Config(**config.to_dict())
            if outer_qc is not None:
                new_cfg.quantization_config = outer_qc
            new_cfg.name_or_path = name_or_path
            return new_cfg, name_or_path
        # VLM: unwrap text_config, preserve quant_config on new cfg.
        new_cfg = Ministral3Config(**text_config.to_dict())
        new_cfg.name_or_path = name_or_path
        outer_qc = getattr(config, "quantization_config", None)
        if outer_qc is not None:
            new_cfg.quantization_config = outer_qc
        return new_cfg, name_or_path

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Claim any FP8-native config whose (inner or outer) model_type is ministral3."""
        # VLM wrapper with a ministral3 text backbone.
        text_config = getattr(config, "text_config", None)
        is_family = (
            (text_config is not None and getattr(text_config, "model_type", None) == "ministral3")
            or getattr(config, "model_type", None) == "ministral3"
        )
        if not is_family:
            return False
        qc = getattr(config, "quantization_config", None)
        if qc is None:
            return False
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        return method == "fp8"


# Backwards-compat aliases — older code referenced size-specific names.
Devstral24BFP8TextForCausalLM = Mistral3FP8ForCausalLM
Devstral123BFP8ForCausalLM = Mistral3FP8ForCausalLM
