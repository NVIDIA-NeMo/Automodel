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
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration as _HFMistral3ForConditionalGeneration,
)

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


def _rotary_reinit_self_hook(module, args, kwargs):
    """One-shot forward pre-hook that recomputes *this rotary module's* own
    ``inv_freq`` on first call.

    Attached per-rotary rather than on the outer VLM so it fires correctly
    under pipeline parallelism, where the outer model's ``forward`` is never
    called directly — the PP schedule dispatches each stage's sub-modules
    individually, and rotary modules run inside every attention layer.

    Background: HF's Ministral3 / Pixtral rotary classes initialise
    ``inv_freq`` (and related attributes) in their ``__init__``. Under
    ``accelerate.init_empty_weights`` that becomes a meta tensor, and the
    subsequent ``to_empty(device)`` call leaves it uninitialised device
    memory. Neither class exposes ``rope_init_fn`` as an attribute, so the
    generic ``_reinit_non_persistent_buffers`` helper doesn't match. We
    recover correctness by re-running the module's own ``__init__`` on the
    target device outside the init_empty_weights context — both Ministral3
    (YaRN) and Pixtral (2D patch positions) produce the right values this
    way, since we defer to the class's authoritative init logic.
    """
    if getattr(module, "_mistral3_fp8_rotary_reinit_done", False):
        return
    # Pick a device that has real storage. Prefer a buffer with non-meta
    # device (rotary modules typically have `inv_freq` buffer only).
    device = None
    for buf in module.buffers(recurse=False):
        if buf.device.type != "meta":
            device = buf.device
            break
    if device is None:
        # Fallback: whatever pytorch's current device default is.
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    try:
        type(module).__init__(module, module.config, device=device)
    except TypeError:
        try:
            type(module).__init__(module, module.config)
        except Exception:
            module._mistral3_fp8_rotary_reinit_done = True
            return
    except Exception:
        module._mistral3_fp8_rotary_reinit_done = True
        return
    # Ensure every tensor the re-init created ends up on the right device.
    # PixtralRotaryEmbedding precomputes per-patch position caches beyond
    # just `inv_freq`, so move the whole module.
    module.to(device=device)
    module._mistral3_fp8_rotary_reinit_done = True


class Mistral3FP8VLMForConditionalGeneration(_HFMistral3ForConditionalGeneration):
    """Full-VLM (vision + text) FP8 loader for Mistral3ForConditionalGeneration.

    Used when the user instantiates through
    ``NeMoAutoModelForImageTextToText.from_pretrained`` on an FP8-native
    Mistral3 VLM checkpoint (e.g. dawn-ridge-128B). We want to keep the
    vision_tower + multi_modal_projector modules, so we extend HF's VLM
    class rather than the text-only ``Ministral3ForCausalLM``.

    Responsibilities:
      * Inherit the full VLM architecture from
        ``transformers.models.mistral3.modeling_mistral3.Mistral3ForConditionalGeneration``
        so ``.vision_tower``, ``.multi_modal_projector``, and the language
        model are all instantiated.
      * Attach ``Mistral3FP8StateDictAdapter.for_vlm_full()`` — identity key
        rewrites, FP8 dequant restricted to language_model layer weights
        (vision / mm_projector / lm_head are BF16 on disk and pass through).
      * Opt into ``_skip_init_weights_on_load = True`` so the DCP load path
        replaces every tensor and HF's ``initialize_weights`` (which can
        issue stage-divergent DTensor collectives under PP) is skipped.
    """

    _skip_init_weights_on_load = True

    def __init__(self, config: PretrainedConfig):
        # HF's Mistral3ForConditionalGeneration.__init__ consults
        # ``config.quantization_config`` and swaps nn.Linear → FP8Linear for
        # every language_model Linear. FP8Linear registers a 0-d
        # ``weight_scale_inv`` parameter — which FSDP2's ``fully_shard``
        # rejects with "doesn't support scalar parameters". We handle FP8
        # dequant ourselves via the state_dict adapter, so tell HF to skip
        # the linear swap by flipping ``dequantize=True`` on the config
        # (see transformers/integrations/finegrained_fp8.py:742, which early-
        # returns from replace_with_fp8_linear when dequantize is truthy).
        # ``apply_model_infrastructure`` still sees hasattr(config,
        # 'quantization_config'), so ``dequantize_base_checkpoint`` is set
        # to True and our adapter's ``to_hf(quantization=True)`` path runs.
        qc = getattr(config, "quantization_config", None)
        if qc is not None:
            if isinstance(qc, dict):
                qc["dequantize"] = True
            else:
                try:
                    qc.dequantize = True
                except AttributeError:
                    pass
        super().__init__(config)
        self.state_dict_adapter = Mistral3FP8StateDictAdapter.for_vlm_full()

        # Lazy non-persistent buffer reinit. HF's Ministral3RotaryEmbedding /
        # PixtralRotaryEmbedding compute `inv_freq` in their __init__. Under
        # the accelerate ``init_empty_weights`` context (used by the NeMo Auto
        # meta-init path), the computed buffer is converted to meta; and when
        # the checkpointer later calls ``to_empty(device)`` it becomes
        # uninitialized real memory — garbage rotary positional embeddings.
        # ``_reinit_non_persistent_buffers`` in checkpointing.py has Pattern-1
        # logic that expects a ``rope_init_fn`` attribute, which HF's rotary
        # classes do not expose. Rather than touch the generic checkpointer,
        # we attach a one-shot forward pre-hook to every rotary submodule so
        # that it recomputes itself the first time it is called. Per-rotary
        # registration (as opposed to a single outer hook) is required so the
        # reinit fires under PP, where the outer VLM's forward is never
        # invoked directly — the PP schedule dispatches stage sub-modules
        # individually and rotary runs inside each attention layer.
        for sub in self.modules():
            if "inv_freq" in getattr(sub, "_buffers", {}):
                sub._mistral3_fp8_rotary_reinit_done = False
                sub.register_forward_pre_hook(
                    _rotary_reinit_self_hook, with_kwargs=True, prepend=True
                )

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Claim FP8-native Mistral3 VLM configs.

        Matches ``Mistral3Config`` (outer VLM) with a ministral3 text backbone
        and ``quantization_config.quant_method == 'fp8'``. The routing logic
        in ``__init__.py`` only calls this hook when the caller is an
        ``ImageTextToText``-style entry point, so text-only flows keep using
        ``Mistral3FP8ForCausalLM``.
        """
        text_config = getattr(config, "text_config", None)
        if text_config is None or getattr(text_config, "model_type", None) != "ministral3":
            return False
        qc = getattr(config, "quantization_config", None)
        if qc is None:
            return False
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        return method == "fp8"
