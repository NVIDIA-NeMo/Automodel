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

"""State-dict adapter for Devstral FP8 checkpoints.

Two responsibilities, both handled inside the nemo_automodel checkpoint flow
(nemo_automodel/components/checkpoint/checkpointing.py ~lines 510, 556):

  1. **Key remapping** between the on-disk HF layout and our custom
     Ministral3ForCausalLM layout.
       - Devstral-Small-2-24B-Instruct-2512 ships as a VLM
         (`Mistral3ForConditionalGeneration`) with all text keys prefixed
         `language_model.` — we train only the text backbone, so model keys
         (`model.layers.*`, `lm_head.*`) map to `language_model.*` on disk.
       - Devstral-2-123B-Instruct-2512 is a dense `Ministral3ForCausalLM`
         with no prefix; the adapter is a pass-through on keys.

  2. **FP8 dequantization**. Both checkpoints carry per-Linear
     `weight_scale_inv` (scalar bf16) and `activation_scale` (unused for
     training) siblings; the adapter pairs them with their weight, dequantizes
     to bf16 (`w_bf16 = w_fp8.to(bf16) * scale`), and drops the scale keys from
     the state dict.

Structurally modelled after
`nemo_automodel/components/models/deepseek_v3/state_dict_adapter.py` — the
`to_hf` path adds `_scale_inv` placeholder entries so the DCP storage reader
actually fetches them from safetensors; `from_hf` consumes those entries to
produce bf16 weights for the model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


# Keys that should NOT be treated as FP8 weights — i.e. we don't pair them
# with a `_scale_inv` on save and we don't try to dequantize them on load.
# Mirrors `modules_to_not_convert` in the Devstral HF config:
#     ['model.vision_tower', 'model.multi_modal_projector', 'lm_head']
# and, for all Devstral variants, layernorms + embeddings.
_NON_QUANTIZED_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "model.norm.weight",
)


def _is_fp8_weight_key(model_key: str) -> bool:
    """Return True iff `model_key` names an FP8 Linear weight in Devstral."""
    if not model_key.endswith(".weight"):
        return False
    return not any(model_key.endswith(suffix) for suffix in _NON_QUANTIZED_SUFFIXES)


def _dequantize_from_fp8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a single FP8 weight using its per-tensor scalar scale.

    Devstral (both 24B and 123B) uses per-tensor quantization
    (`quantization_config.weight_block_size: None`) so `scale_inv` is a 0-d
    scalar and dequant collapses to a simple multiply. The per-block formula
    (`transformers.integrations.finegrained_fp8.Fp8Dequantize.convert`,
    finegrained_fp8.py:867-906) is not needed for this model family.
    """
    return weight_fp8.to(target_dtype) * scale_inv.to(target_dtype)


class DevstralFP8StateDictAdapter(StateDictAdapter):
    """Handles key remapping and FP8 → bf16 dequant for Devstral checkpoints.

    Args:
        key_prefix: Prefix to add (on save) / strip (on load) between HF keys
            and model keys. Use ``"language_model."`` for the 24B VLM
            checkpoint, ``""`` for the 123B dense checkpoint.
    """

    def __init__(self, key_prefix: str = ""):
        self.key_prefix = key_prefix

    # --------------------------------------------------------------------- #
    # model → HF                                                            #
    # --------------------------------------------------------------------- #
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a model-native state dict to HF (on-disk) layout.

        When ``quantization=True`` (i.e. the checkpointer is preparing the
        target state dict before reading from an FP8 HF checkpoint on disk),
        a zero-element placeholder is added for every FP8 weight's
        ``_scale_inv`` key. The DCP storage reader populates these placeholders
        from disk alongside the weights. ``from_hf`` later consumes the
        loaded scales to dequantize.
        """
        hf: dict[str, Any] = {}
        for model_key, value in state_dict.items():
            if exclude_key_regex is not None:
                import re

                if re.match(exclude_key_regex, model_key):
                    continue
            hf_key = f"{self.key_prefix}{model_key}" if self.key_prefix else model_key
            if quantization and _is_fp8_weight_key(model_key):
                # Must cast the weight placeholder to FP8 so the DCP storage
                # reader fetches FP8 bytes verbatim from safetensors instead
                # of casting on read (which would apply no scale and produce
                # garbage). Mirrors deepseek_v3/state_dict_adapter.py:220.
                value = value.to(dtype=torch.float8_e4m3fn)
                scale_placeholder = torch.empty((), dtype=torch.bfloat16)
                hf[hf_key] = value
                hf[hf_key + "_scale_inv"] = scale_placeholder
            else:
                hf[hf_key] = value
        return hf

    # --------------------------------------------------------------------- #
    # HF → model                                                            #
    # --------------------------------------------------------------------- #
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert an HF-format (possibly FP8) state dict to model-native format.

        - FP8 weights are paired with their ``_scale_inv`` sibling and
          dequantized to bf16 on whatever device the weight already lives on.
        - ``activation_scale`` keys are dropped (unused for training).
        - Keys are remapped by stripping ``self.key_prefix``.
        """
        native: dict[str, Any] = {}
        dequantized = 0
        dropped_scales = 0
        dropped_act_scales = 0

        # First pass: collect scale_inv keys so we can look them up O(1).
        scale_map = {
            k[: -len("_scale_inv")]: v
            for k, v in hf_state_dict.items()
            if k.endswith("_scale_inv")
        }

        for hf_key, value in hf_state_dict.items():
            if hf_key.endswith("_scale_inv"):
                dropped_scales += 1
                continue
            if hf_key.endswith(".activation_scale"):
                dropped_act_scales += 1
                continue

            # Strip prefix for the model-side name.
            if self.key_prefix and hf_key.startswith(self.key_prefix):
                model_key = hf_key[len(self.key_prefix):]
            else:
                model_key = hf_key

            # Dequantize if this is an FP8 weight with a paired scale.
            if value.dtype == torch.float8_e4m3fn and hf_key in scale_map:
                scale = scale_map[hf_key]
                value = _dequantize_from_fp8(value, scale, target_dtype=torch.bfloat16)
                dequantized += 1

            native[model_key] = value

        logger.info(
            "DevstralFP8StateDictAdapter.from_hf: dequantized %d FP8 weights, "
            "dropped %d scale_inv + %d activation_scale keys (prefix_strip=%r)",
            dequantized,
            dropped_scales,
            dropped_act_scales,
            self.key_prefix,
        )
        return native

    # --------------------------------------------------------------------- #
    # Per-tensor conversion (used by the save path)                         #
    # --------------------------------------------------------------------- #
    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: Any, **kwargs
    ) -> list[tuple[str, Any]]:
        """Per-tensor model → HF conversion used by
        ``Checkpointer.save_model`` when ``to_hf`` is applied one tensor at a
        time. Devstral adds the VLM prefix and, if ``quantization=True``, emits
        an FP8 weight + scalar ``_scale_inv`` pair.
        """
        quantization = kwargs.get("quantization", False)
        hf_key = f"{self.key_prefix}{fqn}" if self.key_prefix else fqn
        if not quantization or not _is_fp8_weight_key(fqn):
            return [(hf_key, tensor)]
        # Save path: requantize bf16 → fp8. Use a per-tensor scalar scale
        # (Devstral's native format). We avoid actually quantizing here — the
        # only caller that sets quantization=True is the placeholder-allocation
        # path inside `Checkpointer.load_model`, which discards the values
        # before DCP fills them. If the save path ever requires real fp8
        # output, extend here.
        scale_placeholder = torch.empty((), dtype=torch.bfloat16)
        return [(hf_key, tensor), (hf_key + "_scale_inv", scale_placeholder)]
