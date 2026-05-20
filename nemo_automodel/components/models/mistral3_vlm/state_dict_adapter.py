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

"""State-dict adapter for the Mistral-3.5 128B (dawn-ridge) FP8 VLM.

Plugs into the standard nemo_automodel checkpoint flow
(nemo_automodel/components/checkpoint/checkpointing.py ~lines 510, 556) and
handles **FP8 dequantization** during load/save:

  * The checkpoint's language_model Linear weights are stored as per-tensor
    FP8 with a scalar ``weight_scale_inv`` sibling (and an unused
    ``activation_scale`` sibling). The adapter pairs each weight with its
    scale on load, dequantizes to bf16 (``w_bf16 = w_fp8.to(bf16) * scale``),
    and drops the scale keys. Vision tower + multi_modal_projector + lm_head
    are BF16 on disk and pass through unchanged.

Keys round-trip identically: the HF VLM state_dict and the on-disk keys
both use ``model.language_model.*``, ``model.vision_tower.*``,
``model.multi_modal_projector.*``, ``lm_head.weight`` — no rename needed.

Structurally modelled after
`nemo_automodel/components/models/deepseek_v3/state_dict_adapter.py`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


# Keys that should NOT be treated as FP8 weights — no `_scale_inv` sibling on
# save, no dequantize on load. The fixed suffix list covers layernorms +
# embeddings + the lm_head (always non-quantized in this family). VLM
# variants additionally pass module-prefix filters via the adapter's
# `not_fp8_prefixes` knob, matching `modules_to_not_convert` in the HF config
# (e.g. `model.vision_tower`, `model.multi_modal_projector`).
_NON_QUANTIZED_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "model.norm.weight",
)


def _is_fp8_weight_key(model_key: str, not_fp8_prefixes: tuple[str, ...] = ()) -> bool:
    """Return True iff `model_key` names an FP8 Linear weight."""
    if not model_key.endswith(".weight"):
        return False
    if any(model_key.endswith(suffix) for suffix in _NON_QUANTIZED_SUFFIXES):
        return False
    if any(model_key == p or model_key.startswith(p + ".") for p in not_fp8_prefixes):
        return False
    return True


def _dequantize_from_fp8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a single FP8 weight using its per-tensor scalar scale.

    The dawn-ridge 128B checkpoint uses per-tensor quantization
    (``weight_block_size=None``), so ``scale_inv`` is a 0-d scalar and
    dequant collapses to a simple multiply. The per-block formula
    (``transformers.integrations.finegrained_fp8.Fp8Dequantize.convert``,
    finegrained_fp8.py:867-906) is not needed here.
    """
    return weight_fp8.to(target_dtype) * scale_inv.to(target_dtype)


def _identity(k: str) -> str:
    return k


class Mistral3FP8StateDictAdapter(StateDictAdapter):
    """FP8 dequant adapter for the Mistral-3.5 128B dawn-ridge VLM.

    Keys round-trip identity (HF state_dict and on-disk keys match for the
    full VLM). Only language_model layer weights are FP8; vision_tower,
    multi_modal_projector, and lm_head are BF16 and pass through unchanged
    via the ``not_fp8_prefixes`` / ``_NON_QUANTIZED_SUFFIXES`` filters.
    """

    def __init__(
        self,
        *,
        native_to_hf: Callable[[str], str] = _identity,
        hf_to_native: Callable[[str], str] = _identity,
        layout_name: str = "vlm_full",
        not_fp8_prefixes: tuple[str, ...] = (),
    ):
        self._native_to_hf = native_to_hf
        self._hf_to_native = hf_to_native
        self._layout_name = layout_name
        self._not_fp8_prefixes = tuple(not_fp8_prefixes)

    @classmethod
    def for_vlm_full(cls) -> "Mistral3FP8StateDictAdapter":
        """Full-VLM path for Mistral3ForConditionalGeneration checkpoints.

        Keys round-trip identically between HF's VLM ``state_dict()`` and
        disk for the dawn-ridge-128B checkpoint (both use
        ``model.language_model.*``, ``model.vision_tower.*``,
        ``model.multi_modal_projector.*``, ``lm_head.weight``). Only the
        language_model layer weights are FP8; vision / mm_projector / lm_head
        are BF16 on disk and must be passed through without a scale_inv
        placeholder — otherwise DCP would fail trying to fetch a non-existent
        ``_scale_inv`` key.
        """
        not_fp8 = (
            "model.vision_tower",
            "model.multi_modal_projector",
            # "lm_head" already in _NON_QUANTIZED_SUFFIXES via suffix match.
        )
        return cls(layout_name="vlm_full", not_fp8_prefixes=not_fp8)

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

        When ``quantization=True`` the weight placeholder is also cast to
        ``torch.float8_e4m3fn`` so the DCP storage reader fetches FP8 bytes
        verbatim from safetensors (a bf16 target would silently cast-on-read
        and lose the scale multiply — see deepseek_v3/state_dict_adapter.py:220).
        A scalar ``_scale_inv`` placeholder is also emitted so DCP pulls it
        alongside the weight.
        """
        hf: dict[str, Any] = {}
        for model_key, value in state_dict.items():
            if exclude_key_regex is not None:
                import re

                if re.match(exclude_key_regex, model_key):
                    continue
            hf_key = self._native_to_hf(model_key)
            if quantization and _is_fp8_weight_key(model_key, self._not_fp8_prefixes):
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
        """Convert an HF-format (possibly FP8) state dict to model-native format."""
        native: dict[str, Any] = {}
        dequantized = 0
        dropped_scales = 0
        dropped_act_scales = 0

        scale_map = {k[: -len("_scale_inv")]: v for k, v in hf_state_dict.items() if k.endswith("_scale_inv")}

        for hf_key, value in hf_state_dict.items():
            if hf_key.endswith("_scale_inv"):
                dropped_scales += 1
                continue
            if hf_key.endswith(".activation_scale"):
                dropped_act_scales += 1
                continue

            model_key = self._hf_to_native(hf_key)

            if value.dtype == torch.float8_e4m3fn and hf_key in scale_map:
                scale = scale_map[hf_key]
                value = _dequantize_from_fp8(value, scale, target_dtype=torch.bfloat16)
                dequantized += 1

            native[model_key] = value

        logger.info(
            "Mistral3FP8StateDictAdapter[%s].from_hf: dequantized %d FP8 weights, "
            "dropped %d scale_inv + %d activation_scale keys",
            self._layout_name,
            dequantized,
            dropped_scales,
            dropped_act_scales,
        )
        return native

    # --------------------------------------------------------------------- #
    # Per-tensor conversion (save path)                                      #
    # --------------------------------------------------------------------- #
    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Per-tensor model → HF used by ``Checkpointer.save_model``."""
        quantization = kwargs.get("quantization", False)
        hf_key = self._native_to_hf(fqn)
        if not quantization or not _is_fp8_weight_key(fqn, self._not_fp8_prefixes):
            return [(hf_key, tensor)]
        scale_placeholder = torch.empty((), dtype=torch.bfloat16)
        return [(hf_key, tensor), (hf_key + "_scale_inv", scale_placeholder)]
