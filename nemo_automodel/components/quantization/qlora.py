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


from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from torch import nn

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig

HAS_BNB, bitsandbytes = safe_import("bitsandbytes")
HAS_TRANSFORMERS, transformers = safe_import("transformers")


@dataclass(frozen=True)
class BnbQuantizationConfig:
    """BitsAndBytes weight quantization for QLoRA (the recipe's ``quantization:`` block).

    At least one of ``load_in_4bit`` and ``load_in_8bit`` must be given; giving both as
    ``False`` disables quantization. 4-bit loading wins when both are ``True``.

    Attributes:
        load_in_4bit: Load the weights in 4 bits.
        load_in_8bit: Load the weights in 8 bits.
        bnb_4bit_compute_dtype: Compute dtype of the 4-bit layers.
        bnb_4bit_use_double_quant: Also quantize the 4-bit quantization constants.
        bnb_4bit_quant_type: 4-bit data type (``"nf4"`` or ``"fp4"``).
        bnb_4bit_quant_storage: Storage dtype of the packed 4-bit weights.
    """

    load_in_4bit: bool | None = None
    load_in_8bit: bool | None = None
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_quant_storage: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.load_in_4bit is None and self.load_in_8bit is None:
            raise ValueError(
                "quantization requires 'load_in_4bit' or 'load_in_8bit'; remove the quantization section to disable it"
            )

    def build(self) -> BitsAndBytesConfig | None:
        """Create the BitsAndBytes configuration used for QLoRA.

        Returns:
            The BitsAndBytes configuration, or ``None`` when neither 4-bit nor
            8-bit loading is requested.

        Raises:
            ImportError: If bitsandbytes or transformers is unavailable.
        """
        if not HAS_BNB:
            raise ImportError("bitsandbytes is required for QLora")

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for QLora")

        if self.load_in_4bit:
            return transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
            )
        if self.load_in_8bit:
            return transformers.BitsAndBytesConfig(load_in_8bit=True)
        return None


def create_bnb_config(config: Mapping[str, Any]) -> BitsAndBytesConfig | None:
    """Deprecated: build a :class:`BnbQuantizationConfig` and call its ``build()`` instead.

    Recipes build the typed config from their ``quantization:`` block at the recipe
    boundary (``nemo_automodel.recipes._typed_config.bnb_quantization_config_from_node``).
    This helper reads a raw ``quantization:`` mapping inside the component and will be
    removed.

    Args:
        config: The quantization section; only the two ``load_in_*`` flags and the
            four ``bnb_4bit_*`` settings are read.

    Returns:
        The BitsAndBytes configuration, or ``None`` when neither 4-bit nor
        8-bit loading is requested.

    Raises:
        ImportError: If bitsandbytes or transformers is unavailable.
        ValueError: If the quantization section omits both load flags.
    """
    warnings.warn(
        "create_bnb_config is deprecated; use BnbQuantizationConfig(...).build()",
        FutureWarning,
        stacklevel=2,
    )
    settings = {f.name: config.get(f.name, None) for f in fields(BnbQuantizationConfig) if f.name in config}
    return BnbQuantizationConfig(**settings).build()


def verify_qlora_quantization(model: nn.Module) -> bool:
    """Return whether any module carries a BitsAndBytes quantized state.

    Args:
        model: Model to inspect.

    Returns:
        ``True`` when at least one module has a ``quant_state`` of type
        ``bitsandbytes.functional.QuantState``.
    """
    for module in model.modules():
        # bitsandbytes attaches ``quant_state`` dynamically to quantized
        # modules; it is not part of ``nn.Module``'s declared interface.
        quant_state = getattr(module, "quant_state", None)
        if quant_state is not None and isinstance(quant_state, bitsandbytes.functional.QuantState):
            return True
    return False
