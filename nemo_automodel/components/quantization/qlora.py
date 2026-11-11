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

from typing import TYPE_CHECKING, Literal, Protocol, overload

from torch import nn

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig

HAS_BNB, bitsandbytes = safe_import("bitsandbytes")
HAS_TRANSFORMERS, transformers = safe_import("transformers")


class _QuantizationConfig(Protocol):
    """Structural contract for the recipe's quantization configuration node.

    Recipes pass a ``ConfigNode`` (``self.cfg.get("quantization")``), not a
    plain ``dict``. Only the keys read by :func:`create_bnb_config` are
    declared; the ``get`` overloads pin each key to the type transformers
    expects for it.
    """

    def __contains__(self, key: str) -> bool: ...

    @overload
    def get(self, key: Literal["load_in_4bit"], default: bool = ...) -> bool: ...

    @overload
    def get(self, key: Literal["load_in_8bit"], default: bool = ...) -> bool: ...

    @overload
    def get(self, key: Literal["bnb_4bit_compute_dtype"], default: str = ...) -> str: ...

    @overload
    def get(self, key: Literal["bnb_4bit_use_double_quant"], default: bool = ...) -> bool: ...

    @overload
    def get(self, key: Literal["bnb_4bit_quant_type"], default: str = ...) -> str: ...

    @overload
    def get(self, key: Literal["bnb_4bit_quant_storage"], default: str = ...) -> str: ...


def create_bnb_config(config: _QuantizationConfig) -> BitsAndBytesConfig | None:
    """Create the BitsAndBytes configuration used for QLoRA.

    Args:
        config: The recipe's quantization section. A ``ConfigNode`` satisfies
            this protocol; only the two ``load_in_*`` flags and the four
            ``bnb_4bit_*`` settings are read.

    Returns:
        The BitsAndBytes configuration, or ``None`` when neither 4-bit nor
        8-bit loading is requested.

    Raises:
        ImportError: If bitsandbytes or transformers is unavailable.
        ValueError: If the quantization section omits both load flags.
    """
    if not HAS_BNB:
        raise ImportError("bitsandbytes is required for QLora")

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is required for QLora")

    if "load_in_4bit" not in config and "load_in_8bit" not in config:
        raise ValueError(
            "quantization requires 'load_in_4bit' or 'load_in_8bit'; remove the quantization section to disable it"
        )

    if config.get("load_in_4bit", False):
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_quant_storage=config.get("bnb_4bit_quant_storage", "bfloat16"),
        )
    elif config.get("load_in_8bit", False):
        return transformers.BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


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
