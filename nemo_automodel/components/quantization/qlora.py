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


from nemo_automodel.shared.import_utils import safe_import

HAS_BNB, bitsandbytes = safe_import("bitsandbytes")
HAS_TRANSFORMERS, transformers = safe_import("transformers")


def create_bnb_config(config: dict):
    """Create BitsAndBytes config for quantization."""
    if not HAS_BNB:
        raise ImportError("bitsandbytes is required for QLora")

    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is required for QLora")

    if config.load_in_4bit:
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_quant_storage=config.get("bnb_4bit_quant_storage", "bfloat16"),
        )
    elif config.load_in_8bit:
        return transformers.BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def verify_qlora_quantization(model) -> bool:
    """Verify that the model has been properly quantized."""
    for name, module in model.named_modules():
        if hasattr(module, "quant_state") and module.quant_state.__class__ == bitsandbytes.functional.QuantState:
            return True
    return False
