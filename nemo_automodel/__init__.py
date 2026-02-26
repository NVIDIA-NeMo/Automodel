# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib

from .package_info import __package_name__, __version__

# Keep the base package import lightweight.
# Heavy dependencies (e.g., torch/transformers) are intentionally imported lazily
# via __getattr__ so importing tokenizers doesn't pull in the full training stack.

_SUBMODULES = {"recipes", "shared", "components"}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NeMoAutoModelForCausalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForCausalLM"),
    "NeMoAutoModelForImageTextToText": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForImageTextToText"),
    "NeMoAutoModelForMultimodalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForMultimodalLM"),
    "NeMoAutoModelForSequenceClassification": (
        "nemo_automodel._transformers.auto_model",
        "NeMoAutoModelForSequenceClassification",
    ),
    "NeMoAutoModelForTextToWaveform": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForTextToWaveform"),
    "NeMoAutoModelBiencoder": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelBiencoder"),
    "NeMoAutoTokenizer": ("nemo_automodel._transformers.auto_tokenizer", "NeMoAutoTokenizer"),
}

__all__ = sorted([*_SUBMODULES, "__version__", "__package_name__", *_LAZY_ATTRS.keys()])


def __getattr__(name: str):
    """
    Lazily import and cache selected submodules / exported symbols when accessed.

    Raises:
        AttributeError if the name isn’t in __all__.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
