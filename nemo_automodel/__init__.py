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

__all__ = [
    "recipes",
    "shared",
    "components",
    "__version__",
    "__package_name__",
]

# Promote NeMoAutoModelForCausalLM, AutoModelForImageTextToText into the top level
# to enable: `from nemo_automodel import NeMoAutoModelForCausalLM`
try:
    # adjust this import path if your class lives somewhere else
    from nemo_automodel.components._transformers.auto_model import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText

    globals()["NeMoAutoModelForCausalLM"] = NeMoAutoModelForCausalLM
    globals()["NeMoAutoModelForImageTextToText"] = NeMoAutoModelForImageTextToText
    __all__.append("NeMoAutoModelForCausalLM")
    __all__.append("NeMoAutoModelForImageTextToText")
except:
    # optional dependency might be missing,
    # leave the name off the module namespace so other imports still work
    pass


def __getattr__(name: str):
    """
    Lazily import and cache submodules listed in __all__ when accessed.

    Raises:
        AttributeError if the name isn’t in __all__.
    """
    if name in __all__:
        # import submodule on first access
        module = importlib.import_module(f"{__name__}.{name}")
        # cache it in globals() so future lookups do not re-import
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
