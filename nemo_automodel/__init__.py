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
import sys

__all__ = [
    "_peft",
    "config",
    "datasets",
    "distributed",
    "loggers",
    "loss",
    "optim",
    "training",
    "transformers",
    "utils"
]

def __getattr__(name: str):
    if name in __all__:
        # import submodule on first access
        module = importlib.import_module(f"{__name__}.{name}")
        # cache it in globals() so future lookups do not re-import
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {rame!r}")

def __dir__():
    # so that dir(nemo_automodel) shows components
    return sorted(__all__)