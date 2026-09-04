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

import importlib as _importlib

_LAZY_ATTRS = {
    "ConfigNode": (".loader", "ConfigNode"),
    "config_to_yaml_str": (".loader", "config_to_yaml_str"),
    "parse_args_and_load_config": ("._arg_parser", "parse_args_and_load_config"),
    "resolve_yaml_env_vars": (".loader", "resolve_yaml_env_vars"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
