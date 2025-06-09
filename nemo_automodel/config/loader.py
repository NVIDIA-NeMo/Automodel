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

import ast
import yaml
import importlib
import importlib.util
import os
import sys

def translate_value(v):
    """
    Convert a string token into the corresponding Python object.

    This function first checks for a handful of special symbols (None/true/false),
    then falls back to `ast.literal_eval`, and finally to returning the original
    string if parsing fails.

    Args:
        v (str):
            The raw string value to translate.

    Returns:
        The translated Python value, which may be:
          - None, True, or False for the special symbols
          - an int, float, tuple, list, dict, etc. if `ast.literal_eval` succeeds
          - the original string `v` if all parsing attempts fail
    """
    special_symbols = {
        'none': None,
        'None': None,
        'true': True,
        'True': True,
        'false': False,
        'False': False,
    }
    if v in special_symbols:
        return special_symbols[v]
    else:
        try:
            # smart-cast literals: numbers, dicts, lists, True/False, None
            return ast.literal_eval(v)
        except Exception:
            # fallback to raw string
            return v

def _resolve_target(dotted_path: str):
    """
    Resolve a dotted path to a Python object.
    1) Find the longest importable module prefix.
    2) getattr() the rest.
    3) If that fails, fall back to scanning sys.path for .py or package dirs.
    """
    parts = dotted_path.split(".")

    # 1) Try longest‐prefix module import + getattr the rest
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        remainder = parts[i:]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        # we got a module; now walk its attributes
        try:
            obj = module
            for name in remainder:
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            # we imported module_name but one of the remainder attrs failed
            raise ImportError(
                f"Module '{module_name}' loaded, "
                f"but cannot resolve attribute '{'.'.join(remainder)}' in '{dotted_path}'"
            )

    # 2) Fallback: scan sys.path for a .py file or package dir matching parts[:-1]
    for base in sys.path:
        pkg_dir = os.path.join(base, *parts[:-1])
        candidates = [
            pkg_dir + ".py",
            os.path.join(pkg_dir, "__init__.py"),
        ]
        for cand in candidates:
            if not os.path.isfile(cand):
                continue
            module_name = "_dynamic_" + "_".join(parts[:-1])
            spec = importlib.util.spec_from_file_location(module_name, cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            try:
                return getattr(mod, parts[-1])
            except AttributeError:
                raise ImportError(
                    f"Loaded '{cand}' as module but no attribute '{parts[-1]}'"
                )

    # 3) Give up
    raise ImportError(f"Cannot resolve target: {dotted_path}")


class ConfigNode:
    """
    A configuration node that wraps a dictionary (or parts of it) from a YAML file.

    This class allows nested dictionaries and lists to be accessed as attributes and
    provides functionality to instantiate objects from configuration.
    """
    def __init__(self, d):
        """
        Initialize the ConfigNode.

        Args:
            d (dict): A dictionary representing configuration options.
        """
        self.__dict__ = {
            k: self._wrap(k, v) for k, v in d.items()
        }

    def _wrap(self, k, v):
        """
        Wrap a configuration value based on its type.

        Args:
            k (str): The key corresponding to the value.
            v: The value to be wrapped.

        Returns:
            The wrapped value.
        """
        if isinstance(v, dict):
            return ConfigNode(v)
        elif isinstance(v, list):
            return [self._wrap('', i) for i in v]
        elif k.endswith('_fn'):
            return _resolve_target(v)
        else:
            return translate_value(v)

    def instantiate(self, *args, **kwargs):
        """
        Instantiate the target object specified in the configuration.

        This method looks for the "_target_" attribute in the configuration and resolves
        it to a callable function or class which is then instantiated.

        Args:
            *args: Positional arguments for the target instantiation.
            **kwargs: Keyword arguments to override or add to the configuration values.

        Returns:
            The instantiated object.

        Raises:
            AttributeError: If no "_target_" attribute is found in the configuration.
        """
        if not hasattr(self, "_target_"):
            raise AttributeError("No _target_ found to instantiate")

        func = _resolve_target(self._target_)

        # Prepare kwargs from config
        config_kwargs = {}
        for k, v in self.__dict__.items():
            if k == '_target_':
                continue
            if k.endswith('_fn'):
                config_kwargs[k] = v
            else:
                config_kwargs[k] = self._instantiate_value(v)

        # Override/add with passed kwargs
        config_kwargs.update(kwargs)

        return func(*args, **config_kwargs)

    def _instantiate_value(self, v):
        """
        Recursively instantiate configuration values.

        Args:
            v: The configuration value.

        Returns:
            The instantiated value.
        """
        if isinstance(v, ConfigNode) and hasattr(v, "_target_"):
            return v.instantiate()
        elif isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._instantiate_value(i) for i in v]
        else:
            return translate_value(v)

    def to_dict(self):
        """
        Convert the configuration node back to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration node.
        """
        return {
            k: self._unwrap(v) for k, v in self.__dict__.items()
        }

    def _unwrap(self, v):
        """
        Recursively convert wrapped configuration values to basic Python types.

        Args:
            v: The configuration value.

        Returns:
            The unwrapped value.
        """
        if isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._unwrap(i) for i in v]
        else:
            return v

    def get(self, key, default=None):
        """
        Retrieve a configuration value using a dotted key.

        If any component of the path is missing, returns the specified default value.

        Args:
            key (str): The dotted path key.
            default: A default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        parts = key.split(".")
        current = self
        # TODO(@akoumparouli): reduce?
        for p in parts:
            # Traverse dictionaries (ConfigNode)
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return default
            # Traverse lists by numeric index
            elif isinstance(current, list):
                try:
                    idx = int(p)
                    current = current[idx]
                except (ValueError, IndexError):
                    return default
            else:  # Reached a leaf but path still has components
                return default
        return current

    def set_by_dotted(self, dotted_key: str, value):
        """
        Set (or append) a value in the config using a dotted key.
        e.g. set_by_dotted("foo.bar.abc", 1) will ensure self.foo.bar.abc == 1
        """
        parts = dotted_key.split(".")
        node = self
        # walk / create intermediate ConfigNodes
        for p in parts[:-1]:
            if p not in node.__dict__ or not isinstance(node.__dict__[p], ConfigNode):
                node.__dict__[p] = ConfigNode({})
            node = node.__dict__[p]
        # wrap the final leaf value
        node.__dict__[parts[-1]] = node._wrap(parts[-1], value)

    def __repr__(self, level=0):
        """
        Return a string representation of the configuration node with indentation.

        Args:
            level (int): The current indentation level.

        Returns:
            str: An indented string representation of the configuration.
        """
        indent = "  " * level
        lines = [f"{indent}{key}: {self._repr_value(value, level)}" for key, value in self.__dict__.items()]
        return "\n#path: " + "\n".join(lines) + f"\n{indent}"

    def _repr_value(self, value, level):
        """
        Format a configuration value for the string representation.

        Args:
            value: The configuration value.
            level (int): The indentation level.

        Returns:
            str: A formatted string representation of the value.
        """
        if isinstance(value, ConfigNode):
            return value.__repr__(level + 1)
        elif isinstance(value, list):
            return "[\n" + \
                "\n".join([f"{'  ' * (level + 1)}{self._repr_value(i, level + 1)}" for i in value]) \
                + f"\n{'  ' * level}]"
        else:
            return repr(value)

    def __str__(self):
        """
        Return a string representation of the configuration node.

        Returns:
            str: The string representation.
        """
        return self.__repr__(level=0)

    def __contains__(self, key):
        """
        Check if a dotted key exists in the configuration.

        Args:
            key (str): The dotted key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        parts = key.split('.')
        current = self
        for p in parts:
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return False
        return current != self

def load_yaml_config(path):
    """
    Load a YAML configuration file and convert it to a ConfigNode.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        ConfigNode: A configuration node representing the YAML file.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
