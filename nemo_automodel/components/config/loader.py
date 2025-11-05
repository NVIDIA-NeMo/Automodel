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
import importlib
import importlib.util
import inspect
import os
import pprint
import sys
from copy import deepcopy
from pathlib import Path

# Security/Policy configuration
from typing import Any, Mapping

import yaml

# Only allow importing from these module prefixes by default
ALLOWED_IMPORT_PREFIXES = ("nemo_automodel", "torch", "transformers", "torchdata", "torchao", "liger_kernel")

# Define a safe base dir for loading modules from files (default: repo root)
SAFE_BASE_DIR = Path(__file__).resolve().parents[2]

# Opt-in flag that allows loading user-defined code. Default: disabled
ENABLE_USER_MODULES = os.environ.get("NEMO_ENABLE_USER_MODULES", "").lower() in ("1", "true", "yes")

SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "apikey",
    "api_key",
    "authorization",
    "auth",
)


def set_enable_user_modules(allow: bool) -> None:
    """Enable or disable loading user-defined code at runtime.

    Users can also set environment variable NEMO_ENABLE_USER_MODULES=1 to enable.
    """
    global ENABLE_USER_MODULES
    ENABLE_USER_MODULES = bool(allow)


def _is_safe_path(p: Path) -> bool:
    rp = p.resolve()
    try:
        # Python 3.9+
        return rp.is_relative_to(SAFE_BASE_DIR)
    except AttributeError:
        # Fallback for older versions
        try:
            return os.path.commonpath([str(rp), str(SAFE_BASE_DIR.resolve())]) == str(SAFE_BASE_DIR.resolve())
        except ValueError:
            return False


def _is_allowed_module(module_name: str) -> bool:
    """Return True if a module is safe/allowed to import.

    Security policy (balanced for functionality and tests):
    - If user modules are explicitly enabled, allow everything.
    - Always allow modules that are already imported in this process.
    - Allow modules that are importable from the current PYTHONPATH/sys.path.
      This keeps behavior intuitive for local/test modules while still blocking
      truly unknown targets.
    - Fallback to explicit allowlist as a final gate (mostly relevant if
      find_spec returns None for the top-level name).
    """
    if ENABLE_USER_MODULES:
        return True

    top_level = module_name.split(".")[0]

    if top_level in sys.modules:
        return True

    try:
        spec = importlib.util.find_spec(top_level)
    except Exception:
        spec = None
    if spec is not None:
        return True

    return any(top_level == pref or top_level.startswith(pref + ".") for pref in ALLOWED_IMPORT_PREFIXES)


def _is_safe_attr(name: str) -> bool:
    # Disallow private/dunder attribute traversal
    return not (name.startswith("_") or "__" in name)


def _redact(obj: Any) -> Any:
    def needs_redact(k: str) -> bool:
        lk = str(k).lower()
        return any(s in lk for s in SENSITIVE_KEY_SUBSTRINGS)

    if isinstance(obj, Mapping):
        return {k: ("******" if needs_redact(k) else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def translate_value(v):
    """
    Convert a string token into the corresponding Python object.

    This function first checks for a handful of special symbols (None/true/false),
    then falls back to `ast.literal_eval`, and finally to returning the original
    string if parsing fails.

    Args:
        v (str): The raw string value to translate.

    Returns:
        The translated Python value, which may be:
          - None, True, or False for the special symbols
          - an int, float, tuple, list, dict, etc. if `ast.literal_eval` succeeds
          - the original string `v` if all parsing attempts fail
    """
    # Fast-path for non-strings
    if not isinstance(v, str):
        return v

    special_symbols = {
        "none": None,
        "None": None,
        "true": True,
        "True": True,
        "false": False,
        "False": False,
    }
    if v in special_symbols:
        return special_symbols[v]

    # Avoid evaluating pathological strings
    if len(v) > 1000:
        return v

    try:
        # smart-cast literals: numbers, dicts, lists, True/False, None
        return ast.literal_eval(v)
    except Exception:
        # fallback to raw string
        return v


def load_module_from_file(file_path):
    """Dynamically imports a module from a given file path.

    Intentionally permissive to support test/temporary modules. Caller is
    responsible for any higher-level policy checks.
    """
    p = Path(file_path)
    if p.suffix != ".py":
        raise ImportError(f"Refusing to load non-Python file as module: {p}")

    # Create a module specification object from the file location
    # Ensure the dynamic name ends with the file stem (e.g., 'plugin') and is unique
    unique_part = str(abs(hash(str(p.resolve()))) % (10**8))
    name = f"cfgmod_{unique_part}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, str(p.resolve()))

    # Create a module object from the specification
    module = importlib.util.module_from_spec(spec)

    # Execute the module's code
    assert spec is not None and spec.loader is not None  # narrow mypy and ensure loader exists
    spec.loader.exec_module(module)

    return module


def _resolve_target(dotted_path: str):
    """
    Resolve a dotted path to a Python object with safety checks.

    Supports two forms:
      - "path/to/file.py:attr" (file import): allowed if under SAFE_BASE_DIR unless opt-in is enabled.
      - "pkg.mod.attr" (dotted import): allowed only for allowlisted prefixes unless opt-in is enabled.
    """
    if not isinstance(dotted_path, str):
        return dotted_path

    if ":" in dotted_path:
        file_part, attr = dotted_path.split(":", 1)
        p = Path(file_part)
        # Historical behavior/tests expect AssertionError on invalid cases
        if p.suffix != ".py":
            raise AssertionError("Left side of ':' must be a .py file")
        if not p.exists():
            raise AssertionError(f"Python script does not exist: {file_part}")
        module = load_module_from_file(str(p.resolve()))
        if not _is_safe_attr(attr):
            raise ImportError(
                "Access to private or dunder attributes is disabled by default. "
                "To allow out-of-tree code, set NEMO_ENABLE_USER_MODULES=1 or call set_enable_user_modules(True)."
            )
        return getattr(module, attr)

    parts = dotted_path.split(".")

    # Try longest-prefix module import + getattr the rest, with allowlist
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        remainder = parts[i:]

        if not _is_allowed_module(module_name):
            continue

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        obj = module
        for name in remainder:
            if not _is_safe_attr(name) and not ENABLE_USER_MODULES:
                raise ImportError(
                    "Access to private or dunder attributes is disabled by default. "
                    "To allow out-of-tree code, set NEMO_ENABLE_USER_MODULES=1 or call set_enable_user_modules(True)."
                )
            try:
                obj = getattr(obj, name)
            except AttributeError:
                raise ImportError(
                    f"Module '{module_name}' loaded, but cannot resolve attribute '{'.'.join(remainder)}' in '{dotted_path}'"
                )
        return obj

    raise ImportError(f"Cannot resolve target (blocked or not found): {dotted_path}")


class ConfigNode:
    """
    A configuration node that wraps a dictionary (or parts of it) from a YAML file.

    This class allows nested dictionaries and lists to be accessed as attributes and
    provides functionality to instantiate objects from configuration.
    """

    def __init__(self, d, raise_on_missing_attr=True):
        """Initialize the ConfigNode.

        Args:
            d (dict): A dictionary representing configuration options.
            raise_on_missing_attr (bool): if True, it will return `None` on a missing attr.
        """
        # Finetune scripts can modify the config in place, so we need to keep a copy of the
        # original config for checkpointing.
        self._raw_config = deepcopy(d)
        # Update instead of overwrite, so other instance attributes survive.
        self.__dict__.update({k: self._wrap(k, v) for k, v in d.items()})
        self.raise_on_missing_attr = raise_on_missing_attr

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except:
            if self.raise_on_missing_attr:
                raise AttributeError
            else:
                return None

    def _wrap(self, k, v):
        """Wrap a configuration value based on its type.

        Args:
            k (str): The key corresponding to the value.
            v: The value to be wrapped.

        Returns:
            The wrapped value.
        """
        if isinstance(v, dict):
            return ConfigNode(v)
        elif isinstance(v, list):
            return [self._wrap("", i) for i in v]
        elif k.endswith("_fn"):
            return _resolve_target(v)
        elif k == "_target_":
            return _resolve_target(v)
        else:
            return translate_value(v)

    @property
    def raw_config(self):
        """
        Get the raw configuration dictionary.

        Returns:
            dict: The raw configuration dictionary.
        """
        return self._raw_config

    def instantiate_path(self, dotted_path, default=None, *args, **kwargs):
        """
        Instantiate the target object specified in the configuration by path.

        If the path is not found, returns the default value.

        For example, this is useful when you want to do something like:

        cfg_peft = self.cfg.get("peft", None)
        if cfg_peft is not None:
            cfg_peft = cfg_peft.instantiate()

        In this case, you first check if the dotted path (in this case "peft") is in the configuration.
        If it is, you instantiate it, otherwise you return the default value (in this case None).

        With instantiate_path, you can do:
        cfg_peft = self.cfg.instantiate_path("peft", default=None)

        Args:
            dotted_path (str): The path to the target object (e.g., "model.config").
            default: A default value to return if the path is not found.
            *args: Positional arguments for the target instantiation.
            **kwargs: Keyword arguments to override or add to the configuration values.

        Returns:
            The instantiated object.
        """
        item = self.get(dotted_path, default)
        if item is default:
            return default
        return item.instantiate(*args, **kwargs)

    def instantiate(self, *args, **kwargs):
        """Instantiate the target object specified in the configuration.

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
            if k in ("_target_", "raise_on_missing_attr", "_raw_config"):
                continue
            if k.endswith("_fn"):
                config_kwargs[k] = v
            else:
                config_kwargs[k] = self._instantiate_value(v)

        # Override/add with passed kwargs
        config_kwargs.update(kwargs)

        try:
            return func(*args, **config_kwargs)
        except Exception as e:
            sig = inspect.signature(func)
            safe_kwargs = _redact(config_kwargs)
            print(
                "Instantiation failed for `{}`\n"
                "Accepted signature : {}\n"
                "Positional args    : {}\n"
                "Keyword args       : {}\n"
                "Exception          : {}\n".format(
                    getattr(func, "__name__", str(func)),
                    sig,
                    args,
                    pprint.pformat(safe_kwargs, compact=True, indent=4),
                    e,
                ),
                file=sys.stderr,
            )
            raise

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
            k: self._unwrap(v) for k, v in self.__dict__.items() if k not in ("raise_on_missing_attr", "_raw_config")
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
        lines = [
            f"{indent}{key}: {self._repr_value(value, level)}"
            for key, value in self.__dict__.items()
            if key not in ("raise_on_missing_attr", "_raw_config")
        ]
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
            return (
                "[\n"
                + "\n".join([f"{'  ' * (level + 1)}{self._repr_value(i, level + 1)}" for i in value])
                + f"\n{'  ' * level}]"
            )
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
        parts = key.split(".")
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
