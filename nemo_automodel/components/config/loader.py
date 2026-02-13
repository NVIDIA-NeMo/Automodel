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
import re
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


class _OrigValueStr(str):
    """String that keeps its original (unresolved) value for safe printing."""

    def __new__(cls, value: str, orig_value: str):
        obj = super().__new__(cls, value)
        obj._orig_value = orig_value
        # Mark values that already went through env resolution so we don't accidentally
        # re-resolve `$FOO` sequences that happen to appear in secrets.
        obj._no_env_resolve = True
        return obj


def resolve_yaml_env_vars(obj: Any) -> Any:
    """Resolve env var references inside a YAML-loaded container.

    Supported forms inside strings:
    - `${VAR}` / `${VAR,default}`
    - `${var.dot.var}` (dots are treated as part of the env var name)
    - `$VAR` / `$var.dot.var`
    - Back-compat: `${oc.env:VAR}` / `${oc.env:VAR,default}`
    """

    def _resolve_in_str(value: str) -> str:
        # Skip values that opted out / were already resolved.
        if hasattr(value, "_no_env_resolve"):
            return value
        if "$" not in value:
            return value

        def _get_env(var_name: str, default: str | None, token: str) -> str:
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default
            raise KeyError(f"Environment variable '{var_name}' is not set (required by '{token}').")

        # Handle braced patterns first: ${...}
        braced_pattern = re.compile(r"\$\{([^}]+)\}")

        def _braced_repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            # Back-compat: ${oc.env:VAR}
            if expr.startswith("oc.env:"):
                expr = expr[len("oc.env:") :].strip()

            if "," in expr:
                var_name, default = expr.split(",", 1)
                var_name = var_name.strip()
                default = default.strip()
            else:
                var_name, default = expr, None

            return _get_env(var_name, default, match.group(0))

        value = braced_pattern.sub(_braced_repl, value)

        # Then handle $VAR patterns (allow dots in name).
        # `$VAR` and `$var.dot.var` (dot-separated segments; do not allow trailing dot).
        dollar_pattern = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)")

        def _dollar_repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return _get_env(var_name, None, match.group(0))

        return dollar_pattern.sub(_dollar_repl, value)

    if isinstance(obj, dict):
        return {k: resolve_yaml_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_yaml_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_in_str(obj)
    return obj


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
        # Store original string values before resolution (for _fn and _target_ keys)
        self._original_strings = {}
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
            if isinstance(v, str):
                self._original_strings[k] = v
            return _resolve_target(v)
        elif k == "_target_":
            if isinstance(v, str):
                self._original_strings[k] = v
            return _resolve_target(v)
        else:
            # Support `${oc.env:VAR}` (with optional default) in YAML scalars.
            # Store the resolved value for runtime, but keep the original token for safe printing.
            if isinstance(v, str) and "$" in v:
                resolved = resolve_yaml_env_vars(v)
                translated = translate_value(resolved)
                if isinstance(translated, str) and resolved != v:
                    return _OrigValueStr(translated, v)
                return translated
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
            if k in ("_target_", "raise_on_missing_attr", "_raw_config", "_original_strings"):
                continue
            if k.endswith("_fn"):
                config_kwargs[k] = v
            else:
                config_kwargs[k] = self._instantiate_value(v)

        # Override/add with passed kwargs
        config_kwargs.update(kwargs)
        # Resolve env interpolations at the last moment, so printing/saving the config
        # does not leak secrets (e.g., `${oc.env:HF_TOKEN}` remains in YAML output).
        config_kwargs = resolve_yaml_env_vars(config_kwargs)

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
            # Dict-like configs should resolve env vars before being passed to targets.
            return resolve_yaml_env_vars(v.to_dict())
        elif isinstance(v, list):
            return [self._instantiate_value(i) for i in v]
        else:
            # Resolve leaf env vars (strings) before attempting to cast.
            return translate_value(resolve_yaml_env_vars(v))

    def to_dict(self):
        """
        Convert the configuration node back to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration node.
        """
        return {
            k: self._unwrap(v)
            for k, v in self.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }

    def _to_dotted_path(self, obj):
        """
        Convert a callable/class/method object to a dotted path string.

        Best-effort normalization for a few common cases to produce concise, user-friendly paths.
        """
        # Bound method on a class (e.g., Class.from_pretrained)
        try:
            import inspect as _inspect  # local alias to avoid confusion with top-level import

            if _inspect.ismethod(obj):
                owner = getattr(obj, "__self__", None)
                if _inspect.isclass(owner):
                    method_name = getattr(obj, "__name__", "unknown")
                    module_name = getattr(owner, "__module__", None) or ""
                    class_name = getattr(owner, "__name__", "UnknownClass")
                    # Prefer shortened top-level for NeMoAutoModel* classes if possible
                    if class_name.startswith("NeMoAutoModel"):
                        module_name = "nemo_automodel"
                    dotted = f"{module_name}.{class_name}.{method_name}".lstrip(".")
                else:
                    # Bound to instance – fall back to module + qualname
                    module_name = getattr(obj, "__module__", None) or ""
                    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "unknown"))
                    dotted = f"{module_name}.{qualname}".lstrip(".")
            elif _inspect.isfunction(obj):
                module_name = getattr(obj, "__module__", None) or ""
                qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "unknown"))
                dotted = f"{module_name}.{qualname}".lstrip(".")
            elif _inspect.isclass(obj):
                module_name = getattr(obj, "__module__", None) or ""
                class_name = getattr(obj, "__name__", "UnknownClass")
                dotted = f"{module_name}.{class_name}".lstrip(".")
            else:
                module_name = getattr(obj, "__module__", None) or ""
                qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
                dotted = f"{module_name}.{qualname}".lstrip(".")
        except Exception:
            # Fallback to repr if anything goes wrong
            return repr(obj)
        return dotted

    def to_yaml_dict(self, *, resolve_env: bool = False, redact_sensitive: bool = False, use_orig_values: bool = False):
        """
        Convert configuration to a YAML-ready dictionary:
        - Preserves typed scalars (ints, floats, bools)
        - Converts callables/classes/methods (e.g., _target_, *_fn) to dotted path strings
        - Recurses through nested ConfigNodes and lists

        Args:
            resolve_env: If True, resolve `${oc.env:VAR}` interpolations in the returned dict.
                This does not mutate the in-memory config.
            redact_sensitive: If True, redact values for keys that look sensitive (token/secret/etc).
            use_orig_values: If True, prefer `_orig_value` (when present) for safe printing/logging.
        """

        def _convert(key, value):
            # Nested config
            if isinstance(value, ConfigNode):
                return value.to_yaml_dict(
                    resolve_env=resolve_env,
                    redact_sensitive=redact_sensitive,
                    use_orig_values=use_orig_values,
                )
            # Lists
            if isinstance(value, list):
                return [_convert(None, v) for v in value]
            # Dicts (shouldn't normally appear because we wrap into ConfigNode, but handle defensively)
            if isinstance(value, dict):
                return {k: _convert(k, v) for k, v in value.items()}
            # Prefer original YAML string for _target_ / *_fn when use_orig_values is set
            orig_strings = getattr(self, "_original_strings", {})
            if use_orig_values and key in orig_strings:
                return orig_strings[key]
            # Convert targets/functions to dotted path strings
            is_target_like = key == "_target_" or (isinstance(key, str) and key.endswith("_fn")) or key == "collate_fn"
            try:
                import inspect as _inspect

                if is_target_like and (callable(value) or _inspect.ismethod(value) or _inspect.isclass(value)):
                    return self._to_dotted_path(value)
                # Even if the key isn't target-like, convert bare callables to dotted path to avoid <function ...> repr
                if callable(value) or _inspect.ismethod(value) or _inspect.isclass(value):
                    return self._to_dotted_path(value)
            except Exception:
                pass
            if use_orig_values and hasattr(value, "_orig_value"):
                return getattr(value, "_orig_value")
            # Primitive – already typed via translate_value/_wrap
            return value

        # Walk live attributes; exclude internal keys so they never appear in YAML output
        out = {
            k: _convert(k, v)
            for k, v in self.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }
        if resolve_env:
            out = resolve_yaml_env_vars(out)
        if redact_sensitive:
            out = _redact(out)
        return out

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

    def get_as_string(self, key, default=None):
        """
        Get the string representation of a configuration value.

        If the value is a function or class (resolved from an import path),
        returns the original import path string. Otherwise returns the value
        as a string.

        Args:
            key (str): The key to look up.

        Returns:
            str: The string representation of the value, or None if key not found.
        """
        # Check if we stored the original string (for _fn and _target_ keys)
        if key in self._original_strings:
            return self._original_strings[key]
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key {key} not found")

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

    def __repr__(self, level: int = 0, *, use_orig_values: bool = True):
        """
        Return a string representation of the configuration node with indentation.

        Args:
            level (int): The current indentation level.
            use_orig_values (bool): If True, prefer original placeholder strings (e.g. `${VAR}`)
                stored on `_OrigValueStr` values for safe logging. If False, show resolved values.

        Returns:
            str: An indented string representation of the configuration.
        """
        indent = "  " * level
        lines = [
            f"{indent}{key}: {self._repr_value(value, level, use_orig_values=use_orig_values)}"
            for key, value in self.__dict__.items()
            if key not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        ]
        return "\n".join(lines) + f"\n{indent}"

    def _repr_value(self, value, level, *, use_orig_values: bool = True):
        """
        Format a configuration value for the string representation.

        Args:
            value: The configuration value.
            level (int): The indentation level.
            use_orig_values (bool): If True, prefer original placeholder strings stored on
                `_OrigValueStr` values for safe logging. If False, show resolved values.

        Returns:
            str: A formatted string representation of the value.
        """
        if isinstance(value, ConfigNode):
            return value.__repr__(level + 1, use_orig_values=use_orig_values)
        elif isinstance(value, list):
            return (
                "[\n"
                + "\n".join(
                    [
                        f"{'  ' * (level + 1)}{self._repr_value(i, level + 1, use_orig_values=use_orig_values)}"
                        for i in value
                    ]
                )
                + f"\n{'  ' * level}]"
            )
        else:
            if use_orig_values and hasattr(value, "_orig_value"):
                return repr(getattr(value, "_orig_value"))
            return repr(value)

    def __str__(self):
        """
        Return a string representation of the configuration node.

        Returns:
            str: The string representation.
        """
        # Keep printing safe by default: preserve original placeholders (e.g. `${VAR}`) for any
        # values that were resolved from environment variables.
        return self.__repr__(level=0, use_orig_values=True)

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


def config_to_yaml_str(cfg_obj, *, use_orig_values: bool = True):
    """
    Convert a config object to a YAML string suitable for logging/printing.

    Uses original placeholder strings (e.g. `${VAR}`) and original _target_/*_fn
    strings when use_orig_values is True; never includes internal keys like
    _original_strings. If cfg_obj is a ConfigNode, uses to_yaml_dict(); otherwise
    falls back to to_dict() or a plain dict.
    """
    if cfg_obj is None:
        return ""
    if hasattr(cfg_obj, "to_yaml_dict"):
        cfg_dict = cfg_obj.to_yaml_dict(use_orig_values=use_orig_values)
    elif hasattr(cfg_obj, "to_dict"):
        cfg_dict = cfg_obj.to_dict()
    else:
        cfg_dict = dict(cfg_obj)
    return yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False).strip()


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
