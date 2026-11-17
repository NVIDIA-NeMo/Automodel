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
import types
from copy import deepcopy
from pathlib import Path

# Security/Policy configuration
from typing import Any, Mapping, SupportsIndex

import yaml

# Only allow importing from these module prefixes by default
ALLOWED_IMPORT_PREFIXES = ("nemo_automodel", "torch", "transformers", "torchdata", "torchao", "liger_kernel")

# Python files inside the installed nemo_automodel package can be loaded by path without the
# opt-in below, because they are already part of the trusted installation.
SAFE_BASE_DIR = Path(__file__).resolve().parents[2]

# Opt-in flag that allows loading user-defined code: Python files outside SAFE_BASE_DIR and
# private attributes of dotted targets. Default: disabled
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
    """Return True if ``p`` resolves, after following symlinks, to a path inside SAFE_BASE_DIR."""
    return p.resolve().is_relative_to(SAFE_BASE_DIR)


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


def translate_value(v: Any) -> Any:
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

    _orig_value: str
    _no_env_resolve: bool

    def __new__(cls, value: str, orig_value: str) -> "_OrigValueStr":
        obj = super().__new__(cls, value)
        obj._orig_value = orig_value
        obj._no_env_resolve = True
        return obj

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[type["_OrigValueStr"], tuple[str, str]]:
        """Preserve the resolved and original values when copying this string."""
        return type(self), (str(self), self._orig_value)


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
                # A default is part of the config text, so it may itself reference an env var.
                return _resolve_in_str(default)
            raise KeyError(f"Environment variable '{var_name}' is not set (required by '{token}').")

        # One pass over `${...}` and `$VAR` / `$var.dot.var` (dot-separated segments; do not allow trailing dot),
        # so the text substituted for one reference is not rescanned for another.
        def _repl(match: re.Match[str]) -> str:
            if match.group(1) is None:
                return _get_env(match.group(2), None, match.group(0))
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

        return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)", _repl, value)

    if isinstance(obj, dict):
        return {k: resolve_yaml_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_yaml_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_in_str(obj)
    return obj


def load_module_from_file(file_path: str | Path) -> types.ModuleType:
    """Execute a Python file and return it as a module.

    Executing the file runs its top-level code, so a file outside the installed
    ``nemo_automodel`` package (``SAFE_BASE_DIR``) is refused unless user modules
    are enabled with ``NEMO_ENABLE_USER_MODULES=1`` or ``set_enable_user_modules(True)``.

    Args:
        file_path: Path to a ``.py`` file. Symlinks are resolved before the check.

    Returns:
        The executed module, registered in ``sys.modules`` under a generated name.

    Raises:
        ImportError: If the file is not a ``.py`` file, is outside the package
            while user modules are disabled, or cannot be loaded.
    """
    p = Path(file_path)
    if p.suffix != ".py":
        raise ImportError(f"Refusing to load non-Python file as module: {p}")
    resolved = p.resolve()
    if not ENABLE_USER_MODULES and not _is_safe_path(resolved):
        raise ImportError(
            f"Refusing to execute {resolved}: loading a Python file from outside the nemo_automodel package "
            "runs its code and is disabled by default. If you trust this file, set NEMO_ENABLE_USER_MODULES=1 "
            "or call set_enable_user_modules(True)."
        )

    unique_part = str(abs(hash(str(resolved))) % (10**8))
    name = f"cfgmod_{unique_part}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {p}")

    module = importlib.util.module_from_spec(spec)
    # Register the module in sys.modules *before* executing it. The dataclass
    # machinery (and other tools that inspect annotations) looks the module up
    # via sys.modules[cls.__module__]; if the module is absent this raises an
    # AttributeError when the module uses `from __future__ import annotations`.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise

    return module


def _resolve_target(dotted_path: str) -> Any:
    """
    Resolve a target string to a Python object.

    Supports two forms:
      - "path/to/file.py:attr" (file import): executes the file. A file outside the installed
        nemo_automodel package needs the NEMO_ENABLE_USER_MODULES opt-in (see `load_module_from_file`).
        A private or dunder `attr` is rejected before the file runs.
      - "pkg.mod.attr" (dotted import): imports the longest module prefix that `_is_allowed_module`
        accepts, then resolves the rest as attributes. Private attributes need the opt-in.
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
        if not _is_safe_attr(attr):
            raise ImportError(f"Refusing private or dunder attribute '{attr}' in file target '{dotted_path}'.")
        module = load_module_from_file(p)
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
        except ModuleNotFoundError as e:
            # Only a missing candidate (or one of its parents) means "try a shorter prefix".
            # A dependency missing inside an existing module is the real error; falling back
            # would report it as an unresolvable attribute instead.
            if e.name is not None and not (module_name == e.name or module_name.startswith(e.name + ".")):
                raise
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

    def __init__(self, d: dict[str, Any], raise_on_missing_attr: bool = True) -> None:
        """Initialize the ConfigNode.

        Args:
            d (dict): A dictionary representing configuration options.
            raise_on_missing_attr (bool): if True, it will return `None` on a missing attr.
        """
        # Finetune scripts can modify the config in place, so keep the originally loaded
        # values separate from the live, serializable attributes.
        self._raw_config = deepcopy(d)
        # Store original string values before resolution (for _fn and _target_ keys)
        self._original_strings: dict[str, str] = {}
        # Update instead of overwrite, so other instance attributes survive.
        self.__dict__.update({k: self._wrap(k, v) for k, v in d.items()})
        self.raise_on_missing_attr = raise_on_missing_attr

    def __getattr__(self, key: str) -> Any:
        # Dunder methods must raise AttributeError so that Python's
        # copy/pickle/repr protocols (which call
        # getattr(obj, "__setstate__", None), etc.) treat them as absent
        # rather than as missing regular attributes — returning None here
        # would break copy.deepcopy and isinstance checks.
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        try:
            return self.__dict__[key]
        except KeyError:
            # Read the flag safely: on a partially-constructed node
            # (e.g. a deepcopy target that has not yet had __setstate__
            # called) raise_on_missing_attr may itself be missing, which
            # would recurse back into __getattr__.  Default to True.
            if self.__dict__.get("raise_on_missing_attr", True):
                raise AttributeError(key)
            return None

    def _wrap(self, k: str, v: Any) -> Any:
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
    def raw_config(self) -> dict[str, Any]:
        """
        Get the raw configuration dictionary.

        Returns:
            dict: The raw configuration dictionary.
        """
        return self._raw_config

    def instantiate_path(self, dotted_path: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
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

    def instantiate(self, *args: Any, **kwargs: Any) -> Any:
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
            if k in kwargs:
                continue  # will be overridden — skip expensive instantiation
            if k.endswith("_fn"):
                config_kwargs[k] = v
            else:
                config_kwargs[k] = self._instantiate_value(v)

        # Resolve env interpolations on config-derived kwargs only, before merging
        # runtime arguments (e.g. `examples`, `processor`).  Doing it after the
        # update() would cause resolve_yaml_env_vars to scan actual data content,
        # which can contain arbitrary strings like "$P" and raise spurious KeyErrors.
        config_kwargs = resolve_yaml_env_vars(config_kwargs)

        # Override/add with passed kwargs (runtime data — must NOT be env-var-resolved)
        config_kwargs.update(kwargs)

        import traceback

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
            print(traceback.format_exc())
            raise e

    def _instantiate_value(self, v: Any) -> Any:
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

    def to_dict(self) -> dict[str, Any]:
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

    def _to_dotted_path(self, obj: Any) -> str:
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

    def to_yaml_dict(
        self, *, resolve_env: bool = False, redact_sensitive: bool = False, use_orig_values: bool = False
    ) -> dict[str, Any]:
        """
        Convert configuration to a YAML-ready dictionary:
        - Preserves typed scalars (ints, floats, bools)
        - Converts callables/classes/methods (e.g., _target_, *_fn) to dotted path strings
        - Recurses through nested ConfigNodes and lists, emitting tuples as lists

        Args:
            resolve_env: If True, resolve `${oc.env:VAR}` interpolations in the returned dict.
                This does not mutate the in-memory config.
            redact_sensitive: If True, redact values for keys that look sensitive (token/secret/etc).
            use_orig_values: If True, prefer `_orig_value` (when present) for safe printing/logging.
        """

        def _convert(key: str | None, value: Any) -> Any:
            # Nested config
            if isinstance(value, ConfigNode):
                return value.to_yaml_dict(
                    resolve_env=resolve_env,
                    redact_sensitive=redact_sensitive,
                    use_orig_values=use_orig_values,
                )
            # Lists and tuples (CLI values such as `(0.9, 0.95)` parse to tuples); emit plain
            # sequences so the result stays loadable with yaml.safe_load.
            if isinstance(value, (list, tuple)):
                return [_convert(None, v) for v in value]
            # Dicts (shouldn't normally appear because we wrap into ConfigNode, but handle defensively)
            if isinstance(value, dict):
                return {k: _convert(k, v) for k, v in value.items()}
            # Prefer original YAML strings only while their resolved callable remains live.
            # Programmatic target/callback overrides must serialize the replacement.
            orig_strings = getattr(self, "_original_strings", {})
            # Convert targets/functions to dotted path strings
            is_target_like = key == "_target_" or (isinstance(key, str) and key.endswith("_fn")) or key == "collate_fn"
            try:
                import inspect as _inspect

                if is_target_like and (callable(value) or _inspect.ismethod(value) or _inspect.isclass(value)):
                    dotted_value = self._to_dotted_path(value)
                    if use_orig_values and key in orig_strings:
                        try:
                            original_value = _resolve_target(orig_strings[key])
                        except Exception:
                            original_value = None
                        if callable(original_value) and (
                            value is original_value or dotted_value == self._to_dotted_path(original_value)
                        ):
                            return orig_strings[key]
                    return dotted_value
                # Even if the key isn't target-like, convert bare callables to dotted path to avoid <function ...> repr
                if callable(value) or _inspect.ismethod(value) or _inspect.isclass(value):
                    return self._to_dotted_path(value)
            except Exception:
                pass
            if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
                return str(value).removeprefix("torch.")
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

    def _unwrap(self, v: Any) -> Any:
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

    def get_as_string(self, key: str, default: str | None = None) -> str:
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
        if default is not None:
            return default
        raise KeyError(f"Key {key} not found")

    def get(self, key: str, default: Any = None) -> Any:
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

    def set_by_dotted(self, dotted_key: str, value: Any) -> None:
        """
        Set (or append) a value in the config using a dotted key.

        e.g. set_by_dotted("foo.bar.abc", 1) will ensure self.foo.bar.abc == 1

        A numeric part indexes into an existing list, as in ``get``:
        set_by_dotted("optimizer.betas.1", 0.98) replaces only that element.
        """
        parts = dotted_key.split(".")
        node = self
        owner = self  # nearest ConfigNode, used to wrap values stored in lists
        # walk / create intermediate ConfigNodes
        for p in parts[:-1]:
            if isinstance(node, list):
                node = node[self._list_index(node, p, dotted_key)]
            else:
                if p not in node.__dict__ or not isinstance(node.__dict__[p], (ConfigNode, list)):
                    node.__dict__[p] = ConfigNode({})
                node = node.__dict__[p]
            if isinstance(node, ConfigNode):
                owner = node
            elif not isinstance(node, list):
                raise TypeError(
                    f"Cannot set '{dotted_key}': '{p}' is a {type(node).__name__}, not a config node or list"
                )
        # wrap the final leaf value
        if isinstance(node, list):
            node[self._list_index(node, parts[-1], dotted_key)] = owner._wrap("", value)
        else:
            node.__dict__[parts[-1]] = node._wrap(parts[-1], value)

    @staticmethod
    def _list_index(items: list[Any], part: str, dotted_key: str) -> int:
        """Return the list position named by one dotted-key part, or raise."""
        try:
            idx = int(part)
            items[idx]
        except (ValueError, IndexError):
            raise IndexError(
                f"Cannot set '{dotted_key}': '{part}' is not a valid index into a list of length {len(items)}"
            ) from None
        return idx

    def _format(self, level: int = 0, *, use_orig_values: bool = True) -> str:
        """Return a string representation of the configuration node with indentation.

        Args:
            level: The current indentation level.
            use_orig_values: If True, prefer original placeholder strings (e.g. ``${VAR}``)
                stored on ``_OrigValueStr`` values for safe logging.
        """
        indent = "  " * level
        lines = [
            f"{indent}{key}: {self._repr_value(value, level, use_orig_values=use_orig_values)}"
            for key, value in self.__dict__.items()
            if key not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        ]
        return "\n".join(lines) + f"\n{indent}"

    def __repr__(self) -> str:
        return self._format(level=0, use_orig_values=True)

    def _repr_value(self, value: Any, level: int, *, use_orig_values: bool = True) -> str:
        if isinstance(value, ConfigNode):
            return value._format(level + 1, use_orig_values=use_orig_values)
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

    def __str__(self) -> str:
        return self._format(level=0, use_orig_values=True)

    def __contains__(self, key: object) -> bool:
        """
        Check if a dotted key exists in the configuration.

        Args:
            key (str): The dotted key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        if not isinstance(key, str):
            return False
        parts = key.split(".")
        current: Any = self
        for p in parts:
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return False
        return bool(current != self)


def config_to_yaml_str(cfg_obj: Any, *, use_orig_values: bool = True) -> str:
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
    result: str = yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False).strip()
    return result


def load_yaml_config(path: str | Path) -> "ConfigNode":
    """
    Load a YAML configuration file and convert it to a ConfigNode.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        ConfigNode: A configuration node representing the YAML file.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
