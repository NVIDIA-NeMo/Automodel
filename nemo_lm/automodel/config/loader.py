# config_loader.py
import yaml
import importlib
from functools import reduce
import os
import importlib
import importlib.util
import os
import sys
from functools import reduce

class ConfigNode:
    def __init__(self, d):
        self.__dict__ = {
            k: self._wrap(k, v) for k, v in d.items()
        }

    def _wrap(self, k, v):
        if isinstance(v, dict):
            return ConfigNode(v)
        elif isinstance(v, list):
            return [self._wrap('', i) for i in v]
        elif k.endswith('_fn'):
            return self._resolve_target(v)
        elif isinstance(v, (int, float)):
            return v
        elif v.isdigit():
            return int(v)
        else:
            try:
                return float(v)
            except ValueError:
                return v

    def instantiate(self, *args, **kwargs):
        if not hasattr(self, "_target_"):
            raise AttributeError("No _target_ found to instantiate")

        target = self._target_
        func = self._resolve_target(target)

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
        if isinstance(v, ConfigNode) and hasattr(v, "_target_"):
            return v.instantiate()
        elif isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._instantiate_value(i) for i in v]
        else:
            return v

    def _resolve_target(self, dotted_path):
        parts = dotted_path.split(".")

        # Try standard import first
        # e.g.: torchdata.stateful_dataloader.StatefulDataLoader
        # TODO(@akoumparouli): make this more robust
        if len(parts) > 2:
            try:
                module = importlib.import_module('.'.join(parts[:-1]))
                return getattr(module, parts[-1])
            except (ModuleNotFoundError, AttributeError):
                pass
        try:
            module = importlib.import_module(parts[0])
            return reduce(getattr, parts[1:], module)
        except (ModuleNotFoundError, AttributeError):
            pass

        # Try to resolve it as a local module by searching sys.path
        for path in sys.path:
            try_path = os.path.join(path, *parts[:-1]) + ".py"
            if os.path.isfile(try_path):
                module_name = "_dynamic_" + "_".join(parts[:-1])
                spec = importlib.util.spec_from_file_location(module_name, try_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                return getattr(mod, parts[-1])
        raise ImportError(f"Cannot resolve target: {dotted_path}. Searched paths for: {'.'.join(parts[:-1])}.py")

    def to_dict(self):
        return {
            k: self._unwrap(v) for k, v in self.__dict__.items()
        }

    def _unwrap(self, v):
        if isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._unwrap(i) for i in v]
        else:
            return v

    def get(self, key, default=None):
        """
        Retrieve a value by dotted path.  If any component is missing,
        or an index is invalid, ``default`` is returned.
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

    def __repr__(self, level=0):
        # Formatting for indentation
        indent = "  " * level
        lines = [f"{indent}{key}: {self._repr_value(value, level)}" for key, value in self.__dict__.items()]
        return "\n#path: " + "\n".join(lines) + f"\n{indent}"

    def _repr_value(self, value, level):
        if isinstance(value, ConfigNode):
            return value.__repr__(level + 1)
        elif isinstance(value, list):
            return "[\n" + "\n".join([f"{'  ' * (level + 1)}{self._repr_value(i, level + 1)}" for i in value]) + f"\n{'  ' * level}]"
        else:
            return repr(value)

    def __str__(self):
        return self.__repr__(level=0)

    def __contains__(self, key):
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
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
