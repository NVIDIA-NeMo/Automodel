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
from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from nemo_automodel.components.config.loader import (
    ConfigNode,
    _resolve_target,
    load_module_from_file,
    load_yaml_config,
    translate_value,
)


@pytest.fixture()
def tmp_module(tmp_path: Path, monkeypatch):
    """
    Creates a throw-away Python module on disk, adds its directory to sys.path,
    imports and yields it.  Each test using this fixture receives an independent
    module namespace.
    """

    def _factory(name: str, source: str) -> Any:
        mod_path = tmp_path / f"{name}.py"
        mod_path.write_text(textwrap.dedent(source))
        # Prepend tmp_path to sys.path so importlib can find the module
        monkeypatch.syspath_prepend(str(tmp_path))
        return importlib.import_module(name)

    return _factory


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("none", None),
        ("None", None),
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        ("123", 123),  # int via literal_eval
        ("3.14", 3.14),  # float via literal_eval
        ("{'a': 1}", {"a": 1}),  # dict via literal_eval
        ("[1, 2, 3]", [1, 2, 3]),
        ("not_a_number", "not_a_number"),  # fall-back → original string
    ],
)
def test_translate_value(raw, expected):
    """translate_value must return the proper Python object for all branches."""
    assert translate_value(raw) == expected


def test_resolve_target_regular_import(tmp_module):
    """_resolve_target resolves objects that live in a regular importable module."""
    mod = tmp_module(
        "my_module",
        """
        class Dummy:
            pass

        def add(a, b):
            return a + b
        """,
    )
    assert _resolve_target("my_module.Dummy") is mod.Dummy
    assert _resolve_target("my_module.add") is mod.add


def test_resolve_target_filesystem_fallback(tmp_path, monkeypatch):
    """
    If the dotted path cannot be imported directly as a package/module,
    _resolve_target should scan sys.path for a .py file and load it dynamically.
    """
    src = tmp_path / "lonely.py"
    src.write_text("answer = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    # Remove “lonely” from sys.modules to guarantee the fallback code-path
    sys.modules.pop("lonely", None)

    assert _resolve_target("lonely.answer") == 42


def test_resolve_target_error():
    """An unknown dotted path should raise ImportError."""
    with pytest.raises(ImportError):
        _resolve_target("surely.does.not.exist")


def test_confignode_wrap_and_fn_resolution(tmp_module):
    """
    * Scalar values must be translated.
    * Keys ending with _fn must be resolved to callables.
    * Nested dicts and lists must become ConfigNodes or wrapped children.
    """
    mod = tmp_module(
        "wrap_mod",
        """
        def echo(x): return x
        """,
    )
    cfg_dict = {
        "int_val": "123",
        "my_fn": "wrap_mod.echo",  # note the *_fn suffix
        "nested": {"flag": "True"},
        "listy": ["1", "2", "3"],
    }
    cfg = ConfigNode(cfg_dict)

    assert cfg.int_val == 123
    assert callable(cfg.my_fn) and cfg.my_fn is mod.echo
    assert isinstance(cfg.nested, ConfigNode) and cfg.nested.flag is True
    assert cfg.listy == [1, 2, 3]


def test_dotted_get_set_contains():
    cfg = ConfigNode({})
    cfg.set_by_dotted("foo.bar.baz", "456")
    assert cfg.get("foo.bar.baz") == 456
    assert "foo.bar.baz" in cfg
    # List indexing inside dotted path
    cfg.set_by_dotted("arr.values", ["a", "b", "c"])
    assert cfg.get("arr.values.1") == "b"
    assert cfg.get("non.existent", default="sentinel") == "sentinel"


def test_instantiate_simple(tmp_module):
    """
    Instantiate a simple object with scalar arguments supplied as strings
    that must be translated to int.
    """
    mod = tmp_module(
        "factory_mod",
        """
        class Point:
            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y
        """,
    )
    cfg = ConfigNode(
        {"_target_": "factory_mod.Point", "x": "1", "y": "2"},
    )
    obj = cfg.instantiate()
    assert isinstance(obj, mod.Point)
    assert (obj.x, obj.y) == (1, 2)

@pytest.mark.parametrize(
    "path, exists",
    [
        ("factory_mod", True),
        ("NO_FACTORY_MOD", False),
    ],
)
def test_instantiate_path_simple(tmp_module, path, exists):
    """
    Instantiate a simple object with scalar arguments supplied as strings
    that must be translated to int.
    """
    mod = tmp_module(
        "factory_mod",
        """
        class Point:
            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y
        """,
    )
    cfg = ConfigNode(
        {"factory_mod": {"_target_": "factory_mod.Point", "x": "1", "y": "2"}},
    )
    obj = cfg.instantiate_path(path)
    if exists:
        assert isinstance(obj, mod.Point)
        assert (obj.x, obj.y) == (1, 2)
    else:
        assert obj is None


def test_instantiate_simple_raises(tmp_module):
    """
    Instantiate a simple object with scalar arguments supplied as strings
    that must be translated to int.
    """
    mod = tmp_module(  # noqa: F841
        "factory_mod",
        """
        class Point:
            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y
        """,
    )
    cfg = ConfigNode(
        {"_target_": "factory_mod.Point", "x1": "1", "y": "2"},
    )
    with pytest.raises(TypeError, match=r"Point.__init__.. got an unexpected keyword argument 'x1'"):
        obj = cfg.instantiate()  # noqa: F841


def test_instantiate_nested(tmp_module):
    """Nested ConfigNodes with their own _target_ must be instantiated first."""
    mod = tmp_module(  # noqa: F841
        "nested_mod",
        """
        def to_int(value):
            return int(value)

        class Box:
            def __init__(self, width, height):
                self.width = width
                self.height = height
        """,
    )
    cfg = ConfigNode(
        {
            "_target_": "nested_mod.Box",
            "width": "3",
            "height": {
                "_target_": "nested_mod.to_int",
                "value": "7",
            },
        }
    )
    box = cfg.instantiate()
    assert (box.width, box.height) == (3, 7)  # height collected from nested call


def test_instantiate_with_overrides(tmp_module):
    """Keyword overrides passed to instantiate() must take precedence."""
    mod = tmp_module(  # noqa: F841
        "override_mod",
        """
        class Pair:
            def __init__(self, a, b):
                self.a = a
                self.b = b
        """,
    )
    cfg = ConfigNode({"_target_": "override_mod.Pair", "a": "10", "b": "20"})
    pair = cfg.instantiate(b=99)
    assert (pair.a, pair.b) == (10, 99)


def test_instantiate_missing_target():
    """Calling instantiate() without a _target_ key should raise AttributeError."""
    with pytest.raises(AttributeError):
        ConfigNode({"foo": "bar"}).instantiate()


def test_to_dict_roundtrip():
    src = {"alpha": "1", "beta": {"gamma": "True"}}
    cfg = ConfigNode(src)
    roundtrip = cfg.to_dict()
    # translate_value means "1"->1 and "True"->True
    assert roundtrip == {"alpha": 1, "beta": {"gamma": True}}


def test_repr_contains_paths():
    """
    __repr__/__str__ aren’t tested for exact formatting, only that they include
    top-level keys and are well-formed strings (no exceptions).
    """
    cfg = ConfigNode({"x": "1", "y": "2"})
    s = str(cfg)
    assert "x:" in s and "y:" in s


def test_str_does_not_include_internal_original_strings_mapping():
    """
    ConfigNode keeps a private `_original_strings` mapping for `_target_` and `*_fn` keys.
    This internal bookkeeping must never show up in `str(cfg)` / `repr(cfg)`.
    """
    cfg = ConfigNode({"_target_": "builtins.len", "my_fn": "builtins.len", "x": "1"})
    # Ensure the mapping is actually populated so the test is meaningful.
    assert cfg.get_as_string("_target_") == "builtins.len"
    assert cfg.get_as_string("my_fn") == "builtins.len"

    assert "_original_strings" not in str(cfg)
    assert "_original_strings" not in repr(cfg)


def test_load_yaml_config(tmp_path):
    """YAML helper must produce a ConfigNode with translated values."""
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
        learning_rate: "0.001"
        scheduler:
          step_size: "10"
          gamma: "0.5"
        """
    )
    cfg = load_yaml_config(str(yml))
    assert isinstance(cfg, ConfigNode)
    assert cfg.learning_rate == 0.001
    assert cfg.scheduler.step_size == 10
    assert cfg.scheduler.gamma == 0.5


def test_load_yaml_config_resolves_oc_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.local")
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
        dataset:
          delta_storage_options:
            DATABRICKS_HOST: ${oc.env:DATABRICKS_HOST}
        """
    )
    cfg = load_yaml_config(str(yml))
    assert cfg.dataset.delta_storage_options.DATABRICKS_HOST == "https://example.databricks.local"

def test_load_yaml_config_oc_env_default(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NOT_SET", raising=False)
    yml = tmp_path / "cfg.yaml"
    yml.write_text("x: ${oc.env:NOT_SET,default_value}\n")
    cfg = load_yaml_config(str(yml))
    assert cfg.x == "default_value"


def test_load_yaml_config_oc_env_missing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)
    yml = tmp_path / "cfg.yaml"
    yml.write_text("x: ${oc.env:MISSING_ENV_VAR}\n")
    with pytest.raises(KeyError, match="MISSING_ENV_VAR"):
        _ = load_yaml_config(str(yml))


def test_load_yaml_config_resolves_braced_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FOO_ENV", "bar")
    yml = tmp_path / "cfg.yaml"
    yml.write_text("x: ${FOO_ENV}\n")
    cfg = load_yaml_config(str(yml))
    assert cfg.x == "bar"
    s = str(cfg)
    assert "${FOO_ENV}" in s
    assert "bar" not in s


def test_load_yaml_config_resolves_dollar_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FOO_ENV", "bar")
    yml = tmp_path / "cfg.yaml"
    yml.write_text("x: $FOO_ENV\n")
    cfg = load_yaml_config(str(yml))
    assert cfg.x == "bar"
    s = str(cfg)
    assert "$FOO_ENV" in s
    assert "bar" not in s


def test_load_yaml_config_resolves_dotted_env_var_name(monkeypatch, tmp_path: Path):
    # Dots are treated as part of the env var name.
    monkeypatch.setenv("foo.bar.baz", "qux")
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
        a: ${foo.bar.baz}
        b: $foo.bar.baz
        path: ${foo.bar.baz}/$foo.bar.baz
        """
    )
    cfg = load_yaml_config(str(yml))
    assert cfg.a == "qux"
    assert cfg.b == "qux"
    assert cfg.path == "qux/qux"
    # Printing should show original placeholders (avoid leaking resolved values).
    s = str(cfg)
    assert "${foo.bar.baz}" in s
    assert "$foo.bar.baz" in s
    assert "qux/qux" not in s


def test_confignode_repr_uses_orig_value_for_oc_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SECRET_ENV", "super_secret_value")
    yml = tmp_path / "cfg.yaml"
    yml.write_text("x: ${oc.env:SECRET_ENV}\n")
    cfg = load_yaml_config(str(yml))

    # Runtime value is resolved
    assert cfg.x == "super_secret_value"
    # But printing the config should not leak the resolved secret
    s = str(cfg)
    assert "${oc.env:SECRET_ENV}" in s
    assert "super_secret_value" not in s


def test_load_module_from_file(tmp_path):
    """Module is imported, its globals are accessible, and it gets a unique name."""
    py_file = tmp_path / "plugin.py"
    py_file.write_text(
        textwrap.dedent(
            """
            FOO = 123
            def bar():
                return "bar"
            """
        )
    )

    mod = load_module_from_file(py_file.as_posix())

    assert mod.__name__.endswith("plugin")  # dynamic name was created
    assert mod.FOO == 123
    assert mod.bar() == "bar"


def test_resolve_target_non_string_passthrough():
    """If the input is already an object, it is returned unchanged."""
    assert _resolve_target(len) is len
    assert _resolve_target(42) == 42


def test_resolve_target_file_colon(tmp_path):
    """`path/to/file.py:object_name` is resolved correctly."""
    script = tmp_path / "tools.py"
    script.write_text(
        textwrap.dedent(
            """
            def meaning():
                return 42
            """
        )
    )
    target = _resolve_target(f"{script}:{'meaning'}")

    # We get back the function object itself
    assert callable(target)
    assert target() == 42


def test_instantiate_skips_kwargs_overridden_nested(tmp_module):
    """Nested _target_ configs must NOT be instantiated when overridden by kwargs."""
    mod = tmp_module(
        "skip_mod",
        """
        call_count = 0

        def expensive_factory(value=0):
            import skip_mod
            skip_mod.call_count += 1
            return value

        class Container:
            def __init__(self, child, label="default"):
                self.child = child
                self.label = label
        """,
    )
    cfg = ConfigNode(
        {
            "_target_": "skip_mod.Container",
            "child": {"_target_": "skip_mod.expensive_factory", "value": "42"},
            "label": "from_config",
        }
    )

    pre_instantiated_child = "already_built"
    result = cfg.instantiate(child=pre_instantiated_child)

    assert result.child == "already_built"
    assert result.label == "from_config"
    assert mod.call_count == 0, "nested _target_ should not have been called when overridden by kwarg"


def test_resolve_target_asserts_on_non_py_suffix(tmp_path):
    """When the left part before ':' does not end in .py an AssertionError is raised."""
    bad = tmp_path / "data.txt"
    bad.write_text("x = 1")
    with pytest.raises(AssertionError):
        _resolve_target(f"{bad}:x")


def test_resolve_target_asserts_on_missing_file(tmp_path):
    """Missing script should raise AssertionError."""
    missing = tmp_path / "ghost.py"  # file is NOT created
    with pytest.raises(AssertionError):
        _resolve_target(f"{missing}:nothing")
