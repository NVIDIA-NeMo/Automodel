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
