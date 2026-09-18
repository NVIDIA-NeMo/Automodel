# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import textwrap
from pathlib import Path

from tools.lint_no_globals import lint_source

MODULE = Path("nemo_automodel/components/example.py")


def _lint(source: str) -> list[int]:
    return [error.line for error in lint_source(textwrap.dedent(source), MODULE)]


def test_rejects_swapping_a_module_level_function():
    """The pattern this linter exists for: patching a module global around a call."""
    assert _lint(
        """
        def parallelize(model):
            original = globals()["shard"]
            globals()["shard"] = _custom_shard
            try:
                return _run(model)
            finally:
                globals()["shard"] = original
        """
    ) == [3, 4, 8]


def test_rejects_reading_globals():
    assert _lint("def lookup(name):\n    return globals()[name]\n") == [2]


def test_allows_pep_562_lazy_import_cache():
    """``globals()[name] = attr`` in a module-level ``__getattr__`` is the one exception."""
    assert (
        _lint(
            """
            def __getattr__(name):
                module_name, attr_name = _LAZY_ATTRS[name]
                attr = getattr(importlib.import_module(module_name), attr_name)
                globals()[name] = attr
                return attr
            """
        )
        == []
    )


def test_rejects_reads_inside_a_lazy_getattr():
    """Only the assignment form is exempt, not any globals() use in ``__getattr__``."""
    assert _lint("def __getattr__(name):\n    return globals()[name]\n") == [2]


def test_rejects_class_level_getattr():
    """A class ``__getattr__`` is not the module-level lazy-import hook."""
    assert _lint("class Wrapper:\n    def __getattr__(self, name):\n        globals()[name] = 1\n") == [3]


def test_rejects_nested_getattr():
    assert _lint("def outer():\n    def __getattr__(name):\n        globals()[name] = 1\n") == [3]


def test_shipped_package_is_clean():
    """Guards the real tree, so a reintroduced globals() fails unit tests too."""
    from tools.lint_no_globals import DEFAULT_PATHS, collect_python_files, lint_file

    repo_root = Path(__file__).resolve().parents[3]
    paths = [repo_root / name for name in DEFAULT_PATHS]
    errors = [e for path in collect_python_files([p for p in paths if p.exists()]) for e in lint_file(path)]

    assert errors == []
