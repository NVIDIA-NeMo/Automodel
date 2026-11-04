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

import pytest

from tools.lint_no_globals import DEFAULT_PATHS, collect_python_files, lint_file, lint_source, main

MODULE = Path("nemo_automodel/components/example.py")


def _lint(source: str | bytes) -> list[int]:
    if isinstance(source, str):
        source = textwrap.dedent(source)
    return [error.line for error in lint_source(source, MODULE)]


def _messages(source: str) -> list[str]:
    return [error.message for error in lint_source(textwrap.dedent(source), MODULE)]


# --------------------------------------------------------------------------- globals()


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


def test_allows_lazy_import_cache_under_module_level_control_flow():
    """``__getattr__`` defined under a module-level ``if``/``try`` is still the module's hook."""
    assert _lint("if not TYPE_CHECKING:\n    def __getattr__(name):\n        globals()[name] = 1\n") == []


def test_rejects_reads_inside_a_lazy_getattr():
    """Only the assignment form is exempt, not any globals() use in ``__getattr__``."""
    assert _lint("def __getattr__(name):\n    return globals()[name]\n") == [2]


def test_exemption_requires_the_getattr_parameter_as_key():
    """A fixed key is a monkey-patch that happens to live in ``__getattr__``."""
    assert _lint('def __getattr__(name):\n    globals()["shard"] = _custom_shard\n') == [2]


def test_exemption_does_not_extend_to_nested_scopes():
    assert _lint("def __getattr__(name):\n    def patch():\n        globals()[name] = 1\n    patch()\n") == [3]


def test_async_getattr_is_not_a_lazy_import_hook():
    assert _lint("async def __getattr__(name):\n    globals()[name] = 1\n") == [2]


def test_rejects_class_level_getattr():
    """A class ``__getattr__`` is not the module-level lazy-import hook."""
    assert _lint("class Wrapper:\n    def __getattr__(self, name):\n        globals()[name] = 1\n") == [3]


def test_rejects_nested_getattr():
    assert _lint("def outer():\n    def __getattr__(name):\n        globals()[name] = 1\n") == [3]


# ------------------------------------------------------------ equivalent spellings


def test_rejects_setattr_and_delattr_on_own_module():
    assert _messages(
        """
        import sys

        setattr(sys.modules[__name__], "shard", _custom_shard)
        delattr(sys.modules[__name__], "shard")
        """
    ) == [
        "banned use of setattr(sys.modules[__name__], ...)",
        "banned use of delattr(sys.modules[__name__], ...)",
    ]


def test_rejects_own_module_dict():
    assert _lint(
        """
        import sys

        sys.modules[__name__].__dict__["shard"] = _custom_shard
        sys.modules[__name__].__dict__.update(patches)
        vars(sys.modules[__name__])["shard"] = _custom_shard
        """
    ) == [4, 5, 6]


def test_setattr_on_own_module_is_not_exempt_inside_getattr():
    """The cache has exactly one sanctioned spelling."""
    assert _lint("import sys\n\ndef __getattr__(name):\n    setattr(sys.modules[__name__], name, 1)\n") == [4]


def test_rejects_bare_vars_at_module_scope():
    assert _lint('import sys\n\nvars()["shard"] = _custom_shard\n') == [3]


def test_allows_vars_of_an_object_and_locals_vars():
    """``vars(obj)`` is ``obj.__dict__``; bare ``vars()`` inside a function is ``locals()``."""
    assert _lint("def show(obj):\n    print(vars(obj))\n    print(vars())\n") == []


def test_allows_setattr_on_other_modules_and_objects():
    assert _lint("import sys\n\nsetattr(sys.modules['other'], 'x', 1)\nsetattr(obj, 'x', 1)\n") == []


# ---------------------------------------------------------------------- robustness


def test_utf8_bom_is_linted_not_skipped():
    assert _lint(b'\xef\xbb\xbfglobals()["x"] = 1\n') == [1]


def test_non_utf8_coding_cookie_is_linted():
    assert _lint(b'# -*- coding: latin-1 -*-\nx = "caf\xe9"\nglobals()["x"] = 1\n') == [3]


@pytest.mark.parametrize("source", [b"x = 1\x00\n", b"def (:\n"])
def test_unparseable_source_is_reported(source):
    errors = lint_source(source, MODULE)
    assert len(errors) == 1
    assert errors[0].message.startswith("cannot parse file")


# ------------------------------------------------------------------------- main()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_main_fails_on_missing_explicit_path(tmp_path):
    assert main([str(tmp_path / "nope.py")]) == 2


def test_main_fails_outside_a_checkout(tmp_path):
    """A wrong cwd must not turn the CI step into a silent no-op."""
    assert main(["--automodel-dir", str(tmp_path)]) == 2


def test_main_fails_when_nothing_to_lint(tmp_path):
    (tmp_path / "nemo_automodel").mkdir()
    assert main(["--automodel-dir", str(tmp_path)]) == 2


def test_main_exit_codes(tmp_path, capsys):
    clean = _write(tmp_path / "nemo_automodel" / "clean.py", "x = 1\n")
    assert main(["--automodel-dir", str(tmp_path)]) == 0

    bad = _write(tmp_path / "nemo_automodel" / "bad.py", 'globals()["x"] = 1\n')
    assert main(["--automodel-dir", str(tmp_path)]) == 1
    assert "nemo_automodel/bad.py:1:1: banned use of globals()" in capsys.readouterr().err

    assert main([str(clean)]) == 0
    assert main([str(bad), str(clean)]) == 1


def test_shipped_package_is_clean():
    """Guards the real tree, so a reintroduced globals() fails unit tests too."""
    repo_root = Path(__file__).resolve().parents[3]
    paths = [repo_root / name for name in DEFAULT_PATHS]
    errors = [e for path in collect_python_files([p for p in paths if p.exists()]) for e in lint_file(path)]

    assert errors == []
