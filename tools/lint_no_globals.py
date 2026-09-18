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

"""Reject ``globals()`` outside the lazy-import caches in ``__getattr__``.

``globals()`` is most often reached for to swap a module-level function so that some
other function picks up the replacement -- a process-wide mutation that silently
changes behaviour for every concurrent or nested caller. Pass the collaborator in,
or override a method, instead.

The one sanctioned use is the PEP 562 lazy-import cache, ``globals()[name] = attr``
inside a module-level ``def __getattr__``: there is no other way to memoize a lazily
imported attribute into the module namespace.

Ruff has no equivalent rule (``PLW0603`` only covers the ``global`` statement), hence
this checker.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATHS = ("app.py", "nemo_automodel", "examples", "scripts", "tools")

HINT = (
    "globals() is banned: it mutates module state for every caller. "
    "Pass the function/object explicitly, or expose a method subclasses can override. "
    "The only exception is `globals()[name] = ...` inside a module-level `def __getattr__` "
    "(PEP 562 lazy-import cache)."
)


@dataclass(frozen=True)
class LintError:
    """A ``globals()`` use with source location."""

    path: Path
    line: int
    col: int


class _GlobalsVisitor(ast.NodeVisitor):
    """Collect ``globals()`` calls that are not PEP 562 lazy-import caches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[LintError] = []
        self._allowed: set[ast.Call] = set()
        self._func_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 (ast API)
        """Whitelist lazy-import caches in a module-level ``__getattr__``."""
        if node.name == "__getattr__" and self._func_depth == 0:
            self._allowed.update(_lazy_import_cache_calls(node))
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 (ast API)
        """A ``__getattr__`` on a class is not the module-level lazy-import hook."""
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (ast API)
        """Flag every ``globals()`` call that was not whitelisted."""
        if isinstance(node.func, ast.Name) and node.func.id == "globals" and node not in self._allowed:
            self.errors.append(LintError(self.path, node.lineno, node.col_offset))
        self.generic_visit(node)


def _lazy_import_cache_calls(func: ast.FunctionDef) -> list[ast.Call]:
    """Return the ``globals()`` calls used as ``globals()[name] = value`` inside ``func``."""
    calls = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Call)
                and isinstance(target.value.func, ast.Name)
                and target.value.func.id == "globals"
            ):
                calls.append(target.value)
    return calls


def collect_python_files(paths: list[Path]) -> list[Path]:
    """Expand files and directories into the Python files to lint."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py")))
        elif path.suffix == ".py":
            files.append(path)
    return files


def lint_file(path: Path) -> list[LintError]:
    """Lint a single Python file for banned ``globals()`` uses."""
    return lint_source(path.read_text(encoding="utf-8"), path)


def lint_source(source: str, path: Path) -> list[LintError]:
    """Lint Python source as if it came from ``path``."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Not our job to report; ruff already fails on unparseable files.
        return []
    visitor = _GlobalsVisitor(path)
    visitor.visit(tree)
    return visitor.errors


def format_errors(errors: list[LintError], automodel_dir: Path) -> str:
    """Format lint errors for CLI output."""
    lines = [f"{_relative_path(e.path, automodel_dir)}:{e.line}:{e.col + 1}: banned use of globals()" for e in errors]
    lines.append("")
    lines.append(HINT)
    return "\n".join(lines)


def _relative_path(path: Path, automodel_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(automodel_dir.resolve())
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    """Run the globals() linter."""
    parser = argparse.ArgumentParser(description="Reject globals() outside PEP 562 lazy-import caches.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Files or directories to lint. Defaults to: {', '.join(DEFAULT_PATHS)}.",
    )
    parser.add_argument("--automodel-dir", type=Path, default=Path.cwd(), help="Path to the AutoModel repository root.")
    args = parser.parse_args(argv)

    automodel_dir = args.automodel_dir.resolve()
    paths = args.paths or [automodel_dir / p for p in DEFAULT_PATHS]
    files = collect_python_files([p for p in paths if p.exists()])

    errors: list[LintError] = []
    for path in files:
        errors.extend(lint_file(path))

    if errors:
        print(format_errors(errors, automodel_dir), file=sys.stderr)
        return 1

    print(f"Linted {len(files)} Python file(s) for globals().", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
