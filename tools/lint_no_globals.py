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

"""Reject dynamic mutation of the module namespace outside PEP 562 lazy-import caches.

``globals()`` -- and its spellings ``vars()``, ``sys.modules[__name__].__dict__`` and
``setattr(sys.modules[__name__], ...)`` -- is most often used to swap a module-level
function so that some other function picks up the replacement: a process-wide
mutation that silently changes behavior for every concurrent or nested caller. Pass
the collaborator in, or override a method, instead.

The one sanctioned use is the PEP 562 lazy-import cache::

    def __getattr__(name):
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

i.e. ``globals()[<parameter>] = ...`` written directly inside a module-level
``def __getattr__``. Nothing else is exempt, so the cache has exactly one spelling.

Ruff has no equivalent rule: ``PLW0603`` covers the ``global`` statement and ``TID251``
cannot ban a bare builtin call. Hence this checker.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATHS = ("app.py", "nemo_automodel", "examples", "scripts", "tools", "tutorials")

HINT = (
    "Dynamic module-namespace mutation is banned: it changes behavior for every caller of the "
    "module. Pass the function/object explicitly, or expose a method subclasses can override. "
    "The only exception is the PEP 562 lazy-import cache, `globals()[name] = ...` written directly "
    "inside a module-level `def __getattr__(name)`."
)

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class LintError:
    """A lint failure with source location."""

    path: Path
    line: int
    col: int
    message: str


def _is_call_to(node: ast.AST, *names: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names


def _is_own_module(node: ast.AST) -> bool:
    """True for the expression ``sys.modules[__name__]``."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "__name__"
    )


def _walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Yield ``node``'s descendants without entering nested function or class scopes."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, _SCOPE_NODES):
            continue
        yield child
        stack.extend(ast.iter_child_nodes(child))


def _lazy_import_cache_calls(getattr_fn: ast.FunctionDef) -> list[ast.Call]:
    """Return the ``globals()`` calls of ``globals()[<param>] = ...`` directly inside ``getattr_fn``."""
    params = getattr_fn.args.posonlyargs + getattr_fn.args.args
    if not params:
        return []
    key = params[0].arg
    calls = []
    for node in _walk_scope(getattr_fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and _is_call_to(target.value, "globals")
                and not target.value.args
                and not target.value.keywords
                and isinstance(target.slice, ast.Name)
                and target.slice.id == key
            ):
                calls.append(target.value)
    return calls


class _NamespaceMutationVisitor(ast.NodeVisitor):
    """Collect module-namespace mutations that are not PEP 562 lazy-import caches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[LintError] = []
        self._allowed: set[ast.Call] = set()
        self._scope_depth = 0

    def _report(self, node: ast.AST, what: str) -> None:
        self.errors.append(LintError(self.path, node.lineno, node.col_offset, f"banned use of {what}"))

    def _visit_scope(self, node: ast.AST) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Whitelist the lazy-import caches of a module-level ``__getattr__``."""
        if node.name == "__getattr__" and self._scope_depth == 0:
            self._allowed.update(_lazy_import_cache_calls(node))
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """``async def __getattr__`` is not a PEP 562 hook, so nothing inside is exempt."""
        self._visit_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """A class ``__getattr__`` is not the module-level lazy-import hook."""
        self._visit_scope(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Flag ``globals()``, module-scope ``vars()`` and ``setattr``/``delattr`` on the module itself."""
        if _is_call_to(node, "globals"):
            if node not in self._allowed:
                self._report(node, "globals()")
        elif _is_call_to(node, "vars"):
            if not node.args and not node.keywords and self._scope_depth == 0:
                self._report(node, "vars() at module scope")
            elif len(node.args) == 1 and _is_own_module(node.args[0]):
                self._report(node, "vars(sys.modules[__name__])")
        elif _is_call_to(node, "setattr", "delattr") and node.args and _is_own_module(node.args[0]):
            self._report(node, f"{node.func.id}(sys.modules[__name__], ...)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Flag ``sys.modules[__name__].__dict__``."""
        if node.attr == "__dict__" and _is_own_module(node.value):
            self._report(node, "sys.modules[__name__].__dict__")
        self.generic_visit(node)


def lint_source(source: str | bytes, path: Path) -> list[LintError]:
    """Lint Python source as if it came from ``path``.

    Pass ``bytes`` when reading from disk so ``ast`` honors BOMs and ``# -*- coding -*-``
    cookies. A file that does not parse is reported, not skipped.
    """
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as exc:  # ValueError: null bytes on Python < 3.12
        line = getattr(exc, "lineno", None) or 1
        return [LintError(path, line, 0, f"cannot parse file: {exc.__class__.__name__}: {exc}")]
    visitor = _NamespaceMutationVisitor(path)
    visitor.visit(tree)
    return visitor.errors


def lint_file(path: Path) -> list[LintError]:
    """Lint a single Python file."""
    return lint_source(path.read_bytes(), path)


def collect_python_files(paths: list[Path]) -> list[Path]:
    """Expand files and directories into the Python files to lint."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if p.is_file()))
        elif path.suffix == ".py":
            files.append(path)
    return files


def format_errors(errors: list[LintError], automodel_dir: Path) -> str:
    """Format lint errors for CLI output."""
    lines = [f"{_relative_path(e.path, automodel_dir)}:{e.line}:{e.col + 1}: {e.message}" for e in errors]
    if any(e.message.startswith("banned use of") for e in errors):
        lines += ["", HINT]
    return "\n".join(lines)


def _relative_path(path: Path, automodel_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(automodel_dir.resolve())
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    """Run the linter. Exit 1 on violations, 2 when there was nothing valid to lint."""
    parser = argparse.ArgumentParser(description="Reject globals() and equivalent module-namespace mutation.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Files or directories to lint. Defaults to: {', '.join(DEFAULT_PATHS)}.",
    )
    parser.add_argument("--automodel-dir", type=Path, default=Path.cwd(), help="Path to the AutoModel repository root.")
    args = parser.parse_args(argv)

    automodel_dir = args.automodel_dir.resolve()
    if args.paths:
        paths = [path.resolve() for path in args.paths]
        missing = [path for path in paths if not path.exists()]
        if missing:
            print("error: no such path(s): " + ", ".join(str(path) for path in missing), file=sys.stderr)
            return 2
    else:
        if not (automodel_dir / "nemo_automodel").is_dir():
            print(
                f"error: {automodel_dir} is not an AutoModel checkout (no nemo_automodel/); pass --automodel-dir",
                file=sys.stderr,
            )
            return 2
        paths = [automodel_dir / name for name in DEFAULT_PATHS if (automodel_dir / name).exists()]

    files = collect_python_files(paths)
    if not files:
        print("error: no Python files to lint under " + ", ".join(str(path) for path in paths), file=sys.stderr)
        return 2

    errors = [error for path in files for error in lint_file(path)]
    if errors:
        print(format_errors(errors, automodel_dir), file=sys.stderr)
        return 1

    print(f"Linted {len(files)} Python file(s) for module-namespace mutation.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
