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

"""Require literal ``allow_pickle=False`` on statically named NumPy load/save calls.

Recognizes module imports (including aliases) and direct function imports. This
is a syntax check, not runtime data-flow analysis: dynamically obtained callables
and aliases assigned through arbitrary expressions are outside its scope.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_PATHS = ("app.py", "nemo_automodel", "examples", "scripts", "tools", "tutorials", "tests")


def lint_source(source: str | bytes, path: Path) -> list[str]:
    """Return location-bearing diagnostics for unsafe NumPy calls or invalid syntax."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno or 1}:{exc.offset or 1}: cannot parse file: {exc.msg}"]
    except ValueError as exc:
        return [f"{path}:1:1: cannot parse file: {exc}"]

    # Conservatively retain every NumPy import spelling in the file, including
    # conditional and function-local imports; rebinding cannot silence the rule.
    modules: set[str] = set()
    functions: dict[str, str] = {}
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy" and node.level == 0:
            for alias in node.names:
                if alias.name in {"load", "save"}:
                    functions[alias.asname or alias.name] = alias.name
                elif alias.name == "*":
                    errors.append(
                        f"{path}:{node.lineno}:{node.col_offset + 1}: "
                        "use explicit NumPy imports so allow_pickle can be checked"
                    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        operation = None
        if isinstance(function, ast.Name):
            operation = functions.get(function.id)
        elif (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in modules
            and function.attr in {"load", "save"}
        ):
            operation = function.attr
        if operation is None:
            continue
        if not any(
            keyword.arg == "allow_pickle" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            for keyword in node.keywords
        ):
            errors.append(
                f"{path}:{node.lineno}:{node.col_offset + 1}: numpy.{operation} requires explicit allow_pickle=False"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Lint Python paths; return 1 for violations and 2 for invalid input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Python files or directories to check.")
    args = parser.parse_args(argv)
    paths = args.paths or [Path(name) for name in DEFAULT_PATHS if Path(name).exists()]
    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            print(f"error: no such path: {path}", file=sys.stderr)
            return 2
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.suffix == ".py":
            files.add(path)
    if not files:
        print("error: no Python files to lint", file=sys.stderr)
        return 2
    errors = [error for path in sorted(files) for error in lint_source(path.read_bytes(), path)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
