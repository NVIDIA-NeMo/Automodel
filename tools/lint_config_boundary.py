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

"""Check components for ConfigNode use that the import contracts cannot see.

CB001–CB006 cover config types, node-only APIs, reflection, untyped config
parameters, structural substitutes, and dotted YAML lookups. Deprecated shims
are allowed by file and function until removed.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

COMPONENTS = Path("nemo_automodel/components")

CONFIG_TYPES = frozenset({"ConfigNode", "RecipeConfig"})
CONFIG_NODE_ONLY_METHODS = frozenset(
    {"to_yaml_dict", "instantiate", "instantiate_path", "get_as_string", "set_by_dotted"}
)
CONFIG_NODE_ONLY_ATTRIBUTES = frozenset({"raw_config"})
PROBED_ATTRIBUTES = frozenset(
    {"to_dict", "to_yaml_dict", "instantiate", "instantiate_path", "_target_", "raw_config", "get_as_string"}
)
PROTOCOL_LOOKUP_METHODS = frozenset({"get", "to_dict", "to_yaml_dict", "instantiate", "__getattr__"})

# (path relative to the repository root, enclosing function) -> why the shim may still read a raw config.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "nemo_automodel/components/utils/compile_utils.py",
        "_compile_config_from_mapping",
    ): "Used by the deprecated compile helpers to accept nested options; delete this entry with the helpers.",
    (
        "nemo_automodel/components/loggers/mlflow_utils.py",
        "configure_mlflow",
    ): "Back-compat shim for MLflowConfig.build (deprecation in #4947); delete this entry with the shim.",
    (
        "nemo_automodel/components/loggers/comet_utils.py",
        "build_comet",
    ): "Deprecated in favor of CometConfig.build; delete this entry with the shim.",
}

HINT = (
    "Components take typed configs and typed runtime arguments, never a YAML ConfigNode or a stand-in "
    "for one (CODING_GUIDELINES.md, 'Stop YAML at the typed-config boundary'). Read the YAML in the "
    "recipe layer (RecipeConfig in nemo_automodel/recipes/_typed_config.py), build the component's "
    "typed config there, and pass that in."
)


@dataclass(frozen=True)
class LintError:
    """A lint failure with source location."""

    path: Path
    line: int
    col: int
    code: str
    message: str
    function: str = ""


def _is_config_param_name(name: str) -> bool:
    return name == "cfg" or name.endswith("_cfg") or name.startswith("cfg_")


def _name(node: ast.AST) -> str:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return ""


def _annotation_names(node: ast.expr) -> Iterator[str]:
    """Yield the names an annotation mentions, parsing string (forward-reference) annotations."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                parsed = ast.parse(child.value, mode="eval")
            except SyntaxError:
                continue
            yield from _annotation_names(parsed.body)


def _is_any(annotation: ast.expr | None) -> bool:
    """True when ``annotation`` is ``Any``, optionally ``| None`` / ``Optional[...]`` around it."""
    match annotation:
        case ast.Constant(value=str(source)):
            try:
                return _is_any(ast.parse(source, mode="eval").body)
            except SyntaxError:
                return False
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            parts = (left, right)
            return any(_is_any(part) for part in parts) and all(
                _is_any(part) or isinstance(part, ast.Constant) and part.value is None for part in parts
            )
        case ast.Subscript(value=base, slice=inner) if _name(base) == "Optional":
            return _is_any(inner)
        case ast.expr():
            return _name(annotation) == "Any"
    return False


class _ConfigBoundaryVisitor(ast.NodeVisitor):
    """Collect ConfigNode handling, tagged with the enclosing top-level function or method."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[LintError] = []
        self._scope: list[str] = []

    def _report(self, node: ast.expr | ast.stmt | ast.arg, code: str, message: str) -> None:
        self.errors.append(LintError(self.path, node.lineno, node.col_offset, code, message, ".".join(self._scope)))

    def _visit_arguments(self, args: ast.arguments) -> None:
        positional = args.posonlyargs + args.args
        defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        candidates = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
        for arg in (args.vararg, args.kwarg):
            if arg is not None and arg.annotation is not None:
                self._check_annotation(arg.annotation)
        for arg, default in candidates:
            if default is not None:
                self.visit(default)
            if arg.annotation is not None:
                self._check_annotation(arg.annotation)
            if not _is_config_param_name(arg.arg):
                continue
            # A literal default (``enable_cfg=True``) marks a flag, not a config object.
            if arg.annotation is None and isinstance(default, ast.Constant) and default.value is not None:
                continue
            if arg.annotation is None:
                self._report(arg, "CB004", f"config parameter `{arg.arg}` has no type; take a typed config instead")
            elif _is_any(arg.annotation):
                self._report(arg, "CB004", f"config parameter `{arg.arg}` is typed Any; take a typed config instead")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append(node.name)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self._check_annotation(node.returns)
        for child in node.body:
            self.visit(child)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the function scope and check its parameters."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track the function scope and check its parameters."""
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Check lambda parameters too."""
        self._visit_arguments(node.args)
        self.visit(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Flag classes named after ConfigNode and Protocols that mirror its lookup surface."""
        if "ConfigNode" in node.name:
            self._report(node, "CB001", f"class `{node.name}` stands in for ConfigNode")
        if any(_name(base.value if isinstance(base, ast.Subscript) else base) == "Protocol" for base in node.bases):
            members = sorted(
                stmt.name
                for stmt in node.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name in PROTOCOL_LOOKUP_METHODS
            )
            if members:
                self._report(
                    node,
                    "CB005",
                    f"Protocol `{node.name}` declares {', '.join(members)}, the lookup surface of a YAML config node",
                )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _check_annotation(self, annotation: ast.expr) -> None:
        names = set(_annotation_names(annotation)) & CONFIG_TYPES
        for name in sorted(names):
            self._report(annotation, "CB001", f"annotation names `{name}`")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check variable and attribute annotations."""
        self._check_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        """Flag uses of the config types."""
        if node.id in CONFIG_TYPES:
            self._report(node, "CB001", f"reference to `{node.id}`")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Flag ``x.ConfigNode`` and reads of ConfigNode-only attributes."""
        if node.attr in CONFIG_TYPES:
            self._report(node, "CB001", f"reference to `{node.attr}`")
        elif node.attr in CONFIG_NODE_ONLY_ATTRIBUTES:
            self._report(node, "CB002", f"`.{node.attr}` is ConfigNode API")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Flag imports of the config types (import-linter also rejects the module edge)."""
        for alias in node.names:
            if alias.name in CONFIG_TYPES:
                self._report(node, "CB001", f"import of `{alias.name}`")

    def visit_Call(self, node: ast.Call) -> None:
        """Flag ConfigNode-only method calls, reflection probes for its API and dotted-key lookups."""
        match node:
            case ast.Call(func=ast.Attribute(attr=method)) if method in CONFIG_NODE_ONLY_METHODS:
                self._report(node, "CB002", f"`.{method}()` is ConfigNode API")
            case ast.Call(
                func=ast.Name(id="hasattr" | "getattr" as probe), args=[_, ast.Constant(value=str(attr)), *_]
            ):
                if attr in PROBED_ATTRIBUTES:
                    self._report(node, "CB003", f'`{probe}(..., "{attr}")` probes for a config node')
            case ast.Call(func=ast.Attribute(value=receiver, attr="get"), args=[ast.Constant(value=str(key)), *_]):
                name = _name(receiver)
                if (
                    "." in key
                    and " " not in key
                    and (_is_config_param_name(name) or name == "config" or name.endswith("_config"))
                ):
                    self._report(node, "CB006", f'`.get("{key}")` resolves a dotted YAML path')
        self.generic_visit(node)


def lint_source(source: str | bytes, path: Path) -> list[LintError]:
    """Lint Python source as if it came from ``path``. A file that does not parse is reported."""
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as exc:  # ValueError: null bytes on Python < 3.12
        line = getattr(exc, "lineno", None) or 1
        return [LintError(path, line, 0, "CB000", f"cannot parse file: {exc.__class__.__name__}: {exc}")]
    visitor = _ConfigBoundaryVisitor(path)
    visitor.visit(tree)
    return visitor.errors


def _relative(path: Path, automodel_dir: Path) -> str:
    try:
        return path.resolve().relative_to(automodel_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def collect_python_files(paths: list[Path], automodel_dir: Path) -> list[Path]:
    """Expand paths to component Python files, excluding the config package."""
    components = automodel_dir.resolve() / COMPONENTS
    files: list[Path] = []
    for path in paths:
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                candidate.suffix == ".py"
                and candidate.is_file()
                and resolved.is_relative_to(components)
                and not resolved.is_relative_to(components / "config")
            ):
                files.append(candidate)
    return files


def check(files: list[Path], automodel_dir: Path, *, report_stale: bool) -> list[str]:
    """Lint ``files``; return formatted findings plus stale allowlist entries."""
    errors = [error for path in files for error in lint_source(path.read_bytes(), path)]
    used: set[tuple[str, str]] = set()
    lines: list[str] = []
    for error in errors:
        key = (_relative(error.path, automodel_dir), error.function.partition(".")[0])
        if key in ALLOWED:
            used.add(key)
            continue
        where = f" (in `{error.function}`)" if error.function else ""
        lines.append(
            f"{_relative(error.path, automodel_dir)}:{error.line}:{error.col + 1}: {error.code} {error.message}{where}"
        )
    if report_stale:
        for key in sorted(set(ALLOWED) - used):
            lines.append(f"{key[0]}: stale ALLOWED entry for `{key[1]}`; remove it from tools/lint_config_boundary.py")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the linter. Exit 1 on violations, 2 when there was nothing valid to lint."""
    parser = argparse.ArgumentParser(description="Reject YAML ConfigNode handling inside components.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Files or directories to lint. Defaults to {COMPONENTS}.",
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
        paths = [automodel_dir / COMPONENTS]

    files = collect_python_files(paths, automodel_dir)
    if not files:
        if args.paths:
            # pre-commit passes only the changed files; none of them is a component.
            return 0
        print("error: no Python files to lint under " + ", ".join(str(path) for path in paths), file=sys.stderr)
        return 2

    # Stale entries can only be judged on a full run; pre-commit passes a subset of files.
    lines = check(files, automodel_dir, report_stale=not args.paths)
    if lines:
        print("\n".join([*lines, "", HINT]), file=sys.stderr)
        return 1

    print(f"Linted {len(files)} component file(s) for YAML config handling.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
