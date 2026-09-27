# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Enforce the single MeshContext model-parallelizer boundary."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

DEFAULT_PATHS = (Path("nemo_automodel"), Path("examples"))
FORBIDDEN_NAMES = {"ParallelizeContext", "parallel_scheme"}
MODEL_LAYERS = ("nemo_automodel.components.models", "nemo_automodel._diffusers", "nemo_automodel._transformers")


class LintError(NamedTuple):
    """One contract violation."""

    path: Path
    line: int
    message: str


def _identifiers(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword):
        return [node.arg] if node.arg else []
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [name for alias in node.names for name in (alias.name.rsplit(".", 1)[-1], alias.asname) if name]
    return []


def _type_only_nodes(tree: ast.AST) -> set[int]:
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        is_type_checking = (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
            isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
        )
        if is_type_checking:
            result.update(id(child) for statement in node.body for child in ast.walk(statement))
    return result


def _imports_model_layer(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.startswith(MODEL_LAYERS) for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return False
    module = node.module or ""
    return module.startswith(MODEL_LAYERS) or (
        node.level > 0 and module.split(".", 1)[0] in {"models", "_diffusers", "_transformers"}
    )


def lint_source(source: str, path: Path = Path("<string>")) -> list[LintError]:
    """Check forbidden names and reverse runtime imports."""

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [LintError(path, error.lineno or 1, f"cannot parse Python source: {error.msg}")]

    errors = []
    for node in ast.walk(tree):
        for identifier in _identifiers(node):
            if identifier in FORBIDDEN_NAMES:
                errors.append(
                    LintError(
                        path,
                        getattr(node, "lineno", 1),
                        f"{identifier} creates a second model-parallelization interface; use MeshContext only",
                    )
                )
    if "nemo_automodel/components/distributed" in path.as_posix():
        type_only = _type_only_nodes(tree)
        errors.extend(
            LintError(path, node.lineno, "distributed infrastructure may not import model or adapter implementations")
            for node in ast.walk(tree)
            if id(node) not in type_only and _imports_model_layer(node)
        )
    return errors


def _module_path(module: str, root: Path) -> Path:
    relative = Path(*module.split("."))
    package = root / relative / "__init__.py"
    return package if package.is_file() else root / relative.with_suffix(".py")


def _exports(path: Path) -> set[str] | None:
    if not path.is_file():
        return None
    for node in ast.parse(path.read_bytes(), filename=str(path)).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
            return (
                set(value)
                if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
                else None
            )
    return None


def lint_sidecar_exports(source: str, path: Path, root: Path) -> list[LintError]:
    """Check that model sidecars consume only exported distributed symbols."""

    if "nemo_automodel/components/models/" not in path.as_posix():
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    nodes = list(ast.walk(tree))
    if not any("ModelParallelizer" in _identifiers(node) for node in nodes):
        return []

    errors = []
    for node in nodes:
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("nemo_automodel.components.distributed") for alias in node.names
        ):
            errors.append(
                LintError(path, node.lineno, "model sidecars must import named public distributed symbols, not modules")
            )
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            "nemo_automodel.components.distributed"
        ):
            continue
        exported = _exports(_module_path(node.module, root))
        errors.extend(
            LintError(
                path,
                node.lineno,
                f"model sidecars may import only distributed symbols declared in __all__; {alias.name} is not exported",
            )
            for alias in node.names
            if exported is None or alias.name not in exported
        )
    return errors


def collect_python_files(paths: Iterable[Path]) -> list[Path]:
    """Expand input paths to Python files."""

    return sorted({file for path in paths for file in ([path] if path.is_file() else path.rglob("*.py"))})


def main(argv: list[str] | None = None) -> int:
    """Run the contract linter."""

    paths = [Path(arg) for arg in argv] if argv else list(DEFAULT_PATHS)
    errors = []
    for path in collect_python_files(paths):
        source = path.read_text(encoding="utf-8")
        errors.extend(lint_source(source, path))
        errors.extend(lint_sidecar_exports(source, path, Path.cwd()))
    for error in errors:
        print(f"{error.path}:{error.line}: {error.message}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
