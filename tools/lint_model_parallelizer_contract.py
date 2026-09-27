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
SIDECAR_PATHS = ("nemo_automodel/components/models/", "nemo_automodel/_diffusers/")


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


def lint_source(source: str, path: Path = Path("<string>")) -> list[LintError]:
    """Check names that would create a second parallelization contract."""

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
    if path.as_posix().endswith("nemo_automodel/components/distributed/parallelizer.py"):
        parallelizer = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelParallelizer"),
            None,
        )
        methods = (
            [
                node
                for node in parallelizer.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "parallelize"
            ]
            if parallelizer
            else []
        )
        if len(methods) != 1 or _argument_names(methods[0].args) != ["self", "model", "mesh_context"]:
            errors.append(
                LintError(
                    path,
                    methods[0].lineno if methods else parallelizer.lineno if parallelizer else 1,
                    "ModelParallelizer must expose exactly parallelize(self, model, mesh_context)",
                )
            )
    return errors


def _argument_names(arguments: ast.arguments) -> list[str] | None:
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs or arguments.defaults:
        return None
    return [argument.arg for argument in (*arguments.posonlyargs, *arguments.args)]


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
    """Check that sidecars keep one interface and consume only public infrastructure."""

    if not any(prefix in path.as_posix() for prefix in SIDECAR_PATHS):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    nodes = list(ast.walk(tree))
    parallelizer_names = {"ModelParallelizer"}
    parallelizer_names.update(
        alias.asname
        for node in nodes
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nemo_automodel.components.distributed")
        for alias in node.names
        if alias.name == "ModelParallelizer" and alias.asname
    )
    sidecar_classes = [
        node
        for node in nodes
        if isinstance(node, ast.ClassDef)
        and any(parallelizer_names.intersection(_identifiers(base)) for base in node.bases)
    ]
    if not sidecar_classes:
        return []

    errors = [
        LintError(
            path,
            method.lineno,
            "model sidecars must use the inherited parallelize(model, mesh_context) interface",
        )
        for class_node in sidecar_classes
        for method in class_node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (method.name == "parallelize" or method.name.startswith("parallelize_"))
    ]
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
