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

"""Static analysis for imports through Automodel component interfaces."""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComponentImport:
    """An import of a configured component from outside that component."""

    path: Path
    line_number: int
    importer: str
    target_component: str
    imported_module: str
    imported_names: tuple[str, ...]
    violation: str | None

    @property
    def is_public(self) -> bool:
        """Return whether the import uses only the target component's public API."""
        return self.violation is None


def find_component_imports(
    project_root: Path,
    component_modules: set[str],
) -> list[ComponentImport]:
    """Find runtime imports of configured components from outside their package."""
    components = sorted(component_modules, key=lambda module: (-len(module), module))
    root_packages = {component.partition(".")[0] for component in components}
    if len(root_packages) != 1:
        raise ValueError("Configured components must share one root package")

    root_package = root_packages.pop()
    source_root = project_root / root_package
    if not source_root.is_dir():
        raise ValueError(f"Root package does not exist: {root_package}")

    exports = {
        component: _read_exports(project_root / Path(*component.split(".")) / "__init__.py") for component in components
    }
    imports: list[ComponentImport] = []

    for component in components:
        component_path = project_root / Path(*component.split("."))
        if not component_path.is_dir():
            raise ValueError(f"Component package does not exist: {component}")

    for path in sorted(source_root.rglob("*.py")):
        importer = _module_name(project_root, path)
        package = importer if path.name == "__init__.py" else importer.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ComponentImportVisitor(
            path=path.relative_to(project_root),
            importer=importer,
            importer_package=package,
            source_component=_component_for_module(importer, components),
            components=components,
            exports=exports,
        )
        visitor.visit(tree)
        imports.extend(visitor.imports)

    return sorted(imports, key=lambda item: (str(item.path), item.line_number, item.target_component))


class _ComponentImportVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: Path,
        importer: str,
        importer_package: str,
        source_component: str | None,
        components: list[str],
        exports: dict[str, frozenset[str]],
    ) -> None:
        self.path = path
        self.importer = importer
        self.importer_package = importer_package
        self.source_component = source_component
        self.components = components
        self.exports = exports
        self.imports: list[ComponentImport] = []

    def visit_If(self, node: ast.If) -> None:
        type_checking_value = _evaluate_type_checking_guard(node.test)
        if type_checking_value is False:
            for statement in node.orelse:
                self.visit(statement)
            return
        if type_checking_value is True:
            for statement in node.body:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target_component = _component_for_module(alias.name, self.components)
            if target_component is None or target_component == self.source_component:
                continue
            self.imports.append(
                ComponentImport(
                    path=self.path,
                    line_number=node.lineno,
                    importer=self.importer,
                    target_component=target_component,
                    imported_module=alias.name,
                    imported_names=("<module>",),
                    violation=(f"component imports must name symbols from {target_component}.__all__"),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = _resolve_import_from(node, self.importer_package)
        if imported_module is None:
            return

        target_component = _component_for_module(imported_module, self.components)
        if target_component is not None:
            if target_component != self.source_component:
                self._record_from_import(node, imported_module, target_component)
            return

        for alias in node.names:
            possible_module = f"{imported_module}.{alias.name}"
            target_component = _component_for_module(possible_module, self.components)
            if target_component is None or target_component == self.source_component:
                continue
            self.imports.append(
                ComponentImport(
                    path=self.path,
                    line_number=node.lineno,
                    importer=self.importer,
                    target_component=target_component,
                    imported_module=possible_module,
                    imported_names=("<module>",),
                    violation=(f"component imports must name symbols from {target_component}.__all__"),
                )
            )

    def _record_from_import(
        self,
        node: ast.ImportFrom,
        imported_module: str,
        target_component: str,
    ) -> None:
        imported_names = tuple(alias.name for alias in node.names)
        if imported_module != target_component:
            violation = f"{imported_module} is private; import exported symbols from {target_component}"
        else:
            missing_exports = sorted(
                name for name in imported_names if name == "*" or name not in self.exports[target_component]
            )
            violation = None
            if missing_exports:
                names = ", ".join(missing_exports)
                violation = f"{names} not exported by {target_component}.__all__"

        self.imports.append(
            ComponentImport(
                path=self.path,
                line_number=node.lineno,
                importer=self.importer,
                target_component=target_component,
                imported_module=imported_module,
                imported_names=imported_names,
                violation=violation,
            )
        )


def _module_name(project_root: Path, path: Path) -> str:
    relative = path.relative_to(project_root).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _component_for_module(module: str, components: list[str]) -> str | None:
    for component in components:
        if module == component or module.startswith(f"{component}."):
            return component
    return None


def _resolve_import_from(node: ast.ImportFrom, importer_package: str) -> str | None:
    if node.level == 0:
        return node.module
    relative_name = f"{'.' * node.level}{node.module or ''}"
    try:
        return importlib.util.resolve_name(relative_name, importer_package)
    except ImportError:
        return None


def _evaluate_type_checking_guard(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
        return False
    if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _evaluate_type_checking_guard(node.operand)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_type_checking_guard(value) for value in node.values]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(value is True for value in values) else None
        if True in values:
            return True
        return False if all(value is False for value in values) else None
    return None


def _read_exports(init_path: Path) -> frozenset[str]:
    if not init_path.is_file():
        raise ValueError(f"Component package is missing __init__.py: {init_path}")

    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    values: dict[str, object] = {}
    exports: set[str] = set()

    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
            value_node = statement.value
            if isinstance(target, ast.Name) and value_node is not None:
                value = _static_value(value_node, values)
                if value is not _UNKNOWN:
                    values[target.id] = value
                    if target.id == "__all__":
                        exports = _string_set(value)
        elif (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
            and isinstance(statement.op, ast.Add)
        ):
            value = _static_value(statement.value, values)
            if value is not _UNKNOWN:
                exports.update(_string_set(value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "__all__":
            continue
        if node.func.attr == "append" and len(node.args) == 1:
            value = _static_value(node.args[0], values)
            if isinstance(value, str):
                exports.add(value)
        elif node.func.attr == "extend" and len(node.args) == 1:
            value = _static_value(node.args[0], values)
            if value is not _UNKNOWN:
                exports.update(_string_set(value))

    return frozenset(exports)


_UNKNOWN = object()


def _static_value(node: ast.AST, values: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items: list[object] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                value = _static_value(element.value, values)
                if not isinstance(value, (list, tuple, set, frozenset)):
                    return _UNKNOWN
                items.extend(value)
            else:
                value = _static_value(element, values)
                if value is _UNKNOWN:
                    return _UNKNOWN
                items.append(value)
        return items
    if isinstance(node, ast.Dict):
        result: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                return _UNKNOWN
            key = _static_value(key_node, values)
            value = _static_value(value_node, values)
            if key is _UNKNOWN or value is _UNKNOWN:
                return _UNKNOWN
            result[key] = value
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_value(node.left, values)
        right = _static_value(node.right, values)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return [*left, *right]
        return _UNKNOWN
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        value = _static_value(node.args[0], values)
        if isinstance(node.func, ast.Name) and node.func.id in {"list", "set", "sorted", "tuple"}:
            if not isinstance(value, (list, tuple, set, frozenset, dict)):
                return _UNKNOWN
            iterable = value.keys() if isinstance(value, dict) else value
            if node.func.id == "sorted":
                return sorted(iterable)
            return list(iterable)
    if (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "keys"
    ):
        value = _static_value(node.func.value, values)
        if isinstance(value, dict):
            return list(value)
    return _UNKNOWN


def _string_set(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {item for item in value if isinstance(item, str)}
