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

    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if len(targets) != 1 and any(_assigns_name(target, "__all__") for target in targets):
                _unsupported_all(init_path, statement, "chained assignment")
            value_node = statement.value
            value = _static_value(value_node, values) if value_node is not None else _UNKNOWN
            for target in targets:
                if not isinstance(target, ast.Name):
                    if _assigns_name(target, "__all__"):
                        _unsupported_all(init_path, statement, "assignment")
                    continue
                if value is _UNKNOWN:
                    values.pop(target.id, None)
                    if target.id in {"__all__", "_LAZY_ATTRS"}:
                        _unsupported_all(init_path, statement, f"{target.id} value")
                else:
                    values[target.id] = value
                    if target.id == "__all__":
                        _string_list(init_path, statement, value)
                    elif target.id == "_LAZY_ATTRS" and not isinstance(value, dict):
                        _unsupported_all(init_path, statement, "_LAZY_ATTRS value")
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id != "__all__":
                continue
            if not isinstance(statement.op, ast.Add):
                _unsupported_all(init_path, statement, "augmented assignment")
            current = values.get("__all__", _UNKNOWN)
            value = _static_value(statement.value, values)
            if current is _UNKNOWN or value is _UNKNOWN:
                _unsupported_all(init_path, statement, "augmented assignment")
            if not isinstance(current, list):
                _unsupported_all(init_path, statement, "augmented assignment target")
            current.extend(_string_list(init_path, statement, value))
        elif isinstance(statement, ast.Expr) and _is_all_call(statement.value):
            _apply_all_call(init_path, statement, statement.value, values)
        elif isinstance(
            statement, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
        ):
            if _contains_all_mutation(statement):
                _unsupported_all(init_path, statement, "conditional or nested mutation")
        elif isinstance(statement, ast.Delete) and any(
            _assigns_name(target, "__all__") for target in statement.targets
        ):
            _unsupported_all(init_path, statement, "deletion")

    exports = _string_list(init_path, tree, values.get("__all__", []))
    lazy_attrs = values.get("_LAZY_ATTRS")
    if isinstance(lazy_attrs, dict):
        lazy_exports = {key for key in lazy_attrs if isinstance(key, str)}
        typed_exports = _read_type_checking_exports(tree)
        if lazy_exports != typed_exports:
            missing = ", ".join(sorted(lazy_exports - typed_exports)) or "none"
            extra = ", ".join(sorted(typed_exports - lazy_exports)) or "none"
            lazy_assignment = next(
                statement
                for statement in tree.body
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
                and any(_assigns_name(target, "_LAZY_ATTRS") for target in _assignment_targets(statement))
            )
            raise ValueError(
                f"{init_path}:{lazy_assignment.lineno}: TYPE_CHECKING exports do not match _LAZY_ATTRS "
                f"(missing: {missing}; extra: {extra})"
            )

    return frozenset(exports)


def _assignment_targets(statement: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    return statement.targets if isinstance(statement, ast.Assign) else [statement.target]


def _assigns_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _is_all_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "__all__"
    )


def _apply_all_call(init_path: Path, statement: ast.stmt, call: ast.Call, values: dict[str, object]) -> None:
    current = values.get("__all__", _UNKNOWN)
    if current is _UNKNOWN or call.keywords or len(call.args) != 1:
        _unsupported_all(init_path, statement, "method call")
    if not isinstance(current, list):
        _unsupported_all(init_path, statement, "method call target")
    _string_list(init_path, statement, current)
    if call.func.attr == "append":
        value = _static_value(call.args[0], values)
        if not isinstance(value, str):
            _unsupported_all(init_path, statement, "append value")
        current.append(value)
    elif call.func.attr == "extend":
        value = _static_value(call.args[0], values)
        if value is _UNKNOWN:
            _unsupported_all(init_path, statement, "extend value")
        current.extend(_string_list(init_path, statement, value))
    else:
        _unsupported_all(init_path, statement, f"method {call.func.attr!r}")


def _contains_all_mutation(node: ast.AST) -> bool:
    class MutationFinder(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == "__all__" and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.found = True

        def visit_Call(self, node: ast.Call) -> None:
            if _is_all_call(node):
                self.found = True
                return
            self.generic_visit(node)

    finder = MutationFinder()
    finder.visit(node)
    return finder.found


def _read_type_checking_exports(tree: ast.Module) -> set[str]:
    exports: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, ast.If) or not _is_type_checking_name(statement.test):
            continue
        for declaration in statement.body:
            if isinstance(declaration, ast.ImportFrom):
                exports.update(alias.asname or alias.name for alias in declaration.names if alias.name != "*")
    return exports


def _is_type_checking_name(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


def _unsupported_all(init_path: Path, node: ast.AST, shape: str) -> None:
    raise ValueError(f"{init_path}:{node.lineno}: unsupported __all__ {shape}")


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


def _string_list(init_path: Path, node: ast.AST, value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)) or not all(isinstance(item, str) for item in value):
        _unsupported_all(init_path, node, "value")
    return list(value)
