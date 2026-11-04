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

"""Reject unapproved object annotations without importing the checked code.

Run from the repository root with Python 3.11+. Configuration and the legacy
inventory live in tools/annotation_policy.toml. This syntax gate complements ty;
it does not infer types or resolve aliases imported from other source modules.
"""

from __future__ import annotations

import ast
import logging
import tomllib
from collections import Counter
from pathlib import Path

_LOG = logging.getLogger(__name__)


def _bindings(body: list[ast.stmt]) -> dict[str, ast.expr]:
    # Collect bindings in this lexical scope, including TYPE_CHECKING blocks.
    bindings: dict[str, ast.expr] = {}
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] = ast.Constant(None)
            continue
        if isinstance(node, ast.Import):
            for item in node.names:
                bindings[item.asname or item.name.split(".")[0]] = ast.Name(item.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                bindings[item.asname or item.name] = ast.Attribute(ast.Name(node.module or ""), item.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            bindings[node.target.id] = node.value
        pending[0:0] = list(ast.iter_child_nodes(node))
    return bindings


def _name(node: ast.expr, bindings: dict[str, ast.expr], seen: frozenset[str] = frozenset()) -> str:
    if isinstance(node, ast.Name):
        if node.id in bindings and node.id not in seen:
            return _name(bindings[node.id], bindings, seen | {node.id})
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value, bindings, seen)}.{node.attr}"
    return ""


def _has_object(node: ast.expr, bindings: dict[str, ast.expr], seen: frozenset[str] = frozenset()) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False  # ty diagnoses invalid forward references.
        return _has_object(parsed, bindings, seen)
    if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
        return _has_object(bindings[node.id], bindings, seen | {node.id})
    if _name(node, bindings, seen) in {"object", "builtins.object"}:
        return True
    if isinstance(node, ast.Subscript):
        name = _name(node.value, bindings)
        if name in {"typing.Literal", "typing_extensions.Literal"}:
            return False
        elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name in {"typing.Annotated", "typing_extensions.Annotated"}:
            elements = elements[:1]
        return _has_object(node.value, bindings, seen) or any(_has_object(e, bindings, seen) for e in elements)
    return any(
        _has_object(child, bindings, seen) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.bindings = _bindings(tree.body)
        self.scope: list[str] = []
        self.findings: list[tuple[str, int]] = []

    def _check(self, node: ast.expr | None, slot: str) -> None:
        if node is not None and _has_object(node, self.bindings):
            self.findings.append((f"{'.'.join(self.scope) or '<module>'}:{slot}", node.lineno))

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        for arg in args:
            self._check(arg.annotation, arg.arg)
        self._check(node.returns, "return")
        if node.type_comment:
            comment = ast.parse(node.type_comment, mode="func_type")
            ast.increment_lineno(comment, node.lineno - 1)
            for index, annotation in enumerate(comment.argtypes):
                self._check(annotation, f"comment-arg[{index}]")
            self._check(comment.returns, "comment-return")
        previous = self.bindings
        self.bindings = previous | {arg.arg: ast.Constant(None) for arg in args} | _bindings(node.body)
        for statement in node.body:
            self.visit(statement)
        self.bindings = previous
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        previous = self.bindings
        self.bindings = previous | _bindings(node.body)
        for index, base in enumerate(node.bases):
            # Inheriting from object is a runtime use, not an annotation.
            if isinstance(base, ast.Subscript):
                self._check(base, f"base[{index}]")
        for statement in node.body:
            self.visit(statement)
        self.bindings = previous
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # Explicit alias expressions must not hide object from importing modules.
        # Strings and calls here are runtime values, not declared type aliases.
        if node.type_comment:
            annotation = ast.parse(node.type_comment, mode="eval").body
            ast.increment_lineno(annotation, node.lineno - 1)
            self._check(annotation, f"comment[{ast.unparse(node.targets[0])}]")
        value = node.value
        alias = (
            isinstance(value, (ast.Name, ast.Attribute))
            or (
                isinstance(value, ast.Subscript)
                and (
                    _name(value.value, self.bindings) in {"list", "dict", "tuple", "set", "frozenset", "type"}
                    or _name(value.value, self.bindings).startswith(
                        ("typing.", "typing_extensions.", "collections.abc.")
                    )
                )
            )
            or (isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr))
        )
        if alias and not any(isinstance(child, ast.Call) for child in ast.walk(node.value)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._check(node.value, f"alias[{target.id}]")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        slot = ast.unparse(node.target)
        self._check(node.annotation, slot)
        if _name(node.annotation, self.bindings) in {"typing.TypeAlias", "typing_extensions.TypeAlias"}:
            self._check(node.value, f"alias[{slot}]")


def check_file(path: Path) -> list[tuple[str, int]]:
    """Return symbol/annotation-slot keys and lines containing object types."""
    tree = ast.parse(path.read_bytes(), filename=str(path), type_comments=True)
    visitor = _Visitor(tree)
    visitor.visit(tree)
    return visitor.findings


def main() -> int:
    """Check the configured source tree and reject new or stale allowances."""
    config = tomllib.loads(Path("tools/annotation_policy.toml").read_text())
    legacy = config["legacy"]
    exceptions = config["exceptions"]
    if set(legacy) & set(exceptions):
        raise ValueError("An annotation cannot have both a legacy allowance and an exception")
    if any(type(count) is not int or count < 1 for count in legacy.values()):
        raise ValueError("Legacy counts must be positive integers")
    if any(not isinstance(reason, str) or not reason.strip() for reason in exceptions.values()):
        raise ValueError("Every exception must explain why the value is opaque")
    observed: Counter[str] = Counter()
    failed = False
    files = sorted({p for root in config["paths"] for p in Path(root).rglob("*.py")})
    if not files or any(not Path(root).is_dir() for root in config["paths"]):
        raise ValueError("Policy paths must exist and contain Python files")
    for path in files:
        try:
            findings = check_file(path)
        except (SyntaxError, ValueError) as exc:
            _LOG.error("%s: cannot inspect annotations: %s", path, exc)
            failed = True
            continue
        for slot, line in findings:
            key = f"{path.as_posix()}:{slot}"
            observed[key] += 1
            allowance = legacy.get(key, 1 if key in exceptions else 0)
            if observed[key] > allowance:
                _LOG.error(
                    "%s:%s: object annotation (%s): use a precise contract or a reviewed exception", path, line, slot
                )
                failed = True
    for key in legacy.keys() | exceptions.keys():
        expected = legacy.get(key, 1)
        if observed[key] < expected:
            _LOG.error("%s: stale allowance; remove it or reduce its legacy count", key)
            failed = True
    _LOG.info(
        "Annotation policy: checked %s files, %s existing/approved object annotations",
        len(files),
        sum(observed.values()),
    )
    return int(failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
