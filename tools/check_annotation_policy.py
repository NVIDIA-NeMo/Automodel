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

Run from the repository root with Python 3.11+. Configuration and reviewed
exceptions live in tools/annotation_policy.toml. This syntax gate complements ty;
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


def _has_object(
    node: ast.expr, bindings: dict[str, ast.expr], seen: frozenset[str] = frozenset(), *, implicit_alias: bool = False
) -> bool:
    if implicit_alias and isinstance(node, ast.Call):
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if implicit_alias:
            return False
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False  # ty diagnoses invalid forward references.
        return _has_object(parsed, bindings, seen)
    if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
        return _has_object(bindings[node.id], bindings, seen | {node.id}, implicit_alias=implicit_alias)
    if _name(node, bindings, seen) in {"object", "builtins.object"}:
        return True
    if isinstance(node, ast.Subscript):
        name = _name(node.value, bindings)
        if name in {"typing.Literal", "typing_extensions.Literal"}:
            return False
        elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name in {"typing.Annotated", "typing_extensions.Annotated"}:
            elements = elements[:1]
        return _has_object(node.value, bindings, seen, implicit_alias=implicit_alias) or any(
            _has_object(e, bindings, seen) for e in elements
        )
    return any(
        _has_object(child, bindings, seen, implicit_alias=implicit_alias)
        for child in ast.iter_child_nodes(node)
        if isinstance(child, ast.expr)
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.bindings = _bindings(tree.body)
        self.lexical_bindings = self.bindings
        self.scope: list[str] = []
        self.findings: list[tuple[str, int]] = []

    def _check(self, node: ast.expr | None, slot: str, *, implicit_alias: bool = False) -> None:
        if node is not None and _has_object(node, self.bindings, implicit_alias=implicit_alias):
            self.findings.append((f"{'.'.join(self.scope) or '<module>'}:{slot}", node.lineno))

    def _check_comment(self, text: str | None, line: int, slot: str) -> None:
        if text:
            annotation = ast.parse(text, mode="eval").body
            ast.increment_lineno(annotation, line - 1)
            self._check(annotation, slot)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self._check_comment(node.type_comment, node.lineno, f"comment[{ast.unparse(node.target)}]")
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        targets = ", ".join(ast.unparse(item.optional_vars) for item in node.items if item.optional_vars is not None)
        self._check_comment(node.type_comment, node.lineno, f"comment[{targets}]")
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        for arg in args:
            self._check(arg.annotation, arg.arg)
            self._check_comment(arg.type_comment, arg.lineno, f"comment[{arg.arg}]")
        self._check(node.returns, "return")
        if node.type_comment:
            comment = ast.parse(node.type_comment, mode="func_type")
            ast.increment_lineno(comment, node.lineno - 1)
            for index, annotation in enumerate(comment.argtypes):
                self._check(annotation, f"comment-arg[{index}]")
            self._check(comment.returns, "comment-return")
        previous = self.bindings
        previous_lexical = self.lexical_bindings
        self.bindings = previous_lexical | {arg.arg: ast.Constant(None) for arg in args} | _bindings(node.body)
        self.lexical_bindings = self.bindings
        for statement in node.body:
            self.visit(statement)
        self.bindings = previous
        self.lexical_bindings = previous_lexical
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        previous = self.bindings
        for index, base in enumerate(node.bases):
            # Inheriting from object is a runtime use, not an annotation.
            if isinstance(base, ast.Subscript):
                self._check(base, f"base[{index}]")
        # Class locals affect method signatures, but are not a closure for method bodies.
        self.bindings = self.lexical_bindings | _bindings(node.body)
        for statement in node.body:
            self.visit(statement)
        self.bindings = previous
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # Explicit alias expressions must not hide object from importing modules.
        # Strings and calls here are runtime values, not declared type aliases.
        self._check_comment(node.type_comment, node.lineno, f"comment[{ast.unparse(node.targets[0])}]")
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
                    self._check(node.value, f"alias[{target.id}]", implicit_alias=True)

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
    """Check the configured source tree and reject unapproved annotations and stale exceptions."""
    config = tomllib.loads(Path("tools/annotation_policy.toml").read_text())
    if "legacy" in config:
        raise ValueError("Legacy allowances are not supported; existing annotations must satisfy the policy")
    exceptions = config["exceptions"]
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
            allowance = 1 if key in exceptions else 0
            if observed[key] > allowance:
                _LOG.error(
                    "%s:%s: object annotation (%s): use a precise contract or a reviewed exception", path, line, slot
                )
                failed = True
    for key in exceptions:
        if observed[key] == 0:
            _LOG.error("%s: stale exception; remove it", key)
            failed = True
    _LOG.info(
        "Annotation policy: checked %s files, %s object annotations",
        len(files),
        sum(observed.values()),
    )
    return int(failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
