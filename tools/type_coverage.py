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

"""Report raw and conservative annotation coverage of committed package source.

Uses only the standard library and never imports the measured code. Strict
coverage excludes Any, object, nullable unions/defaults, bare generics, open Callables,
casts, type-checker suppressions and no_type_check. These are conservative
exclusions, not findings: optional types and casts can be correct. A return
annotation of None is credited. This syntax metric does not prove type
soundness, resolve cross-module aliases, or measure inferred expression types.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

_BARE_GENERICS = {
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "type",
    *(
        f"typing.{name}"
        for name in (
            "List",
            "Dict",
            "Tuple",
            "Set",
            "FrozenSet",
            "Type",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "MutableSequence",
            "Mapping",
            "MutableMapping",
            "Collection",
            "Generator",
            "AsyncIterable",
            "AsyncIterator",
            "Awaitable",
            "Coroutine",
            "ContextManager",
            "IO",
            "Pattern",
            "Match",
            "Optional",
            "Union",
            "Annotated",
            "Literal",
        )
    ),
    *(
        f"collections.abc.{name}"
        for name in (
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "MutableSequence",
            "Mapping",
            "MutableMapping",
            "Collection",
            "Generator",
            "AsyncIterable",
            "AsyncIterator",
            "Awaitable",
            "Coroutine",
        )
    ),
}
_IGNORE = re.compile(r"#\s*(?:type|ty|pyright):\s*ignore\b")
_FILE_IGNORE = re.compile(r"#\s*(?:mypy:.*ignore_errors\s*=\s*True|pyright:.*(?:false|off|basic))\b")


def _name(node: ast.expr, imports: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return imports.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value, imports)}.{node.attr}"
    return ""


def _resolve(node: ast.expr, aliases: dict[str, ast.expr], seen: frozenset[str] = frozenset()) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in aliases and node.id not in seen:
        return _resolve(aliases[node.id], aliases, seen | {node.id})
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _resolve(ast.parse(node.value, mode="eval").body, aliases, seen)
        except SyntaxError:
            return node
    return node


def _is_none(node: ast.expr, aliases: dict[str, ast.expr], imports: dict[str, str]) -> bool:
    resolved = _resolve(node, aliases)
    return (
        isinstance(resolved, ast.Constant) and resolved.value is None or _name(resolved, imports) in {"types.NoneType"}
    )


def _annotation_reasons(
    node: ast.expr,
    imports: dict[str, str],
    aliases: dict[str, ast.expr],
    seen: frozenset[str] = frozenset(),
    parameterized: bool = False,
) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _annotation_reasons(ast.parse(node.value, mode="eval").body, imports, aliases, seen)
        except SyntaxError:
            return {"invalid annotation"}
    name = _name(node, imports).replace("typing_extensions.", "typing.", 1)
    if isinstance(node, ast.Name) and node.id in aliases and node.id not in seen:
        return _annotation_reasons(aliases[node.id], imports, aliases, seen | {node.id}, parameterized)
    reasons: set[str] = set()
    if name in {"Any", "typing.Any", "typing_extensions.Any"}:
        reasons.add("Any")
    if name in {"object", "builtins.object"}:
        reasons.add("object")
    if name.removeprefix("builtins.") in _BARE_GENERICS and not parameterized:
        reasons.add("bare generic")
    if isinstance(node, ast.Subscript):
        container = _name(_resolve(node.value, aliases), imports)
        reasons |= _annotation_reasons(node.value, imports, aliases, seen, parameterized=True)
        elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if container in {"typing.Literal", "typing_extensions.Literal"}:
            return reasons
        if container in {"typing.Annotated", "typing_extensions.Annotated"}:
            elements = elements[:1]  # Metadata is not a type annotation.
        if container in {"typing.Optional", "typing_extensions.Optional"} or (
            container in {"typing.Union", "typing_extensions.Union"}
            and any(_is_none(e, aliases, imports) for e in elements)
        ):
            reasons.add("nullable union")
        if (
            container in {"typing.Callable", "collections.abc.Callable"}
            and elements
            and (isinstance(elements[0], ast.Constant) and elements[0].value is Ellipsis)
        ):
            reasons.add("open Callable")
        for element in elements:
            reasons |= _annotation_reasons(element, imports, aliases, seen)
        return reasons
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and any(_is_none(e, aliases, imports) for e in (node.left, node.right))
    ):
        reasons.add("nullable union")
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            reasons |= _annotation_reasons(child, imports, aliases, seen)
    return reasons


@dataclass(frozen=True)
class _Function:
    path: str
    name: str
    line: int
    parameters: int
    annotated_parameters: int
    annotated_return: bool
    public: bool
    exclusions: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.annotated_return and self.parameters == self.annotated_parameters


class _Visitor(ast.NodeVisitor):
    def __init__(self, source: bytes, path: str) -> None:
        self.path = path
        self.tree = ast.parse(source, filename=path)
        self.imports: dict[str, str] = {}
        self.aliases: dict[str, ast.expr] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and node.module in {
                "typing",
                "typing_extensions",
                "collections.abc",
                "builtins",
            }:
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.aliases[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.aliases[node.target.id] = node.value
        comments = [t for t in tokenize.tokenize(io.BytesIO(source).readline) if t.type == tokenize.COMMENT]
        self.ignored_lines = {t.start[0] for t in comments if _IGNORE.search(t.string)}
        first_statement = self.tree.body[0].lineno if self.tree.body else 0
        self.file_ignored = any(_FILE_IGNORE.search(t.string) for t in comments) or any(
            line < first_statement for line in self.ignored_lines
        )
        self.scopes: list[tuple[str, str]] = []
        self.no_type_check = False
        self.class_suppressed = False
        self.functions: list[_Function] = []

    def _unchecked(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return self.no_type_check or any(
            _name(d, self.imports) in {"typing.no_type_check", "typing_extensions.no_type_check"}
            for d in node.decorator_list
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.no_type_check
        suppressed = self.class_suppressed
        self.no_type_check = self._unchecked(node)
        self.class_suppressed |= node.lineno in self.ignored_lines
        self.scopes.append(("class", node.name))
        self.generic_visit(node)
        self.scopes.pop()
        self.no_type_check = previous
        self.class_suppressed = suppressed

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = list(node.args.posonlyargs) + list(node.args.args)
        if (
            self.scopes
            and self.scopes[-1][0] == "class"
            and args
            and args[0].arg in {"self", "cls"}
            and not any(
                _name(d, self.imports) in {"staticmethod", "builtins.staticmethod"} for d in node.decorator_list
            )
        ):
            args = args[1:]
        args += list(node.args.kwonlyargs) + [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        annotations = [a.annotation for a in args] + [node.returns]
        reasons = set().union(
            *(
                _annotation_reasons(a, self.imports, self.aliases) if a is not None else {"missing annotation"}
                for a in annotations
            )
        )
        positional = node.args.posonlyargs + node.args.args
        defaults = list(zip(positional[-len(node.args.defaults) :], node.args.defaults)) if node.args.defaults else []
        defaults += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
        if any(isinstance(default, ast.Constant) and default.value is None for _, default in defaults):
            reasons.add("None default")
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        if (
            self.file_ignored
            or self.class_suppressed
            or any(start <= line <= (node.end_lineno or node.lineno) for line in self.ignored_lines)
        ):
            reasons.add("type suppression")
        unchecked = self._unchecked(node)
        if unchecked:
            reasons.add("no_type_check")
        for child in ast.walk(node):
            if isinstance(child, ast.AnnAssign):
                reasons |= _annotation_reasons(child.annotation, self.imports, self.aliases)
            if isinstance(child, ast.Call) and _name(_resolve(child.func, self.aliases), self.imports) in {
                "typing.cast",
                "typing_extensions.cast",
            }:
                reasons.add("cast")
        self.functions.append(
            _Function(
                self.path,
                ".".join([name for _, name in self.scopes] + [node.name]),
                node.lineno,
                len(args),
                sum(a.annotation is not None for a in args),
                node.returns is not None,
                not node.name.startswith("_")
                and not any(kind == "function" or name.startswith("_") for kind, name in self.scopes),
                tuple(sorted(reasons)),
            )
        )
        previous = self.no_type_check
        self.no_type_check = unchecked
        self.scopes.append(("function", node.name))
        self.generic_visit(node)
        self.scopes.pop()
        self.no_type_check = previous

    visit_AsyncFunctionDef = visit_FunctionDef


@dataclass(frozen=True)
class _Totals:
    functions: int
    parameters: int
    annotated_parameters: int
    annotated_returns: int
    complete: int
    strict: int


def _totals(functions: list[_Function]) -> _Totals:
    return _Totals(
        len(functions),
        sum(f.parameters for f in functions),
        sum(f.annotated_parameters for f in functions),
        sum(f.annotated_return for f in functions),
        sum(f.complete for f in functions),
        sum(f.complete and not f.exclusions for f in functions),
    )


def _area(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[1:3] if parts[1] == "components" else parts[1:2])


@dataclass(frozen=True)
class _Snapshot:
    revision: str
    files: int
    functions: list[_Function]


def _git(*args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=input_data, check=True, capture_output=True, timeout=60).stdout


def _snapshot(revision: str) -> _Snapshot:
    revision = _git("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()
    functions: list[_Function] = []
    objects: list[tuple[str, bytes]] = []
    for entry in _git("ls-tree", "-r", "-z", revision, "--", "nemo_automodel").split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if mode in {b"100644", b"100755"} and kind == b"blob" and path.endswith(b".py"):
            objects.append((path.decode(), oid))
    if objects:
        # Read blobs directly: working-tree edits and export-ignore attributes cannot hide code.
        blobs = io.BytesIO(_git("cat-file", "--batch", input_data=b"".join(oid + b"\n" for _, oid in objects)))
        for path, _ in objects:
            _, _, size = blobs.readline().split()
            visitor = _Visitor(blobs.read(int(size)), path)
            blobs.read(1)  # Git's separator after each blob.
            visitor.visit(visitor.tree)
            functions.extend(visitor.functions)
    return _Snapshot(revision, len(objects), functions)


def _snapshot_report(snapshot: _Snapshot) -> dict[str, object]:
    return {
        "revision": snapshot.revision,
        "files": snapshot.files,
        "total": asdict(_totals(snapshot.functions)),
        "public": asdict(_totals([f for f in snapshot.functions if f.public])),
        "areas": {
            area: asdict(_totals([f for f in snapshot.functions if _area(f.path) == area]))
            for area in sorted({_area(f.path) for f in snapshot.functions})
        },
        "exclusions": dict(sorted(Counter(reason for f in snapshot.functions for reason in f.exclusions).items())),
        "functions": [
            asdict(f) | {"complete": f.complete, "strict": f.complete and not f.exclusions} for f in snapshot.functions
        ],
    }


def _percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "N/A"


def _delta(current: tuple[int, int], previous: tuple[int, int]) -> str:
    if not current[1] or not previous[1]:
        return "N/A"
    return f"{100 * (current[0] / current[1] - previous[0] / previous[1]):+.2f} pp"


def _summary(head: _Snapshot, base: _Snapshot | None) -> str:
    total = _totals(head.functions)
    public = _totals([f for f in head.functions if f.public])
    base_total = _totals(base.functions if base else [])
    base_public = _totals([f for f in base.functions if f.public] if base else [])
    lines = ["## Python annotation coverage", "", f"Head: `{head.revision}`"]
    if base:
        lines.append(f"Base (merge-base): `{base.revision}`")
    lines += [
        "",
        f"Scope: **{head.files} tracked Python files**, **{total.functions} definitions** under `nemo_automodel/`.",
        "",
        "| Metric | Base | Head | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    metrics = [
        (
            "Parameters annotated",
            (total.annotated_parameters, total.parameters),
            (base_total.annotated_parameters, base_total.parameters),
        ),
        (
            "Returns annotated",
            (total.annotated_returns, total.functions),
            (base_total.annotated_returns, base_total.functions),
        ),
        ("Complete signatures", (total.complete, total.functions), (base_total.complete, base_total.functions)),
        ("**Strict complete signatures**", (total.strict, total.functions), (base_total.strict, base_total.functions)),
        (
            "Public complete signatures",
            (public.complete, public.functions),
            (base_public.complete, base_public.functions),
        ),
        ("Strict public signatures", (public.strict, public.functions), (base_public.strict, base_public.functions)),
    ]
    for label, current, previous in metrics:
        lines.append(
            f"| {label} | {_percentage(*previous)} | {_percentage(*current)} ({current[0]}/{current[1]}) | {_delta(current, previous)} |"
        )
    lines += [
        "",
        "### Coverage by area",
        "",
        "| Area | Definitions | Complete | Strict | Strict change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    areas = {_area(f.path) for f in head.functions}
    if base:
        areas |= {_area(f.path) for f in base.functions}
    for area in sorted(areas):
        current = _totals([f for f in head.functions if _area(f.path) == area])
        previous = _totals([f for f in base.functions if _area(f.path) == area] if base else [])
        label = escape(area).replace("|", "&#124;").replace("\n", " ")
        lines.append(
            f"| {label} | {current.functions} | {_percentage(current.complete, current.functions)} | {_percentage(current.strict, current.functions)} | {_delta((current.strict, current.functions), (previous.strict, previous.functions))} |"
        )
    reasons = Counter(reason for f in head.functions for reason in f.exclusions)
    previous_reasons = Counter(reason for f in base.functions for reason in f.exclusions) if base else Counter()
    lines += [
        "",
        "### Strict exclusions",
        "",
        "Counts are definitions affected; reasons can overlap.",
        "",
        "| Reason | Base | Head |",
        "| --- | ---: | ---: |",
    ]
    for reason in sorted(reasons.keys() | previous_reasons.keys()):
        lines.append(f"| {reason} | {previous_reasons[reason] if base else 'N/A'} | {reasons[reason]} |")
    lines += [
        "",
        "Complete means every parameter and return is annotated, excluding method `self`/`cls`.",
        "Strict additionally excludes `Any`, `object`, nullable unions (`T | None`, `Optional[T]`), `None` parameter defaults, bare generics, "
        "`Callable[..., T]`, functions containing `cast` or type-ignore comments, and `no_type_check` scopes. "
        "Top-level type-ignore directives exclude the file. A return of `-> None` is credited.",
        "",
        "These conservative exclusions can include valid contracts; this is annotation coverage, not proof of type soundness. "
        "Signature and local variable annotations, module-local aliases and quoted annotations are inspected; "
        "cross-module aliases and inferred types are not resolved. "
        "All definitions, including nested functions, kernels and vendored code, are counted. "
        "Public uses non-underscore names outside nested functions and private classes. "
        "Both revisions use the same reporter. Reporting does not enforce a coverage threshold.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write coverage.json and summary.md for Git revisions; return 2 on invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="HEAD", help="Commit to measure (working-tree changes are excluded).")
    parser.add_argument("--base", help="Compare against this revision's merge-base with head.")
    parser.add_argument("--output-dir", type=Path, default=Path("type-coverage"))
    args = parser.parse_args(argv)
    try:
        head = _snapshot(args.head)
        base = None
        if args.base:
            base_sha = _git("rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}").decode().strip()
            base = _snapshot(_git("merge-base", base_sha, head.revision).decode().strip())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"schema_version": 1, "head": _snapshot_report(head), "base": _snapshot_report(base) if base else None}
        (args.output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "summary.md").write_text(_summary(head, base))
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"Cannot measure annotation coverage: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
