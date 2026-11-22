# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise coverage metrics and Git revision comparisons through the CLI."""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "type_coverage.py"


class TestTypeCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.output = self.root / "report"
        self._git("init", "--quiet", "--initial-branch=main")

    def _git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Coverage Test",
                "-c",
                "user.email=coverage@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    def _commit(self, source: str, path: str = "nemo_automodel/sample.py") -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source))
        self._git("add", "--all")
        self._git("commit", "--quiet", "--allow-empty", "-m", "fixture")
        return self._git("rev-parse", "HEAD")

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--output-dir", str(self.output), *args],
            cwd=self.repo,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def _report(self, *args: str) -> dict:
        result = self._run(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((self.output / "coverage.json").read_text())

    def test_denominators_and_void_returns(self) -> None:
        self._commit("""
            def partial(value, *, count: int = 1): pass
            def typed(value: int, /, *others: int, **options: str) -> None: pass
            async def empty() -> None: pass
        """)
        totals = self._report()["head"]["total"]
        self.assertEqual(
            totals,
            {
                "functions": 3,
                "parameters": 5,
                "annotated_parameters": 4,
                "annotated_returns": 2,
                "complete": 2,
                "strict": 2,
            },
        )

    def test_excludes_broad_annotations_and_aliases(self) -> None:
        cases = [
            ("from typing import Any as A", "list[A]", "Any"),
            ("import typing as t\nAlias = t.Any", "'Alias'", "Any"),
            ("import typing as t\nAlias = list[t.Any]", "Alias", "Any"),
            ("", "object", "object"),
            ("from builtins import object as O", "O", "object"),
            ("", "int | None", "nullable union"),
            ("", "'int | None'", "nullable union"),
            ("from typing import Optional as O", "O[int]", "nullable union"),
            ("from typing import Union as U", "U[int, None]", "nullable union"),
            ("N = None", "int | N", "nullable union"),
            ("", "list", "bare generic"),
            ("from builtins import list as L", "L", "bare generic"),
            ("from typing import Mapping as M", "M", "bare generic"),
            ("from typing import Callable as C", "C[..., int]", "open Callable"),
            ("from typing import Callable\nC = Callable", "C[..., int]", "open Callable"),
            ("from typing_extensions import List as L", "L", "bare generic"),
        ]
        for prefix, annotation, reason in cases:
            with self.subTest(annotation=annotation, prefix=prefix):
                self._commit(f"{prefix}\ndef f(value: {annotation}) -> int: return 1\n")
                head = self._report()["head"]
                self.assertEqual(head["total"]["complete"], 1)
                self.assertEqual(head["total"]["strict"], 0)
                self.assertIn(reason, head["functions"][0]["exclusions"])

    def test_none_defaults_cannot_inflate_strict_coverage(self) -> None:
        self._commit("""
            def incorrect(value: int = None) -> int: return 1
            def keyword(*, value: str = None) -> None: pass
            def valid_void() -> None: pass
        """)
        head = self._report()["head"]
        self.assertEqual(head["total"]["complete"], 3)
        self.assertEqual(head["total"]["strict"], 1)
        self.assertEqual(head["exclusions"], {"None default": 2})

    def test_literal_and_annotated_metadata_are_not_types(self) -> None:
        self._commit("""
            from typing import Annotated, Literal, Callable
            def f(value: Annotated[Literal['Any', 'None'], 'Any | None']) -> None: pass
            def callback(fn: Callable[[int], None]) -> tuple[int, ...]: return (1,)
        """)
        self.assertEqual(self._report()["head"]["total"]["strict"], 2)

    def test_local_annotations_and_cast_aliases_cannot_hide_escapes(self) -> None:
        self._commit("""
            from typing import Any, cast
            trust = cast
            def f(x: int) -> int:
                unchecked: Any = x
                return unchecked
            def g(x: int) -> int:
                return trust(int, x)
        """)
        head = self._report()["head"]
        self.assertEqual(head["total"]["complete"], 2)
        self.assertEqual(head["total"]["strict"], 0)
        self.assertEqual(head["exclusions"], {"Any": 1, "cast": 1})

    def test_suppressions_and_casts_do_not_count_as_strict(self) -> None:
        self._commit("""
            from typing import cast as trust, no_type_check as unchecked
            @unchecked
            class Model:
                def method(self, x: int) -> int: return x
            def converted(x: int) -> int: return trust(int, x)
            def ignored(x: int) -> int:
                return x  # ty: ignore[invalid-return-type]
            class Suppressed:  # type: ignore
                def method(self) -> None: pass
            def valid() -> str:
                return '# type: ignore'  # An ordinary comment.
        """)
        head = self._report()["head"]
        self.assertEqual(head["total"]["complete"], 5)
        self.assertEqual(head["total"]["strict"], 1)
        self.assertEqual(head["exclusions"], {"cast": 1, "no_type_check": 1, "type suppression": 2})

    def test_file_suppressions_exclude_all_functions(self) -> None:
        for directive in ("# type: ignore", "# mypy: ignore_errors=True", "# pyright: reportArgumentType=false"):
            with self.subTest(directive=directive):
                self._commit(f"{directive}\ndef one() -> None: pass\ndef two() -> int: return 2\n")
                self.assertEqual(self._report()["head"]["exclusions"], {"type suppression": 2})

    def test_receivers_static_methods_and_public_scope(self) -> None:
        self._commit("""
            class Public:
                def method(self, x: int) -> int: return x
                @classmethod
                def make(cls) -> None: pass
                @staticmethod
                def static(self) -> None: pass
            class _Private:
                def method(self) -> None: pass
            def outer() -> None:
                def nested() -> None: pass
        """)
        head = self._report()["head"]
        self.assertEqual(head["total"]["parameters"], 2)
        self.assertEqual(head["public"]["functions"], 4)
        self.assertEqual(head["public"]["complete"], 3)
        self.assertEqual(head["total"]["functions"], 6)

    def test_never_imports_source_and_ignores_working_tree(self) -> None:
        sha = self._commit("raise RuntimeError('must not execute')\ndef f() -> int: return 1\n")
        (self.repo / "nemo_automodel/sample.py").write_text("def f(x): pass")
        (self.repo / "nemo_automodel/untracked.py").write_text("not valid Python(")
        head = self._report()["head"]
        self.assertEqual(head["revision"], sha)
        self.assertEqual(head["files"], 1)
        self.assertEqual(head["total"]["strict"], 1)

    def test_compare_merge_base_and_show_strict_regression(self) -> None:
        original = self._commit("def f(x: int) -> int: return x\n")
        self._git("checkout", "--quiet", "-b", "feature")
        head = self._commit("from typing import Any\ndef f(x: Any) -> Any: return x\n")
        self._git("checkout", "--quiet", "-b", "target", original)
        base_tip = self._commit("def f(x): return x\n")
        report = self._report("--head", head, "--base", base_tip)
        self.assertEqual(report["base"]["revision"], original)
        self.assertEqual(report["base"]["total"]["strict"], 1)
        self.assertEqual(report["head"]["total"]["strict"], 0)
        summary = (self.output / "summary.md").read_text()
        self.assertIn("-100.00 pp", summary)
        self.assertIn("**Strict complete signatures**", summary)

    def test_export_ignore_cannot_hide_untyped_code(self) -> None:
        self._commit("nemo_automodel/sample.py export-ignore\n", ".gitattributes")
        self._commit("def untyped(value): return value\n")
        head = self._report()["head"]
        self.assertEqual(head["files"], 1)
        self.assertEqual(head["total"]["functions"], 1)
        self.assertEqual(head["total"]["strict"], 0)

    def test_empty_denominators_and_deleted_areas(self) -> None:
        base = self._commit("def f() -> None: pass\n", "nemo_automodel/components/old/module.py")
        self._git("rm", "--quiet", "nemo_automodel/components/old/module.py")
        self._commit("constant = 1\n")
        report = self._report("--base", base)
        self.assertEqual(report["head"]["total"]["functions"], 0)
        self.assertIn("| components/old | 0 | N/A | N/A | N/A |", (self.output / "summary.md").read_text())

    def test_invalid_source_and_revision_fail(self) -> None:
        self._commit("def broken(\n")
        for args in ((), ("--head", "does-not-exist")):
            with self.subTest(args=args):
                result = self._run(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Cannot measure annotation coverage", result.stderr)
                self.assertFalse((self.output / "coverage.json").exists())


if __name__ == "__main__":
    unittest.main()
