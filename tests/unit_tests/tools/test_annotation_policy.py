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

"""Exercise the policy and combined ty entry point without training dependencies."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


class TestAnnotationPolicy(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tools").mkdir()
        (self.root / "src").mkdir()
        for script in ("check_annotation_policy.py", "check_types.py"):
            shutil.copyfile(_ROOT / "tools" / script, self.root / "tools" / script)
        self.configure()

    def configure(self, exceptions: str = "") -> None:
        (self.root / "tools/annotation_policy.toml").write_text(f'paths = ["src"]\n[exceptions]\n{exceptions}\n')

    def run_policy(self, source: str, script: str = "check_annotation_policy.py") -> subprocess.CompletedProcess[str]:
        (self.root / "src/example.py").write_text(textwrap.dedent(source))
        return subprocess.run(
            [sys.executable, str(self.root / "tools" / script)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_rejects_annotations_and_aliases(self) -> None:
        for annotation in (
            "object",
            "list[object]",
            '"dict[str, object]"',
            "object | None",
            "tuple[int, object]",
            "b.object",
            "Opaque",
            "Alias",
            "list[Alias]",
        ):
            with self.subTest(annotation=annotation):
                result = self.run_policy(f"""
                    import builtins as b
                    from builtins import object as Opaque
                    Alias = Opaque
                    async def f(x: {annotation}) -> None: pass
                """)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("f:x", result.stderr)

    def test_return_attributes_varargs_and_alias_declarations(self) -> None:
        result = self.run_policy("""
            from typing import TypeAlias
            Alias: TypeAlias = "list[object]"
            class C:
                value: object
                def f(self, *args: object, **kwargs: object) -> object:
                    self.local: list[object] = []
        """)
        self.assertEqual(result.returncode, 1, result.stderr)
        for slot in ("alias[Alias]", "C:value", "C.f:args", "C.f:kwargs", "C.f:return", "C.f:self.local"):
            self.assertIn(slot, result.stderr)

    def test_type_comments_cannot_bypass_policy(self) -> None:
        result = self.run_policy("""
            value = None  # type: object
            def f(x):
                # type: (object) -> object
                return x
        """)
        self.assertEqual(result.returncode, 1, result.stderr)
        for slot in ("comment[value]", "f:comment-arg[0]", "f:comment-return"):
            self.assertIn(slot, result.stderr)

    def test_runtime_values_metadata_and_precise_types_pass(self) -> None:
        result = self.run_policy("""
            from typing import Annotated as A, Literal as L
            SENTINEL = object()
            text = "object"
            item = {"object": 1}["object"]
            class C(object):
                def f(self, x: A[int, "object"], y: L["object"]) -> str:
                    return str(x)
            def other(x: list[int] | None) -> None: pass
        """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_alias_chains_pass(self) -> None:
        result = self.run_policy("""
            MISSING = object()
            DEFAULT = MISSING
            CHAINED = DEFAULT
            text = "object"
            copy = text
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("object", "list[object]", 'list["object"]'):
            with self.subTest(value=value):
                result = self.run_policy(f"Alias = {value}\nChained = Alias")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("alias[Chained]", result.stderr)

    def test_remaining_native_type_comments(self) -> None:
        templates = (
            "def f(\n    value,  # type: {annotation}\n):\n    pass",
            "for value in []:  # type: {annotation}\n    pass",
            "with manager() as value:  # type: {annotation}\n    pass",
            "async def f():\n    async for value in source:  # type: {annotation}\n        pass",
            "async def f():\n    async with manager() as value:  # type: {annotation}\n        pass",
        )
        for template in templates:
            for annotation in ("object", "int", 'Literal["object"]'):
                with self.subTest(template=template, annotation=annotation):
                    source = "from typing import Literal\n" + template.format(annotation=annotation)
                    result = self.run_policy(source)
                    self.assertEqual(result.returncode, int(annotation == "object"), result.stderr)
                    self.assertEqual(result.stderr.count("object annotation ("), int(annotation == "object"))
                    if annotation == "object":
                        self.assertIn("comment[value]", result.stderr)
        result = self.run_policy("for value in []:  # type: int\n    nested: object")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("<module>:nested", result.stderr)

    def test_class_namespaces_are_not_method_closures(self) -> None:
        result = self.run_policy("""
            class C(list[object]):
                object = int
                def method(self, x: object):
                    def inner(y: object): pass
            def outer():
                from builtins import object as Opaque
                class Nested:
                    Opaque = int
                    def method(self, x: Opaque):
                        def inner(y: Opaque): pass
        """)
        self.assertEqual(result.returncode, 1, result.stderr)
        for slot in ("C:base[0]", "C.method.inner:y", "outer.Nested.method.inner:y"):
            self.assertIn(slot, result.stderr)
        for slot in ("C.method:x", "outer.Nested.method:x"):
            self.assertNotIn(slot, result.stderr)

    def test_local_aliases_and_sibling_scopes(self) -> None:
        result = self.run_policy("""
            def outer():
                from builtins import object as O
                Alias = list[O]
                def inner(value: Alias): pass
            def sibling():
                Alias = int
                def inner(value: Alias): pass
        """)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("outer.inner:value", result.stderr)
        self.assertNotIn("sibling.inner", result.stderr)

    def test_shadowed_builtin_is_not_object(self) -> None:
        result = self.run_policy("""
            from library import object
            def f(value: object): pass
        """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_alias_cycles_terminate(self) -> None:
        result = self.run_policy("""
            A = list[B]
            B = A
            def f(value: A): pass
        """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_allowances_are_rejected(self) -> None:
        (self.root / "tools/annotation_policy.toml").write_text(
            'paths = ["src"]\n[exceptions]\n[legacy]\n"src/example.py:f:x" = 1\n'
        )
        result = self.run_policy("def f(x: object): pass")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Legacy allowances are not supported", result.stderr)

    def test_stale_exception_fails(self) -> None:
        self.configure(exceptions='"src/example.py:f:x" = "Opaque serializer input"')
        result = self.run_policy("def f(x: int): pass")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("stale exception", result.stderr)

    def test_reviewed_exception_requires_reason_and_is_narrow(self) -> None:
        self.configure(exceptions='"src/example.py:f:x" = "Opaque serializer input"')
        self.assertEqual(self.run_policy("def f(x: object): pass").returncode, 0)
        self.assertEqual(self.run_policy("def g(x: object): pass").returncode, 1)
        self.configure(exceptions='"src/example.py:f:x" = ""')
        result = self.run_policy("def f(x: object): pass")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must explain", result.stderr)

    def test_syntax_errors_and_missing_roots_fail_closed(self) -> None:
        self.assertNotEqual(self.run_policy("def broken(").returncode, 0)
        (self.root / "tools/annotation_policy.toml").write_text('paths = ["missing"]\n[exceptions]')
        result = self.run_policy("def f(x: int): pass")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must exist", result.stderr)

    def test_all_gates_run_and_any_failure_fails_combined_command(self) -> None:
        (self.root / "ty.toml").write_text('[[overrides]]\ninclude = ["src/**"]\n')
        # A tiny module stands in for the external executable, recording arguments
        # and returning a controlled exit code. The real policy still runs.
        for ty_code, annotation, expected in ((0, "int", 0), (1, "int", 1), (0, "object", 1)):
            with self.subTest(ty_code=ty_code, annotation=annotation):
                (self.root / "ty.py").write_text(
                    "import json, sys\nfrom pathlib import Path\n"
                    'Path("ty-args.json").write_text(json.dumps(sys.argv[1:]))\n'
                    f"raise SystemExit({ty_code})\n"
                )
                result = self.run_policy(f"def f(x: {annotation}) -> None: pass", "check_types.py")
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(json.loads((self.root / "ty-args.json").read_text()), ["check", "--verbose", "src"])

    def test_combined_command_rejects_missing_signatures(self) -> None:
        (self.root / "ty.toml").write_text('[[overrides]]\ninclude = ["src/**"]\n')
        (self.root / "ty.py").write_text('from pathlib import Path\nPath("ty-ran").touch()\n')
        # Completeness must not inherit Ruff settings that suppress missing
        # annotations, even on fully untyped functions and dummy arguments.
        (self.root / "ruff.toml").write_text(
            '[lint]\nignore = ["ANN"]\n[lint.flake8-annotations]\n'
            "ignore-fully-untyped = true\nsuppress-dummy-args = true\nmypy-init-return = true\n"
        )
        cases = (
            ("def f(value) -> None: pass", "ANN001"),
            ("def f(_unused) -> None: pass", "ANN001"),
            ("def f(*args) -> None: pass", "ANN002"),
            ("def f(**kwargs) -> None: pass", "ANN003"),
            ("def f(): pass", "ANN201"),
            ("def _get_cloud(name: str): return name", "ANN202"),
            ("class C:\n    def __init__(self, value: int): pass", "ANN204"),
            ("class C:\n    @staticmethod\n    def f(): pass", "ANN205"),
            ("class C:\n    @classmethod\n    def f(cls): pass", "ANN206"),
            ("async def f(value) -> None: pass", "ANN001"),
            ("def f() -> None:\n    def nested(): pass", "ANN202"),
            ("def f(): pass  # noqa: ANN201", "ANN201"),
        )
        for source, rule in cases:
            with self.subTest(rule=rule, source=source):
                (self.root / "ty-ran").unlink(missing_ok=True)
                result = self.run_policy(source, "check_types.py")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(rule, result.stdout)
                self.assertTrue((self.root / "ty-ran").exists(), "ty must still run after the signature gate fails")

    def test_signature_gate_follows_opt_in_scope_and_exclusions(self) -> None:
        (self.root / "ty.py").write_text("")
        (self.root / "ty.toml").write_text(
            '[src]\nexclude = ["src/excluded.py"]\n[[overrides]]\ninclude = ["src/**"]\n'
        )
        (self.root / "src/excluded.py").write_text("def untyped(value): return value\n")
        valid = "class C:\n    def __init__(self, value: int) -> None: pass\n    def f(self) -> None: pass\n"
        self.assertEqual(self.run_policy(valid, "check_types.py").returncode, 0)
        # Narrow the opt-in scope to one file: an untyped sibling is not opted in.
        (self.root / "ty.toml").write_text('[[overrides]]\ninclude = ["src/example.py"]\n')
        self.assertEqual(self.run_policy(valid, "check_types.py").returncode, 0)
        # Opting the sibling in must immediately expose its existing omissions.
        (self.root / "ty.toml").write_text('[[overrides]]\ninclude = ["src/**"]\n')
        result = self.run_policy(valid, "check_types.py")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ANN001", result.stdout)
        self.assertIn("ANN201", result.stdout)


if __name__ == "__main__":
    unittest.main()
