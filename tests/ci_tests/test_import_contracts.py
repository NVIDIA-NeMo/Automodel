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

"""Exercise import contracts without importing the training runtime.

Run with the linting dependency group: python tests/ci_tests/test_import_contracts.py.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RecipeImportContractTests(unittest.TestCase):
    """Check discovery and enforcement against isolated copies of real sources."""

    def setUp(self) -> None:
        """Copy sources so injected imports cannot modify the checkout."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        checkout = Path(__file__).resolve().parents[2]
        shutil.copytree(
            checkout / "nemo_automodel",
            self.root / "nemo_automodel",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(checkout / "pyproject.toml", self.root / "pyproject.toml")

    def _run(self, code: str) -> subprocess.CompletedProcess[str]:
        # A fresh interpreter resolves nemo_automodel from the temporary tree,
        # regardless of imports or sys.path changes in the test runner.
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_all_recipe_modules_are_in_graph(self) -> None:
        """Every recipe source must be visible to the architecture gate."""
        result = self._run(
            "import grimp, json; "
            "graph = grimp.build_graph('nemo_automodel', "
            "exclude_type_checking_imports=True, cache_dir=None); "
            "print(json.dumps(sorted(graph.modules)))"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        modules = set(json.loads(result.stdout))
        recipes = self.root / "nemo_automodel" / "recipes"
        expected = {"nemo_automodel.recipes"}
        for source in recipes.rglob("*.py"):
            relative = source.relative_to(self.root)
            module = relative.parent if source.name == "__init__.py" else relative.with_suffix("")
            expected.add(".".join(module.parts))
        self.assertFalse(
            expected - modules,
            f"Recipe modules missing from graph: {sorted(expected - modules)}",
        )

    def test_models_cannot_import_recipes(self) -> None:
        """The real contract must reject imports into each newly visible tree."""
        model = self.root / "nemo_automodel" / "components" / "models" / "gpt2.py"
        original = model.read_text()
        for target, statement in (
            (
                "dllm.train_ft",
                "from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe",
            ),
            ("llm.train_ft", "import nemo_automodel.recipes.llm.train_ft"),
            ("vlm.finetune", "import nemo_automodel.recipes.vlm.finetune"),
        ):
            with self.subTest(target=target):
                model.write_text(original + "\n" + statement + "\n")
                result = self._run(
                    "from importlinter.cli import lint_imports_command; lint_imports_command(['--no-cache'])"
                )
                output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
                self.assertEqual(result.returncode, 1, output)
                self.assertIn("Models must not import recipe configuration BROKEN", output)
                self.assertIn("nemo_automodel.components.models.gpt2", output)
                self.assertIn(f"nemo_automodel.recipes.{target}", output)


if __name__ == "__main__":
    unittest.main()
