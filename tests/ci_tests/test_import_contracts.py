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
import tomllib
import unittest
from pathlib import Path

CONFIG_BOUNDARY_CONTRACT = "Components must not import the YAML config layer or recipes"


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

    def _importlinter_config(self) -> dict:
        return tomllib.loads((self.root / "pyproject.toml").read_text())["tool"]["importlinter"]

    def _lint_imports(self) -> tuple[int, str]:
        result = self._run("from importlinter.cli import lint_imports_command; lint_imports_command(['--no-cache'])")
        return result.returncode, re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

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
        exclude = self._importlinter_config()["exclude_type_checking_imports"]
        result = self._run(
            "import grimp, json; "
            "graph = grimp.build_graph('nemo_automodel', "
            f"exclude_type_checking_imports={exclude!r}, cache_dir=None); "
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

    def test_type_checking_imports_count(self) -> None:
        """A type-only import is still a dependency; ``TYPE_CHECKING`` must not hide it from the contracts."""
        self.assertIs(self._importlinter_config()["exclude_type_checking_imports"], False)

    def test_components_cannot_import_config_even_for_type_checking(self) -> None:
        """A ``TYPE_CHECKING``-guarded ``ConfigNode`` import into any component breaks the config-boundary contract."""
        # The canonical guard spelling, which grimp recognizes as a type-checking import.
        injected = (
            "\nfrom typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from nemo_automodel.components.config.loader import ConfigNode\n"
        )
        # loggers is also covered by the independence contract; speculative only by the config-boundary one.
        for module in ("loggers/mlflow_utils.py", "speculative/dflash/target_options.py"):
            with self.subTest(module=module):
                source = self.root / "nemo_automodel" / "components" / module
                original = source.read_text()
                source.write_text(original + injected)
                try:
                    returncode, output = self._lint_imports()
                finally:
                    source.write_text(original)
                self.assertEqual(returncode, 1, output)
                self.assertIn(f"{CONFIG_BOUNDARY_CONTRACT} BROKEN", output)
                self.assertIn("nemo_automodel.components.config.loader", output)

    def test_config_boundary_contract_covers_every_component(self) -> None:
        """Every component package except ``config`` is a source of the config-boundary contract."""
        contracts = self._importlinter_config()["contracts"]
        (contract,) = [c for c in contracts if c["name"] == CONFIG_BOUNDARY_CONTRACT]
        components = self.root / "nemo_automodel" / "components"
        expected = {
            f"nemo_automodel.components.{path.parent.name}"
            for path in components.glob("*/__init__.py")
            if path.parent.name != "config"
        }
        self.assertEqual(expected, set(contract["source_modules"]))
        self.assertEqual(
            {"nemo_automodel.components.config", "nemo_automodel.recipes"}, set(contract["forbidden_modules"])
        )


if __name__ == "__main__":
    unittest.main()
