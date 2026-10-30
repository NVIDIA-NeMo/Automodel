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

"""CPU contracts for fictional data and count-checked reporting."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tutorials.decision_models.decision_tutorial.data import (
    make_records,
    validate_splits,
    validate_summary,
    write_dataset,
)
from tutorials.decision_models.decision_tutorial.reporting import summarize_run


class TutorialContractTests(unittest.TestCase):
    """Test public fixture behavior without framework imports or downloads."""

    def test_notebook_code_matches_paired_scripts(self) -> None:
        """Notebook and script readers execute the same Python workflow."""
        root = Path(__file__).resolve().parents[3] / "tutorials" / "decision_models"
        notebooks = sorted(root.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 7)
        for notebook in notebooks:
            with self.subTest(notebook=notebook.name):
                cells = json.loads(notebook.read_text())["cells"]
                code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
                self.assertEqual(
                    ast.dump(ast.parse(code)),
                    ast.dump(ast.parse(notebook.with_suffix(".py").read_text())),
                )

    def test_deterministic_disjoint_reserved_fixture(self) -> None:
        """Identical seeds reproduce bytes and every split has distinct states."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = write_dataset(root / "first", train_records=16, validation_records=8, test_records=8)
            second = write_dataset(root / "second", train_records=16, validation_records=8, test_records=8)
            self.assertEqual(first, second)
            for name in first:
                self.assertEqual(
                    hashlib.sha256((root / "first" / f"{name}.jsonl").read_bytes()).hexdigest(),
                    hashlib.sha256((root / "second" / f"{name}.jsonl").read_bytes()).hexdigest(),
                )
            self.assertEqual(sum(len(rows) for rows in first.values()), 32)
            with self.assertRaises(FileExistsError):
                write_dataset(root / "first")

    def test_invalid_gold_and_overlap_are_rejected(self) -> None:
        """Label typos and reused evaluation observations fail loudly."""
        records = make_records(count=2, seed=9)
        with self.assertRaises(ValueError):
            validate_splits({"train": records[:1], "test": records[:1]})
        invalid = copy.deepcopy(records)
        invalid[0]["answers"]["verdict"] = "unknown"
        with self.assertRaises(ValueError):
            validate_splits({"train": invalid})
        invalid = copy.deepcopy(records)
        invalid[0]["state"]["source_ip"] = "8.8.8.8"
        with self.assertRaises(ValueError):
            validate_splits({"train": invalid})

    def test_generated_prose_requires_reserved_identifiers(self) -> None:
        """Optional prose cannot add routable IPs or real-looking emails."""
        validate_summary("Contact user@example.com about host 192.0.2.4.")
        for text in ("Contact user@company.com", "Connect to 8.8.8.8", "nvapi-" + "a" * 30):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_summary(text)

    def test_reporting_keeps_stages_and_checks_counts(self) -> None:
        """Retain negative changes and reject reports with unequal counts."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for stage, accuracy in (("base", 0.7), ("sft", 0.6)):
                directory = root / stage
                directory.mkdir()
                overall = dict(records=8, n=40, accuracy=accuracy, exact_match=0.5, nll=0.8, brier=0.4, reward=0.3)
                (directory / "metrics_test.json").write_text(json.dumps({"overall": overall}))
            rows = summarize_run(root)
            self.assertLess(rows[1]["accuracy"], rows[0]["accuracy"])
            path = root / "sft/metrics_test.json"
            summary = json.loads(path.read_text())
            summary["overall"]["records"] = 7
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                summarize_run(root)


if __name__ == "__main__":
    unittest.main()
