# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Exercise the actual preview workflow's Data navigation staging step."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/fern-docs-preview.yml").read_text())
STAGE = next(
    step["run"] for step in WORKFLOW["jobs"]["preview"]["steps"] if step["name"] == "Stage validated Data navigation"
)
SCRIPT = STAGE.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


class PreviewNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "fern-preview/docs"
        self.target = self.docs / "fern/versions/nightly.yml"
        self.target.parent.mkdir(parents=True)
        self.source = self.root / "pr-source/docs/fern/versions/nightly.yml"
        self.source.parent.mkdir(parents=True)
        self.original = {
            "navigation": [
                {"section": "Model Coverage", "contents": [{"page": "Existing model", "path": "../../model.mdx"}]},
                {"section": "Data", "slug": "datasets", "contents": []},
            ]
        }
        self.target.write_text(yaml.safe_dump(self.original))
        self.page = self.docs / "dataset-coverage/example/toy.mdx"
        self.page.parent.mkdir(parents=True)
        self.page.write_text("A synthetic dataset card.")

    def _stage(self, data: dict) -> subprocess.CompletedProcess[str]:
        self.source.write_text(yaml.safe_dump({"navigation": [data]}))
        return subprocess.run(
            [sys.executable, "-I", "-c", SCRIPT],
            cwd=self.root,
            env={**os.environ, "RUNNER_TEMP": str(self.root)},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _reject(self, entry: dict) -> None:
        result = self._stage({"section": "Data", "slug": "datasets", "contents": [entry]})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(yaml.safe_load(self.target.read_text()), self.original)

    def test_all_dataset_cards_are_published_without_replacing_model_navigation(self) -> None:
        navigation = yaml.safe_load((ROOT / "docs/fern/versions/nightly.yml").read_text())["navigation"]
        data = next(item for item in navigation if item.get("slug") == "datasets")
        pending = [data]
        cards = set()
        while pending:
            item = pending.pop()
            pending.extend(item.get("contents", []))
            if "path" in item:
                page = (self.target.parent / item["path"]).resolve()
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text("Preview page.")
                if "dataset-coverage" in item["path"]:
                    cards.add(item["path"])
        self.assertEqual(len(cards), 36)  # 35 Hub datasets and the overview.
        result = self._stage(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = yaml.safe_load(self.target.read_text())
        self.assertEqual(actual["navigation"][0], self.original["navigation"][0])
        self.assertEqual(actual["navigation"][1], data)

    def test_rejects_paths_outside_staged_docs(self) -> None:
        (self.root / "outside.mdx").write_text("Private file")
        self._reject({"page": "Escape", "path": "../../../../outside.mdx"})

    def test_rejects_symlink_escape(self) -> None:
        outside = self.root / "outside.mdx"
        outside.write_text("Private file")
        self.page.unlink()
        self.page.symlink_to(outside)
        self._reject({"page": "Escape", "path": "../../dataset-coverage/example/toy.mdx"})

    def test_rejects_missing_pages_and_non_markdown_files(self) -> None:
        for path in ("../../missing.mdx", "../../fern/versions/nightly.yml"):
            with self.subTest(path=path):
                self._reject({"page": "Invalid page", "path": path})

    def test_rejects_configuration_and_folder_injection(self) -> None:
        for entry in ({"libraries": {}}, {"folder": "../../"}, {"js": "./script.js"}, {"icon": "./secret"}):
            with self.subTest(entry=entry):
                self._reject(entry)

    def test_rejects_environment_interpolation(self) -> None:
        self._reject({"page": "${FERN_TOKEN}", "path": "../../dataset-coverage/example/toy.mdx"})

    def test_rejects_link_entries(self) -> None:
        for href in (
            "https://example.com",
            "//example.com",
            "/nemo/automodel/../../secret",
            "/nemo/automodel/nightly/model-coverage/overview",
        ):
            with self.subTest(href=href):
                self._reject({"link": "Models", "href": href})

    def test_rejects_recursive_aliases(self) -> None:
        entry = {"section": "Loop", "contents": []}
        entry["contents"].append(entry)
        self._reject(entry)


if __name__ == "__main__":
    unittest.main()
