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
"""Offline regression checks for the dataset-card authoring contract."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("dataset_cards", ROOT / "tools/validate_dataset_cards.py")
cards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cards)

ENTRY = {
    "id": "example/toy",
    "task": "Question answering",
    "card": "docs/dataset-coverage/example/toy.mdx",
    "recipes": ["examples/toy.yaml"],
    "source": "loader.py",
    "revision": "a" * 40,
}
CARD = """---
title: "example/toy"
description: "A toy dataset for the documentation test."
slug: dataset-coverage/example/toy
---

[Hub](https://huggingface.co/datasets/example/toy)

## Task

Question answering.

| Property | Description |
| --- | --- |
| Input | A question. |
| Target | An answer. |
| Modality | Text. |
| License Metadata | See source. |

## Example Record

This is a synthetic example.

```json
{"question": "What is 2 plus 2?", "answer": "4"}
```

## Schema and Splits

| Field | Type | Meaning |
| --- | --- | --- |
| `question` | string | Question. |
| `answer` | string | Answer. |

| Configuration | Split or Source | Rows |
| --- | --- | --- |
| default | train | - |

## Use with NeMo AutoModel

[View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/toy.yaml)

## Related Resources

[Pinned source](https://huggingface.co/datasets/example/toy/blob/REVISION/README.md)
""".replace("REVISION", ENTRY["revision"])


class DatasetCardContractTests(unittest.TestCase):
    def test_valid_card(self):
        self.assertEqual(cards.card_errors(CARD, ENTRY), [])

    def assert_invalid(self, text, message):
        self.assertTrue(any(message in error for error in cards.card_errors(text, ENTRY)))

    def test_exact_dataset_identity(self):
        self.assert_invalid(CARD.replace('title: "example/toy"', 'title: "Toy"'), "canonical Hub")

    def test_stable_route(self):
        self.assert_invalid(CARD.replace("slug: dataset-coverage/", "slug: datasets/"), "slug must")

    def test_missing_task_input(self):
        self.assert_invalid(CARD.replace("| Input | A question. |", ""), "describe Input")

    def test_example_must_be_labeled(self):
        self.assert_invalid(CARD.replace("synthetic example", "example"), "labeled synthetic")

    def test_invalid_json(self):
        self.assert_invalid(CARD.replace('"answer": "4"', '"answer":'), "invalid example JSON")

    def test_empty_json_record(self):
        self.assert_invalid(CARD.replace('{"question": "What is 2 plus 2?", "answer": "4"}', "{}"), "nonempty JSON")

    def test_schema_disagrees_with_example(self):
        self.assert_invalid(
            CARD.replace("| `answer` |", "| `completion` |"), "example field 'answer' is absent from the schema"
        )

    def test_partial_example_can_omit_documented_fields(self):
        text = CARD.replace(', "answer": "4"', "")
        self.assertEqual(cards.card_errors(text, ENTRY), [])

    def test_example_cannot_add_undocumented_fields(self):
        text = CARD.replace('"answer": "4"', '"answer": "4", "explanation": "Addition"')
        self.assert_invalid(text, "example field 'explanation' is absent from the schema")

    def test_nested_schema_paths_document_example_root(self):
        for field, schema_path, value in (
            ("answers", "answers.text", {"text": "4"}),
            ("conversations", "conversations[].role", [{"role": "assistant", "content": "4"}]),
        ):
            with self.subTest(schema_path=schema_path):
                text = CARD.replace(
                    '{"question": "What is 2 plus 2?", "answer": "4"}',
                    json.dumps({"question": "What is 2 plus 2?", field: value}),
                ).replace("| `answer` | string | Answer. |", f"| `{schema_path}` | string | Answer field. |")
                self.assertEqual(cards.card_errors(text, ENTRY), [])
                self.assert_invalid(
                    text.replace(f"`{schema_path}`", f"`other.{schema_path}`"),
                    f"example field {field!r} is absent from the schema",
                )

    def test_source_file_table_is_not_a_field_schema(self):
        text = CARD.replace("| Configuration | Split or Source | Rows |", "| Source File | Contents | Rows |").replace(
            "| default | train | - |", "| `train.jsonl` | Training records. | - |"
        )
        self.assertEqual(cards.card_errors(text, ENTRY), [])

    def test_source_file_row_cannot_document_example_field(self):
        text = (
            CARD.replace("| `answer` | string | Answer. |\n", "")
            .replace("| Configuration | Split or Source | Rows |", "| Source File | Contents | Rows |")
            .replace("| default | train | - |", "| `answer` | Training records. | - |")
        )
        self.assert_invalid(text, "example field 'answer' is absent from the schema")

    def test_missing_field_schema_table(self):
        self.assert_invalid(
            CARD.replace("| Field | Type | Meaning |", "| Attribute | Type | Meaning |"), "missing field schema table"
        )

    def test_missing_split_or_source_table(self):
        self.assert_invalid(
            CARD.replace("| Configuration | Split or Source | Rows |", "| Notes | Details | Rows |"),
            "missing upstream split/source table",
        )

    def test_wrong_recipe(self):
        self.assert_invalid(CARD.replace("examples/toy.yaml", "examples/other.yaml"), "recipe links")

    def test_unpinned_source(self):
        self.assert_invalid(CARD.replace(ENTRY["revision"], "main"), "revision-pinned")

    def test_extra_section(self):
        self.assert_invalid(CARD + "\n## Extra\nMore.\n", "H2 sections")

    def test_fenced_headings_are_example_content(self):
        for section in ("Task", "Example Record"):
            with self.subTest(section=section):
                text = CARD.replace(
                    f"## {section}\n",
                    f"## {section}\n\n```text\n## Example source text\n```\n",
                )
                self.assertEqual(cards.card_errors(text, ENTRY), [])

    def test_hub_ids_from_nested_recipes(self):
        value = {
            "dataset": {"dataset_name": "org/direct", "data_dir_list": [{"path": "hf://org/retrieval/subset"}]},
            "recipe_args": {"train_data_path": "org/chat"},
            "model": {"pretrained_model_name_or_path": "org/model"},
        }
        self.assertEqual(cards.hub_ids(value), {"org/direct", "org/retrieval", "org/chat"})

    def test_local_paths_are_not_hub_ids(self):
        for path in ("./local", "../local", "/data/local", "data/train.jsonl", "${DATA}/train", "plain"):
            self.assertEqual(cards.hub_ids({"path_or_dataset_id": path}), set())


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write(ENTRY["card"], CARD)
        self.write("docs/dataset-coverage/catalog.json", json.dumps([ENTRY]))
        self.write("docs/dataset-coverage/index.mdx", "[Toy](/dataset-coverage/example/toy)")
        self.write("docs/fern/versions/nightly.yml", "path: ../../dataset-coverage/example/toy.mdx\n")
        self.write("examples/toy.yaml", "dataset:\n  path_or_dataset_id: example/toy\n")
        self.write("loader.py", "")

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def test_valid_catalog(self):
        self.assertEqual(cards.validate(self.root), [])

    def test_unregistered_card_is_checked(self):
        self.write("docs/dataset-coverage/new/invalid.md", "Invalid new card")
        self.assertTrue(any("unregistered" in error for error in cards.validate(self.root)))

    def test_missing_recipe_is_rejected(self):
        (self.root / "examples/toy.yaml").unlink()
        self.assertTrue(any("missing or invalid repository path" in error for error in cards.validate(self.root)))

    def test_new_example_dataset_needs_card(self):
        self.write("examples/new.yml", "dataset:\n  dataset_name: new/dataset\n")
        self.assertTrue(any("no card for Hub dataset new/dataset" in error for error in cards.validate(self.root)))

    def test_alias_matches_existing_card(self):
        self.write("docs/dataset-coverage/catalog.json", json.dumps([{**ENTRY, "aliases": ["legacy/toy"]}]))
        self.write("examples/toy.yaml", "dataset:\n  dataset_name: legacy/toy\n")
        self.assertEqual(cards.validate(self.root), [])

    def test_missing_navigation(self):
        self.write("docs/fern/versions/nightly.yml", "")
        self.assertTrue(any("missing nightly navigation" in error for error in cards.validate(self.root)))

    def test_commented_navigation_is_missing(self):
        self.write(
            "docs/fern/versions/nightly.yml",
            "navigation:\n"
            "  - page: Other\n"
            "    path: ../../other.mdx\n"
            "  # - page: Toy\n"
            "  #   path: ../../dataset-coverage/example/toy.mdx\n",
        )
        self.assertTrue(any("missing nightly navigation" in error for error in cards.validate(self.root)))

    def test_quoted_nested_navigation_is_valid(self):
        for quote in ("'", '"'):
            with self.subTest(quote=quote):
                self.write(
                    "docs/fern/versions/nightly.yml",
                    "navigation:\n"
                    "  - section: Dataset Coverage\n"
                    "    contents:\n"
                    "      - page: Toy\n"
                    f"        path: {quote}../../dataset-coverage/example/toy.mdx{quote}\n",
                )
                self.assertEqual(cards.validate(self.root), [])

    def test_malformed_navigation_reports_yaml_error(self):
        self.write(
            "docs/fern/versions/nightly.yml",
            "path: ../../dataset-coverage/example/toy.mdx\nnavigation: [\n",
        )
        self.assertTrue(any("invalid nightly navigation YAML" in error for error in cards.validate(self.root)))


if __name__ == "__main__":
    unittest.main()
