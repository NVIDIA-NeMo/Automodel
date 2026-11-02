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
"""Offline regression checks for the dataset-card authoring contract."""

import json
from pathlib import Path

import pytest

from tools.docs.checks.dataset_cards import card_errors, hub_ids, validate, validate_dataset_cards

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

[View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/REPO_REVISION/examples/toy.yaml)

## Related Resources

[Pinned source](https://huggingface.co/datasets/example/toy/blob/REVISION/README.md)
""".replace("REPO_REVISION", "b" * 40).replace("REVISION", ENTRY["revision"])


def _assert_card_invalid(text: str, message: str) -> None:
    errors = card_errors(text, ENTRY)
    assert any(message in error for error in errors), errors


def test_valid_card():
    assert card_errors(CARD, ENTRY) == []


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('title: "example/toy"', 'title: "Toy"', "canonical Hub"),
        ("slug: dataset-coverage/", "slug: datasets/", "slug must"),
        ("| Input | A question. |", "", "describe Input"),
        ("synthetic example", "example", "labeled synthetic"),
        ('"answer": "4"', '"answer":', "invalid example JSON"),
        ('{"question": "What is 2 plus 2?", "answer": "4"}', "{}", "nonempty JSON"),
        ("| `answer` |", "| `completion` |", "example field 'answer' is absent from the schema"),
        ('"answer": "4"', '"answer": "4", "explanation": "Addition"', "example field 'explanation' is absent"),
        ("| Field | Type | Meaning |", "| Attribute | Type | Meaning |", "missing field schema table"),
        ("| Configuration | Split or Source | Rows |", "| Notes | Details | Rows |", "missing upstream split/source"),
        ("examples/toy.yaml", "examples/other.yaml", "recipe links"),
        (ENTRY["revision"], "main", "revision-pinned"),
    ],
)
def test_card_contract_violations(old: str, new: str, message: str):
    _assert_card_invalid(CARD.replace(old, new), message)


def test_extra_section_is_rejected():
    _assert_card_invalid(CARD + "\n## Extra\nMore.\n", "H2 sections")


def test_partial_example_can_omit_documented_fields():
    assert card_errors(CARD.replace(', "answer": "4"', ""), ENTRY) == []


@pytest.mark.parametrize(
    ("field", "schema_path", "value"),
    [
        ("answers", "answers.text", {"text": "4"}),
        ("conversations", "conversations[].role", [{"role": "assistant", "content": "4"}]),
    ],
)
def test_nested_schema_paths_document_example_root(field: str, schema_path: str, value: object):
    text = CARD.replace(
        '{"question": "What is 2 plus 2?", "answer": "4"}',
        json.dumps({"question": "What is 2 plus 2?", field: value}),
    ).replace("| `answer` | string | Answer. |", f"| `{schema_path}` | string | Answer field. |")
    assert card_errors(text, ENTRY) == []
    _assert_card_invalid(
        text.replace(f"`{schema_path}`", f"`other.{schema_path}`"), f"example field {field!r} is absent from the schema"
    )


def test_source_file_table_is_not_a_field_schema():
    text = CARD.replace("| Configuration | Split or Source | Rows |", "| Source File | Contents | Rows |").replace(
        "| default | train | - |", "| `train.jsonl` | Training records. | - |"
    )
    assert card_errors(text, ENTRY) == []


def test_source_file_row_cannot_document_example_field():
    text = (
        CARD.replace("| `answer` | string | Answer. |\n", "")
        .replace("| Configuration | Split or Source | Rows |", "| Source File | Contents | Rows |")
        .replace("| default | train | - |", "| `answer` | Training records. | - |")
    )
    _assert_card_invalid(text, "example field 'answer' is absent from the schema")


@pytest.mark.parametrize("ref", ["main", "v0.5.0", "b" * 39])
def test_repository_links_require_commit_pins(ref: str):
    _assert_card_invalid(CARD.replace("b" * 40, ref), "40-character commit SHA")


@pytest.mark.parametrize("section", ["Task", "Example Record"])
def test_fenced_headings_are_example_content(section: str):
    text = CARD.replace(f"## {section}\n", f"## {section}\n\n```text\n## Example source text\n```\n")
    assert card_errors(text, ENTRY) == []


def test_hub_ids_from_nested_recipes():
    value = {
        "dataset": {"dataset_name": "org/direct", "data_dir_list": [{"path": "hf://org/retrieval/subset"}]},
        "recipe_args": {"train_data_path": "org/chat", "val_data_path": "org/evaluation"},
        "model": {"pretrained_model_name_or_path": "org/model"},
    }
    assert hub_ids(value) == {"org/direct", "org/retrieval", "org/chat", "org/evaluation"}


@pytest.mark.parametrize(
    "path", ["./local", "./data/my_corpus", "../local", "/data/local", "data/train.jsonl", "${DATA}/train", "plain"]
)
def test_local_paths_are_not_hub_ids(path: str):
    assert hub_ids({"path_or_dataset_id": path}) == set()


class _Catalog:
    """A minimal repository holding one valid registered dataset card."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.write(ENTRY["card"], CARD)
        self.write("docs/dataset-coverage/catalog.json", json.dumps([ENTRY]))
        self.write("docs/dataset-coverage/index.mdx", "[Toy](/dataset-coverage/example/toy)")
        self.write("docs/fern/versions/nightly.yml", "path: ../../dataset-coverage/example/toy.mdx\n")
        self.write("examples/toy.yaml", "dataset:\n  path_or_dataset_id: example/toy\n")
        self.write("loader.py", "")

    def write(self, name: str, value: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def errors(self) -> list[str]:
        return validate(self.root)


@pytest.fixture
def catalog(tmp_path: Path) -> _Catalog:
    return _Catalog(tmp_path)


def test_valid_catalog(catalog: _Catalog):
    assert catalog.errors() == []
    validate_dataset_cards(catalog.root)


def test_shared_check_reports_every_catalog_error(catalog: _Catalog):
    (catalog.root / "examples/toy.yaml").unlink()
    catalog.write("examples/new.yml", "dataset:\n  dataset_name: new/dataset\n")
    with pytest.raises(ValueError, match="Dataset catalog validation failed") as error:
        validate_dataset_cards(catalog.root)
    assert "missing or invalid repository path: examples/toy.yaml" in str(error.value)
    assert "no card for Hub dataset new/dataset" in str(error.value)


def test_unregistered_card_is_checked(catalog: _Catalog):
    catalog.write("docs/dataset-coverage/new/invalid.md", "Invalid new card")
    assert any("unregistered" in error for error in catalog.errors())


def test_missing_recipe_is_rejected(catalog: _Catalog):
    (catalog.root / "examples/toy.yaml").unlink()
    assert any("missing or invalid repository path" in error for error in catalog.errors())


def test_new_example_dataset_needs_card(catalog: _Catalog):
    catalog.write("examples/new.yml", "dataset:\n  dataset_name: new/dataset\n")
    errors = catalog.errors()
    assert any("no card for Hub dataset new/dataset" in error for error in errors)
    assert any("docs/templates/dataset-card-guide.md" in error for error in errors)


def test_alias_matches_existing_card(catalog: _Catalog):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps([{**ENTRY, "aliases": ["legacy/toy"]}]))
    catalog.write("examples/toy.yaml", "dataset:\n  dataset_name: legacy/toy\n")
    assert catalog.errors() == []


@pytest.mark.parametrize(
    "navigation",
    [
        "",
        "navigation:\n"
        "  - page: Other\n"
        "    path: ../../other.mdx\n"
        "  # - page: Toy\n"
        "  #   path: ../../dataset-coverage/example/toy.mdx\n",
    ],
    ids=["empty", "commented"],
)
def test_missing_navigation(catalog: _Catalog, navigation: str):
    catalog.write("docs/fern/versions/nightly.yml", navigation)
    assert any("missing nightly navigation" in error for error in catalog.errors())


@pytest.mark.parametrize("quote", ["'", '"'])
def test_quoted_nested_navigation_is_valid(catalog: _Catalog, quote: str):
    catalog.write(
        "docs/fern/versions/nightly.yml",
        "navigation:\n"
        "  - section: Dataset Coverage\n"
        "    contents:\n"
        "      - page: Toy\n"
        f"        path: {quote}../../dataset-coverage/example/toy.mdx{quote}\n",
    )
    assert catalog.errors() == []


def test_malformed_navigation_reports_yaml_error(catalog: _Catalog):
    catalog.write("docs/fern/versions/nightly.yml", "path: ../../dataset-coverage/example/toy.mdx\nnavigation: [\n")
    assert any("invalid nightly navigation YAML" in error for error in catalog.errors())


@pytest.mark.parametrize("key", ["id", "task", "card", "source", "revision", "recipes", "aliases", "evidence"])
@pytest.mark.parametrize("value", [None, 123, {}, [None]])
def test_malformed_catalog_field_identifies_entry_and_field(catalog: _Catalog, key: str, value: object):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps([{**ENTRY, key: value}]))
    assert any("catalog entry 1" in error and key in error for error in catalog.errors())


@pytest.mark.parametrize(
    "entry",
    ["not an object", {key: value for key, value in ENTRY.items() if key != "recipes"}],
    ids=["not-object", "missing-recipes"],
)
def test_malformed_catalog_entry_identifies_entry(catalog: _Catalog, entry: object):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps([entry]))
    assert any("catalog entry 1" in error for error in catalog.errors())


def test_string_recipes_identify_dataset_and_field(catalog: _Catalog):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps([{**ENTRY, "recipes": "examples/toy.yaml"}]))
    assert any("example/toy" in error and "recipes" in error for error in catalog.errors())


@pytest.mark.parametrize("entries", [[ENTRY, ENTRY], [{**ENTRY, "aliases": [ENTRY["id"]]}]], ids=["id", "alias"])
def test_duplicate_ids_and_aliases_are_rejected(catalog: _Catalog, entries: list[dict]):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps(entries))
    assert any("duplicate dataset ID or alias" in error for error in catalog.errors())


def test_card_path_must_preserve_hub_id(catalog: _Catalog):
    catalog.write("docs/dataset-coverage/catalog.json", json.dumps([{**ENTRY, "card": "docs/toy.mdx"}]))
    assert any("path must preserve the canonical Hub ID" in error for error in catalog.errors())


def test_missing_index_link_is_rejected(catalog: _Catalog):
    catalog.write("docs/dataset-coverage/index.mdx", "No cards linked")
    assert any("missing catalog index link" in error for error in catalog.errors())


def test_invalid_example_yaml_identifies_file(catalog: _Catalog):
    catalog.write("examples/toy.yaml", "dataset: [\n")
    assert any("examples/toy.yaml: invalid YAML" in error for error in catalog.errors())
