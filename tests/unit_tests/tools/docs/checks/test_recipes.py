# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Regression tests for recipe documentation tools."""

import os
from pathlib import Path

import pytest

from tools.docs.checks import recipes as recipe_coverage
from tools.docs.checks.recipes import (
    _collect_recipe_model_ids,
    validate_recipe_architectures,
    validate_recipe_publisher_aliases,
    validate_recipe_publishers,
)

pytestmark = pytest.mark.timeout(70)


def test_recipe_archs_have_doc_coverage():
    if os.environ.get("NEMO_DOC_COVERAGE_RESOLVE_ARCHS", "0") != "1":
        pytest.skip("Set NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1 for optional HF config resolution")
    validate_recipe_architectures(Path(__file__).resolve().parents[5])


def test_yaml_recipe_scan_finds_model_ids():
    assert len(_collect_recipe_model_ids(Path(__file__).resolve().parents[5] / "examples")) >= 10


def test_recipe_model_ids_live_under_publishing_org_dir():
    validate_recipe_publishers(Path(__file__).resolve().parents[5])


def test_hf_org_to_doc_slug_targets_exist():
    validate_recipe_publisher_aliases(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize("publisher", ["example", "wrong-publisher"])
def test_recipe_publisher_validator_checks_the_card_slug(tmp_path: Path, publisher: str):
    recipe = tmp_path / "examples/model.yaml"
    recipe.parent.mkdir()
    recipe.write_text("model:\n  pretrained_model_name_or_path: Example/Model-1B\n")
    card = tmp_path / "docs/model-coverage/llm/old-provider/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text(
        f"---\nslug: model-coverage/large-language-models/{publisher}/Model-1B\n---\n"
        "https://huggingface.co/Example/Model-1B\n"
    )

    if publisher == "example":
        recipe_coverage.validate_recipe_publishers(tmp_path)
    else:
        with pytest.raises(ValueError, match="documented outside their publisher routes"):
            recipe_coverage.validate_recipe_publishers(tmp_path)


def test_recipe_scanner_ignores_local_paths_and_finds_nested_model_ids(tmp_path: Path):
    (tmp_path / "recipe.yaml").write_text(
        "models:\n  - pretrained_model_name_or_path: Example/Model-1B\n"
        "  - pretrained_model_name_or_path: /path/to/checkpoint\n"
        "  - pretrained_model_name_or_path: ./local/checkpoint\n"
        "  - pretrained_model_name_or_path: your-org/placeholder\n"
    )

    assert recipe_coverage._collect_recipe_model_ids(tmp_path) == {"Example/Model-1B"}


@pytest.mark.parametrize("exists", [True, False])
def test_recipe_publisher_alias_validator_detects_removed_directories(tmp_path: Path, monkeypatch, exists: bool):
    monkeypatch.setattr(recipe_coverage, "_HF_ORG_TO_DOC_SLUG", {"OldOrg": "new-org"})
    if exists:
        (tmp_path / "docs/model-coverage/llm/new-org").mkdir(parents=True)
        recipe_coverage.validate_recipe_publisher_aliases(tmp_path)
    else:
        with pytest.raises(ValueError, match="non-existent doc directories"):
            recipe_coverage.validate_recipe_publisher_aliases(tmp_path)
