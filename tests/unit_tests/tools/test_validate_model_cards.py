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

from pathlib import Path

import pytest

from tools.docs import recipe_coverage
from tools.docs import validate_model_cards as checks
from tools.docs.architecture import validate_architecture_reference_tables


@pytest.mark.parametrize("synchronized", [True, False])
def test_flux_compatibility_copy_must_match_its_canonical_card(tmp_path: Path, monkeypatch, synchronized: bool):
    alias = "docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx"
    canonical = "docs/model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base.mdx"
    monkeypatch.setattr(checks, "MODEL_CARD_PATH_ALIASES", {alias: canonical})
    content = (
        '---\ntitle: "flux-3-image-klein-base"\n'
        "slug: model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base\n---\nBody\n"
    )
    (tmp_path / canonical).parent.mkdir(parents=True)
    (tmp_path / canonical).write_text(content)
    (tmp_path / alias).write_text(content if synchronized else content + "Compatibility-only edit\n")

    if synchronized:
        checks._validate_card_aliases(tmp_path)
    else:
        with pytest.raises(ValueError, match="Compatibility card must stay synchronized"):
            checks._validate_card_aliases(tmp_path)


@pytest.mark.parametrize(
    ("description", "message"),
    [
        ("", "length 0"),
        ("A" * 180, "outside 120-170"),
        ("A" * 130, "does not contain the exact title"),
        ("Model-1B NeMo AutoModel " + "a" * 100 + "é", "non-ASCII"),
    ],
)
def test_description_validator_rejects_invalid_metadata(tmp_path: Path, description: str, message: str):
    card = tmp_path / "docs/model-coverage/llm/org/model-1b.mdx"
    card.parent.mkdir(parents=True)
    card.write_text(f'---\ntitle: "Model-1B"\ndescription: "{description}"\n---\n')

    with pytest.raises(ValueError, match=message):
        checks._validate_descriptions(tmp_path)


def test_description_validator_accepts_a_valid_model_card(tmp_path: Path):
    card = tmp_path / "docs/model-coverage/llm/org/model-1b.mdx"
    card.parent.mkdir(parents=True)
    description = (
        "Fine-tune Model-1B with NeMo AutoModel using repository-backed training recipes, "
        "checkpoint loading, and distributed configuration."
    )
    card.write_text(f'---\ntitle: "Model-1B"\ndescription: "{description}"\n---\n')

    checks._validate_descriptions(tmp_path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "missing '## Model Reference'"),
        ("## Model Reference\n", "missing '### Model Architecture'"),
        ("## Model Reference\n### Model Architecture\n", "missing architecture Property/Value table"),
        (
            "## Model Reference\n### Model Architecture\n| Property | Value |\n### Model Summary\n",
            "legacy '### Model Summary'",
        ),
        ("## Model Reference\n### Model Architecture\n| Property | Value |\né", "non-ASCII"),
    ],
)
def test_architecture_reference_validator_reports_bad_cards(tmp_path: Path, body: str, message: str):
    card = tmp_path / "docs/model-coverage/llm/org/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text("---\nslug: model-coverage/large-language-models/org/model\n---\n" + body)

    with pytest.raises(ValueError, match=message):
        validate_architecture_reference_tables(tmp_path)


def test_architecture_reference_validator_accepts_the_shared_contract(tmp_path: Path):
    card = tmp_path / "docs/model-coverage/llm/org/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text(
        "---\nslug: model-coverage/large-language-models/org/model\n---\n"
        "## Model Reference\n### Model Architecture\n| Property | Value |\n|---|---|\n"
    )

    validate_architecture_reference_tables(tmp_path)


@pytest.mark.parametrize("publisher", ["example", "wrong-publisher"])
def test_recipe_publisher_validator_checks_the_card_directory(tmp_path: Path, publisher: str):
    recipe = tmp_path / "examples/model.yaml"
    recipe.parent.mkdir()
    recipe.write_text("model:\n  pretrained_model_name_or_path: Example/Model-1B\n")
    card = tmp_path / f"docs/model-coverage/llm/{publisher}/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text("https://huggingface.co/Example/Model-1B\n")

    if publisher == "example":
        recipe_coverage.validate_recipe_publishers(tmp_path)
    else:
        with pytest.raises(ValueError, match="documented outside the org directory"):
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
