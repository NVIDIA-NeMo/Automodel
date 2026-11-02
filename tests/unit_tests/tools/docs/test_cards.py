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

"""Model-card identity follows slugs and navigation, independent of source layout."""

from pathlib import Path

import pytest
import yaml

from tools.docs.build.coverage_tables import _load_model_doc_catalog
from tools.docs.cards import _model_card_aliases
from tools.docs.checks.architecture import validate_architecture_reference_tables, validate_registered_architectures
from tools.docs.checks.model_cards import _validate_card_aliases
from tools.docs.checks.recipes import validate_recipe_publishers


def _write_card(repo: Path, relative: str, slug: str) -> Path:
    path = repo / "docs" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"title": "Released-7B", "slug": slug}, explicit_start=True)
        + "---\n[Checkpoint](https://huggingface.co/org/Released-7B)\n"
        + "| **Architecture** | `ReleasedModel` |\n"
    )
    return path


def _write_navigation(repo: Path, paths: list[str]) -> None:
    config = repo / "docs/fern/versions/nightly.yml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(
            {
                "navigation": [
                    {
                        "section": "Models",
                        "contents": [{"page": "Released-7B", "path": "../../" + path} for path in paths],
                    }
                ]
            }
        )
    )


@pytest.mark.parametrize(
    "source",
    [
        "model-coverage/llm/wrong-org/preview.mdx",
        "model-coverage/legacy.mdx",
        "model-coverage/legacy/deep/nested/preview.mdx",
        "guides/old-announcement.mdx",
    ],
)
def test_catalog_uses_slug_category_and_publisher_instead_of_source_directory(tmp_path: Path, source: str):
    slug = "model-coverage/vision-language-models/org/Released-7B"
    card = _write_card(tmp_path, source, slug)
    before = card.read_bytes()
    _write_navigation(tmp_path, [source])

    models, architectures, catalog = _load_model_doc_catalog(tmp_path / "docs")

    assert len(catalog) == 1
    model = catalog[0]
    assert model.model_type == "VLM"
    assert model.docs_page == "/" + slug
    assert model.source_path == "docs/" + source
    assert models["org/Released-7B"] == [model]
    assert architectures == {"ReleasedModel"}
    assert card.read_bytes() == before


@pytest.mark.parametrize("selected", ["aaa-legacy.mdx", "zzz-current.mdx"])
def test_shared_slug_uses_navigation_to_select_canonical_source(tmp_path: Path, selected: str):
    slug = "model-coverage/large-language-models/org/Released-7B"
    sources = ["model-coverage/llm/org/" + name for name in ("aaa-legacy.mdx", "zzz-current.mdx")]
    for source in sources:
        _write_card(tmp_path, source, slug)
    canonical = "model-coverage/llm/org/" + selected
    _write_navigation(tmp_path, [canonical])

    _validate_card_aliases(tmp_path)
    _, _, catalog = _load_model_doc_catalog(tmp_path / "docs")

    assert [model.source_path for model in catalog] == ["docs/" + canonical]
    aliases = _model_card_aliases(tmp_path)
    assert list(aliases.values()) == [tmp_path / "docs" / canonical]
    assert all((tmp_path / "docs" / source).is_file() for source in sources)


@pytest.mark.parametrize("selection", [[], ["a.mdx", "b.mdx"], ["a.mdx", "a.mdx"]])
def test_duplicate_slugs_require_exactly_one_navigation_entry(tmp_path: Path, selection: list[str]):
    for name in ("a.mdx", "b.mdx"):
        _write_card(tmp_path, "model-coverage/llm/org/" + name, "model-coverage/large-language-models/org/Released-7B")
    _write_navigation(tmp_path, ["model-coverage/llm/org/" + name for name in selection])

    with pytest.raises(ValueError, match="must select exactly one file in nightly navigation"):
        _load_model_doc_catalog(tmp_path / "docs")


def test_unlisted_compatibility_copy_cannot_diverge(tmp_path: Path):
    slug = "model-coverage/large-language-models/org/Released-7B"
    canonical = "model-coverage/llm/org/current.mdx"
    _write_card(tmp_path, canonical, slug)
    alias = _write_card(tmp_path, "guides/shared-preview.mdx", slug)
    alias.write_text(alias.read_text() + "Stale instructions\n")
    _write_navigation(tmp_path, [canonical])

    with pytest.raises(ValueError, match="Compatibility card must stay synchronized"):
        _validate_card_aliases(tmp_path)


def test_shared_source_outside_model_coverage_participates_in_all_source_checks(tmp_path: Path):
    (tmp_path / "docs/model-coverage").mkdir(parents=True)
    card = _write_card(tmp_path, "guides/shared-preview.mdx", "model-coverage/large-language-models/org/Released-7B")
    card.write_text(card.read_text() + "## Model Reference\n### Model Architecture\n| Property | Value |\n")
    recipe = tmp_path / "examples/model.yaml"
    recipe.parent.mkdir()
    recipe.write_text("model:\n  pretrained_model_name_or_path: org/Released-7B\n")
    registry = tmp_path / "nemo_automodel/_transformers/registry.py"
    registry.parent.mkdir(parents=True)
    registry.write_text('MODEL_ARCH_MAPPING = OrderedDict([("ReleasedModel", ("models.released", "ReleasedModel"))])\n')

    validate_recipe_publishers(tmp_path)
    validate_architecture_reference_tables(tmp_path)
    validate_registered_architectures(tmp_path)

    card.write_text(card.read_text().replace("/large-language-models/org/", "/large-language-models/wrong-org/"))
    with pytest.raises(ValueError, match="documented outside their publisher routes"):
        validate_recipe_publishers(tmp_path)
    card.write_text(card.read_text().replace("### Model Architecture", "### Wrong Heading"))
    with pytest.raises(ValueError, match="missing '### Model Architecture'"):
        validate_architecture_reference_tables(tmp_path)
