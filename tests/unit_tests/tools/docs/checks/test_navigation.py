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


# Over the default 5s budget on purpose: this module drives git through subprocesses over throwaway repositories.
# Shrink the work or the process count before raising this further.

import shutil
from pathlib import Path

import pytest

from tools.docs.build.coverage_tables import _load_model_doc_catalog
from tools.docs.checks.navigation import (
    _validate_internal_links,
    _validate_provider_sections,
    _validate_published_training_routes,
    _validate_recipe_routes,
    _validate_redirects,
    _validate_release_navigation,
)

pytestmark = pytest.mark.timeout(70)


def test_regrouped_training_pages_preserve_published_urls():
    _validate_published_training_routes(Path(__file__).resolve().parents[5])


def test_model_release_docs_pages_exist_in_nightly_navigation():
    _validate_release_navigation(Path(__file__).resolve().parents[5])


def test_model_coverage_pages_use_provider_sections_and_checkpoint_slugs():
    _validate_provider_sections(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize("checkpoint_slug", ["FLUX.1-dev", "Wrong-Checkpoint"])
def test_checkpoint_route_uses_metadata_while_preserving_the_source_path(tmp_path: Path, checkpoint_slug: str):
    repo_root = Path(__file__).resolve().parents[5]
    shutil.copytree(repo_root / "docs/model-coverage", tmp_path / "docs/model-coverage")
    for relative in ("docs/fern/docs.yml", "docs/fern/versions/nightly.yml"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, destination)
    source_path = "docs/model-coverage/diffusion/black-forest-labs/flux.mdx"
    card = tmp_path / source_path
    old_slug = "slug: model-coverage/diffusion/black-forest-labs/FLUX.1-dev"
    document = card.read_text()
    assert old_slug in document
    card.write_text(
        document.replace(old_slug, f"slug: model-coverage/diffusion/black-forest-labs/{checkpoint_slug}", 1)
    )

    if checkpoint_slug == "FLUX.1-dev":
        _validate_provider_sections(tmp_path)
        _, _, catalog = _load_model_doc_catalog(tmp_path / "docs")
        model = next(model for model in catalog if model.source_path == source_path)
        assert model.title == "FLUX.1-dev"
        assert model.docs_page == "/model-coverage/diffusion/black-forest-labs/FLUX.1-dev"
        assert card.is_file()
    else:
        with pytest.raises(ValueError, match="page title 'FLUX.1-dev' does not match 'Wrong-Checkpoint'"):
            _validate_provider_sections(tmp_path)


def test_recipe_backed_model_sizes_have_exact_index_routes_and_one_card():
    _validate_recipe_routes(Path(__file__).resolve().parents[5])


def test_retired_model_routes_redirect_to_provider_indexes():
    _validate_redirects(Path(__file__).resolve().parents[5])


def test_internal_links_from_model_coverage_resolve_to_nightly_routes():
    _validate_internal_links(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize(
    "source",
    [
        "docs/model-coverage/llm/wrong-provider/deep/preview.mdx",
        "docs/guides/shared-announcement.mdx",
    ],
)
def test_navigation_accepts_a_card_at_a_previously_shared_source_location(tmp_path: Path, source: str):
    repo_root = Path(__file__).resolve().parents[5]
    shutil.copytree(repo_root / "docs/model-coverage", tmp_path / "docs/model-coverage")
    for relative in ("docs/fern/docs.yml", "docs/fern/versions/nightly.yml"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, target)
    current = "docs/model-coverage/diffusion/black-forest-labs/flux.mdx"
    card = tmp_path / current
    document = card.read_bytes()
    shared = tmp_path / source
    shared.parent.mkdir(parents=True, exist_ok=True)
    card.rename(shared)
    config = tmp_path / "docs/fern/versions/nightly.yml"
    config.write_text(
        config.read_text().replace("../../" + current.removeprefix("docs/"), "../../" + source.removeprefix("docs/"))
    )

    _validate_provider_sections(tmp_path)
    _, _, catalog = _load_model_doc_catalog(tmp_path / "docs")

    model = next(model for model in catalog if model.source_path == source)
    assert model.model_type == "Diffusion"
    assert model.docs_page == "/model-coverage/diffusion/black-forest-labs/FLUX.1-dev"
    assert shared.read_bytes() == document
    assert not card.exists()
