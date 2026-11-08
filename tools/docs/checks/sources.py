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

"""Coordinate documentation source validation before rendering."""

import logging
import subprocess
from pathlib import Path

import yaml

from tools.docs.build.coverage_tables import (
    _generate_tables,
    _load_model_doc_catalog,
    _validate_generated_tables_are_not_committed,
)
from tools.docs.catalog import (
    MODEL_TYPE_OVERVIEW_PATHS,
    SUPPORT_LOG_END_MARKER,
    SUPPORT_LOG_START_MARKER,
)
from tools.docs.checks.architecture import (
    validate_architecture_aliases,
    validate_architecture_reference_tables,
    validate_registered_architectures,
)
from tools.docs.checks.model_cards import (
    _validate_card_aliases,
    _validate_changed_card_identity,
    _validate_commands,
    _validate_descriptions,
)
from tools.docs.checks.navigation import (
    _validate_internal_links,
    _validate_provider_sections,
    _validate_published_training_routes,
    _validate_recipe_routes,
    _validate_redirects,
    _validate_release_navigation,
)
from tools.docs.checks.recipes import (
    validate_recipe_publisher_aliases,
    validate_recipe_publishers,
)


def _validate_generated_tables(repo_root: Path) -> None:
    _validate_generated_tables_are_not_committed(repo_root)


def _validate_overview_tables(repo_root: Path) -> None:
    for _, relative_path in MODEL_TYPE_OVERVIEW_PATHS:
        document = (repo_root / "docs" / "model-coverage" / relative_path).read_text(encoding="utf-8")
        if document.count("## Supported Models") != 1:
            raise ValueError(f"{relative_path}: expected exactly one Supported Models section")
        if "## Model Support Log" in document:
            raise ValueError(f"{relative_path}: remove the legacy Model Support Log section")
        if document.count(SUPPORT_LOG_START_MARKER) != 1:
            raise ValueError(f"{relative_path}: expected exactly one support log start marker")
        if document.count(SUPPORT_LOG_END_MARKER) != 1:
            raise ValueError(f"{relative_path}: expected exactly one support log end marker")


def _validate_generated_coverage(repo_root: Path) -> None:
    (_, _, documented_models) = _load_model_doc_catalog(repo_root / "docs")
    generated = _generate_tables(repo_root)
    overview_paths = {
        model_type: repo_root / "docs" / "model-coverage" / relative_path
        for (model_type, relative_path) in MODEL_TYPE_OVERVIEW_PATHS
    }
    overview_paths["Encoder-Decoder"] = overview_paths["LLM"]
    missing = [
        (model.model_type, model.docs_page)
        for model in documented_models
        if model.docs_page not in generated[overview_paths[model.model_type]]
    ]
    if missing:
        raise ValueError(f"Documented model families missing from generated support tables: {missing}")


def validate_sources(repo_root: Path) -> None:
    """Validate repository documentation sources before generating derived files.

    Args:
        repo_root: AutoModel repository root.
    """
    errors: list[str] = []
    for check in (
        _validate_published_training_routes,
        _validate_release_navigation,
        _validate_generated_tables,
        _validate_overview_tables,
        _validate_card_aliases,
        _validate_changed_card_identity,
        _validate_provider_sections,
        _validate_descriptions,
        _validate_commands,
        _validate_recipe_routes,
        _validate_redirects,
        _validate_internal_links,
        _validate_generated_coverage,
        validate_architecture_reference_tables,
        validate_registered_architectures,
        validate_architecture_aliases,
        validate_recipe_publishers,
        validate_recipe_publisher_aliases,
    ):
        try:
            check(repo_root)
        except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
            errors.append(f"{check.__name__}: {error}")
    if errors:
        raise ValueError("Documentation source validation failed:\n" + "\n".join(errors))


def main() -> int:
    """Validate documentation sources and return a failing exit status on errors."""
    try:
        validate_sources(Path(__file__).resolve().parents[3])
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    logging.getLogger(__name__).info("Validated documentation source contracts.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
