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

"""Generate model-coverage tables from their canonical repository sources."""

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse

TABLE_ROW_COUNT = 10
HOMEPAGE_START_MARKER = "{/* BEGIN GENERATED LATEST MODEL SUPPORT */}"
HOMEPAGE_END_MARKER = "{/* END GENERATED LATEST MODEL SUPPORT */}"
SUPPORT_LOG_START_MARKER = "{/* BEGIN GENERATED MODEL SUPPORT LOG */}"
SUPPORT_LOG_END_MARKER = "{/* END GENERATED MODEL SUPPORT LOG */}"
REGISTRY_START_MARKER = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"
REGISTRY_END_MARKER = "{/* END GENERATED MODEL ARCHITECTURES */}"
HF_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RELEASE_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RECIPE_PATH_PATTERN = re.compile(r"^examples/[A-Za-z0-9_./-]+\.yaml$")
DOCS_PAGE_PATTERN = re.compile(r"^/[a-z0-9][a-z0-9/-]*$")
MODEL_TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 -]*$")
MARKDOWN_UNSAFE_PATTERN = re.compile(r"[\[\]|<>`\r\n]")
REPOSITORY_URL = "https://github.com/NVIDIA-NeMo/Automodel/blob/main"
COMPACT_TABLE_STYLE = """<style>{`
  .compact-model-tables .fern-table-root {
    width: 100% !important;
  }

  .compact-model-tables .fern-table {
    width: 100% !important;
    min-width: 0 !important;
    table-layout: auto !important;
  }

  .compact-model-tables .fern-table th,
  .compact-model-tables .fern-table td {
    text-align: left !important;
  }

  .compact-model-tables .fern-table th:nth-child(-n + 2),
  .compact-model-tables .fern-table td:nth-child(-n + 2) {
    width: 1%;
    white-space: nowrap;
  }

  .compact-model-tables .fern-table th:last-child,
  .compact-model-tables .fern-table td:last-child {
    width: 100%;
  }
`}</style>"""
_BrevStatus = Literal["available", "planned", "unavailable"]


@dataclass(frozen=True)
class _ModelRelease:
    release_date: str
    model: str
    hf_model_id: str
    architectures: tuple[str, ...]
    docs_page: str
    model_type: str
    recipe: str
    brev_status: _BrevStatus
    brev_url: str | None


def _replace_generated_block(document: str, start_marker: str, end_marker: str, generated_block: str) -> str:
    start = document.find(start_marker)
    end = document.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Expected one ordered marker pair: {start_marker}, {end_marker}")
    if document.find(start_marker, start + len(start_marker)) != -1:
        raise ValueError(f"Found multiple start markers: {start_marker}")
    if document.find(end_marker, end + len(end_marker)) != -1:
        raise ValueError(f"Found multiple end markers: {end_marker}")
    return document[:start] + generated_block + document[end + len(end_marker) :]


def _require_catalog_text(raw_entry: dict[str, object], field: str, index: int) -> str:
    value = raw_entry.get(field)
    if not isinstance(value, str) or not value or MARKDOWN_UNSAFE_PATTERN.search(value):
        raise ValueError(f"Model release {index} field {field!r} must be a non-empty Markdown-safe string")
    return value


def _load_new_model_creation_dates(repo_root: Path, hf_model_ids: set[str]) -> dict[str, str]:
    """Return first-YAML dates for model IDs introduced by the current branch.

    Squash merges discard the originating branch's file history, so dates for
    existing models remain stored in the catalog. For model IDs introduced by
    the current branch, full Git history still contains their first YAML
    occurrence and lets CI validate the catalog before that provenance is lost.
    """
    if not (repo_root / ".git").exists():
        return {}

    def run_git(arguments: list[str], *, allow_no_match: bool = False) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ValueError(f"Could not inspect recipe creation history: {error}") from error
        if result.returncode != 0 and not (allow_no_match and result.returncode == 1):
            raise ValueError(f"Could not inspect recipe creation history: {result.stderr.strip()}")
        return result

    if run_git(["rev-parse", "--is-shallow-repository"]).stdout.strip() == "true":
        return {}

    base_ref = next(
        (
            candidate
            for candidate in ("origin/main", "main")
            if subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True,
            ).returncode
            == 0
        ),
        None,
    )
    if base_ref is None:
        return {}

    merge_base = run_git(["merge-base", "HEAD", base_ref]).stdout.strip()

    grep_patterns = [argument for hf_model_id in sorted(hf_model_ids) for argument in ("-e", hf_model_id)]

    def find_model_ids(revision: str) -> set[str]:
        result = run_git(["grep", "-F", *grep_patterns, revision, "--", "*.yaml"], allow_no_match=True)
        return {
            hf_model_id
            for hf_model_id in hf_model_ids
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(hf_model_id)}(?![A-Za-z0-9_.-])",
                result.stdout,
            )
        }

    new_model_ids = find_model_ids("HEAD") - find_model_ids(merge_base)
    if not new_model_ids:
        return {}

    creation_dates: dict[str, str] = {}
    for hf_model_id in new_model_ids:
        log_result = run_git(
            [
                "log",
                "--reverse",
                "--no-renames",
                f"-S{hf_model_id}",
                "--format=%as",
                f"{merge_base}..HEAD",
                "--",
                "*.yaml",
            ]
        )
        first_date = next(
            (line for line in log_result.stdout.splitlines() if RELEASE_DATE_PATTERN.fullmatch(line)), None
        )
        if first_date is None:
            raise ValueError(f"Model {hf_model_id!r} is missing a first YAML commit")
        creation_dates[hf_model_id] = first_date
    return creation_dates


def _load_model_releases(catalog_path: Path, repo_root: Path) -> list[_ModelRelease]:
    try:
        raw_entries = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model release catalog {catalog_path}: {error}") from error
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Model release catalog must contain a non-empty JSON list")

    releases = []
    seen_models = set()
    seen_hf_model_types = set()
    previous_date: str | None = None
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Model release {index} must be a JSON object")
        unknown_fields = set(raw_entry) - {
            "date",
            "model",
            "hf_model_id",
            "architectures",
            "docs_page",
            "type",
            "recipe",
            "brev_status",
            "brev_url",
        }
        if unknown_fields:
            raise ValueError(f"Model release {index} has unknown fields: {', '.join(sorted(unknown_fields))}")
        missing_fields = {
            "date",
            "model",
            "hf_model_id",
            "architectures",
            "docs_page",
            "type",
            "recipe",
            "brev_status",
        } - set(raw_entry)
        if missing_fields:
            raise ValueError(f"Model release {index} is missing fields: {', '.join(sorted(missing_fields))}")

        release_date = _require_catalog_text(raw_entry, "date", index)
        if not RELEASE_DATE_PATTERN.fullmatch(release_date):
            raise ValueError(f"Model release {index} date must use YYYY-MM-DD: {release_date!r}")
        try:
            date.fromisoformat(release_date)
        except ValueError as error:
            raise ValueError(f"Model release {index} has invalid date {release_date!r}") from error
        if previous_date is not None and previous_date < release_date:
            raise ValueError(
                f"Model release catalog is not reverse chronological at {previous_date} then {release_date}"
            )
        previous_date = release_date

        model = _require_catalog_text(raw_entry, "model", index)
        hf_model_id = _require_catalog_text(raw_entry, "hf_model_id", index)
        architectures_value = raw_entry.get("architectures")
        if not isinstance(architectures_value, list) or any(
            not isinstance(architecture, str) or not architecture or MARKDOWN_UNSAFE_PATTERN.search(architecture)
            for architecture in architectures_value
        ):
            raise ValueError(f"Model release {index} architectures must be a list of Markdown-safe strings")
        architectures = tuple(architectures_value)
        docs_page = _require_catalog_text(raw_entry, "docs_page", index)
        model_type = _require_catalog_text(raw_entry, "type", index)
        if not HF_MODEL_ID_PATTERN.fullmatch(hf_model_id):
            raise ValueError(f"Model release {index} has invalid Hugging Face model ID {hf_model_id!r}")
        if not DOCS_PAGE_PATTERN.fullmatch(docs_page) or "//" in docs_page or ".." in docs_page:
            raise ValueError(f"Model release {index} has invalid version-agnostic docs page {docs_page!r}")
        if not MODEL_TYPE_PATTERN.fullmatch(model_type):
            raise ValueError(f"Model release {index} has invalid type {model_type!r}")
        if model in seen_models:
            raise ValueError(f"Model release catalog contains duplicate model {model!r}")
        hf_model_type = (hf_model_id, model_type)
        if hf_model_type in seen_hf_model_types:
            raise ValueError(
                f"Model release catalog contains duplicate Hugging Face model ID {hf_model_id!r} "
                f"for type {model_type!r}"
            )
        seen_models.add(model)
        seen_hf_model_types.add(hf_model_type)

        recipe = _require_catalog_text(raw_entry, "recipe", index)
        recipe_path = Path(recipe)
        if (
            recipe_path.is_absolute()
            or not RECIPE_PATH_PATTERN.fullmatch(recipe)
            or not recipe_path.parts
            or recipe_path.parts[0] != "examples"
            or ".." in recipe_path.parts
            or recipe_path.suffix != ".yaml"
        ):
            raise ValueError(f"Model release {index} has invalid recipe path {recipe!r}")
        if not (repo_root / recipe_path).is_file():
            raise ValueError(f"Recipe for {model} does not exist in this checkout: {recipe}")

        brev_status_value = raw_entry.get("brev_status")
        if brev_status_value not in {"available", "planned", "unavailable"}:
            raise ValueError(f"Model release {index} has invalid Brev status {brev_status_value!r}")
        brev_status = cast(_BrevStatus, brev_status_value)
        brev_url_value = raw_entry.get("brev_url")
        if brev_url_value is not None and not isinstance(brev_url_value, str):
            raise ValueError(f"Model release {index} Brev URL must be a string or absent")
        brev_url = brev_url_value
        if brev_status == "available":
            parsed_brev_url = urlparse(brev_url or "")
            if (
                parsed_brev_url.scheme != "https"
                or parsed_brev_url.netloc != "brev.nvidia.com"
                or MARKDOWN_UNSAFE_PATTERN.search(brev_url or "")
                or any(character in (brev_url or "") for character in {'"', "<", ">"})
            ):
                raise ValueError(f"Model release {index} requires an https://brev.nvidia.com URL")
        elif brev_url is not None:
            raise ValueError(f"Model release {index} has a Brev URL but status is {brev_status!r}")

        releases.append(
            _ModelRelease(
                release_date=release_date,
                model=model,
                hf_model_id=hf_model_id,
                architectures=architectures,
                docs_page=docs_page,
                model_type=model_type,
                recipe=recipe,
                brev_status=brev_status,
                brev_url=brev_url,
            )
        )

    model_creation_dates = _load_new_model_creation_dates(repo_root, {release.hf_model_id for release in releases})
    for release in releases:
        creation_date = model_creation_dates.get(release.hf_model_id)
        if creation_date is not None and release.release_date != creation_date:
            raise ValueError(
                f"Model {release.hf_model_id!r} uses date {release.release_date}, but its first YAML "
                f"was created on {creation_date}"
            )
    return releases


def _render_recipe_link(release: _ModelRelease) -> str:
    return f"[{Path(release.recipe).name}]({REPOSITORY_URL}/{release.recipe})"


def _render_model_with_recipe(release: _ModelRelease) -> str:
    model_name = release.hf_model_id.split("/", 1)[1]
    model_name = model_name[:1].upper() + model_name[1:]
    return f"[{model_name}]({release.docs_page}) ({_render_recipe_link(release)})"


def _render_release_table(releases: list[_ModelRelease]) -> str:
    table_rows = [
        f"| {release.release_date} | {release.model_type} | {_render_model_with_recipe(release)} |"
        for release in releases
    ]

    return "\n".join(
        [
            "| Date | Type | Model |",
            "|:-----|:-----|:-----|",
            *table_rows,
        ]
    )


def _render_release_tabs(releases: list[_ModelRelease], *, row_limit: int | None = None) -> str:
    preferred_types = (
        "LLM",
        "VLM",
        "Omni",
        "dLLM",
        "Multimodal",
        "Diffusion",
        "Encoder-Decoder",
        "Embedding",
        "Reranking",
    )
    preferred_type_order = {model_type: index for index, model_type in enumerate(preferred_types)}
    model_types = sorted(
        {release.model_type for release in releases},
        key=lambda model_type: (
            preferred_type_order.get(model_type, len(preferred_type_order)),
            model_type.casefold(),
        ),
    )
    tabs = [("All", releases)] + [
        (model_type, [release for release in releases if release.model_type == model_type])
        for model_type in model_types
    ]
    tab_blocks = [
        "\n".join(
            [
                f'<Tab title="{title}">',
                "",
                _render_release_table(tab_releases if row_limit is None else tab_releases[:row_limit]),
                "",
                "</Tab>",
            ]
        )
        for title, tab_releases in tabs
    ]
    return "\n\n".join(
        [
            '<div className="compact-model-tables">',
            COMPACT_TABLE_STYLE,
            "<Tabs>",
            *tab_blocks,
            "</Tabs>",
            "</div>",
        ]
    )


def _render_support_log_table(releases: list[_ModelRelease]) -> str:
    return "\n\n".join([SUPPORT_LOG_START_MARKER, _render_release_tabs(releases), SUPPORT_LOG_END_MARKER])


def _render_homepage_table(releases: list[_ModelRelease]) -> str:
    if len(releases) < TABLE_ROW_COUNT:
        raise ValueError(f"Model release catalog contains only {len(releases)} releases")
    return "\n\n".join(
        [HOMEPAGE_START_MARKER, _render_release_tabs(releases, row_limit=TABLE_ROW_COUNT), HOMEPAGE_END_MARKER]
    )


def _parse_registry_entries(source: str) -> list[tuple[str, str, str]]:
    tree = ast.parse(source)
    mapping_value: ast.AST | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "MODEL_ARCH_MAPPING" for target in node.targets):
            mapping_value = node.value
            break

    if not isinstance(mapping_value, ast.Call) or not isinstance(mapping_value.func, ast.Name):
        raise ValueError("MODEL_ARCH_MAPPING must be an OrderedDict call")
    if mapping_value.func.id != "OrderedDict" or len(mapping_value.args) != 1:
        raise ValueError("MODEL_ARCH_MAPPING must contain one OrderedDict sequence")

    raw_entries = ast.literal_eval(mapping_value.args[0])
    entries = []
    for architecture, specification in raw_entries:
        if not isinstance(architecture, str) or not isinstance(specification, tuple) or len(specification) < 2:
            raise ValueError("Every MODEL_ARCH_MAPPING entry must contain an architecture and module/class tuple")
        module_path, class_name = specification[:2]
        if not isinstance(module_path, str) or not isinstance(class_name, str):
            raise ValueError(f"Invalid registry target for {architecture}")
        entries.append((architecture, module_path, class_name))
    return sorted(entries, key=lambda entry: entry[0].casefold())


def _parse_doc_arch_aliases(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_DOC_ARCH_ALIASES" for target in node.targets):
            continue
        aliases = ast.literal_eval(node.value)
        if not isinstance(aliases, dict) or any(
            not isinstance(architecture, str) or not isinstance(documented_name, str)
            for architecture, documented_name in aliases.items()
        ):
            raise ValueError("_DOC_ARCH_ALIASES must be a string-to-string dictionary")
        return aliases
    raise ValueError("Could not find _DOC_ARCH_ALIASES")


def _render_architecture_name(architecture: str, aliases: dict[str, str]) -> str:
    documented_name = aliases.get(architecture, architecture)
    if documented_name == architecture:
        return f"`{architecture}`"
    return f"`{documented_name}` (`{architecture}`)"


def _render_registry_table(
    entries: list[tuple[str, str, str]], recipe_architectures: set[str], aliases: dict[str, str]
) -> str:
    native_architectures = {architecture for architecture, _, _ in entries}
    native_rows = [
        f"| {_render_architecture_name(architecture, aliases)} | NeMo native | `{module_path}.{class_name}` |"
        for architecture, module_path, class_name in entries
        if architecture in recipe_architectures
    ]
    hf_rows = [
        f"| {_render_architecture_name(architecture, aliases)} | Hugging Face | `transformers` |"
        for architecture in sorted(recipe_architectures - native_architectures, key=str.casefold)
    ]
    return "\n".join(
        [
            REGISTRY_START_MARKER,
            "| Architecture | Source | Implementation |",
            "|---|---|---|",
            *native_rows,
            *hf_rows,
            REGISTRY_END_MARKER,
        ]
    )


def _generate_tables(repo_root: Path) -> dict[Path, str]:
    release_catalog_path = repo_root / "docs" / "model-coverage" / "model-releases.json"
    support_log_path = repo_root / "docs" / "model-coverage" / "latest-models.mdx"
    homepage_path = repo_root / "docs" / "index.mdx"
    overview_path = repo_root / "docs" / "model-coverage" / "overview.mdx"
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"
    aliases_path = repo_root / "tests" / "unit_tests" / "_transformers" / "test_doc_coverage.py"

    support_log = support_log_path.read_text(encoding="utf-8")
    homepage = homepage_path.read_text(encoding="utf-8")
    overview = overview_path.read_text(encoding="utf-8")
    registry_source = registry_path.read_text(encoding="utf-8")
    aliases_source = aliases_path.read_text(encoding="utf-8")

    releases = _load_model_releases(release_catalog_path, repo_root)
    generated_support_log = _render_support_log_table(releases)
    generated_homepage = _render_homepage_table(releases)
    recipe_architectures = {architecture for release in releases for architecture in release.architectures}
    generated_registry = _render_registry_table(
        _parse_registry_entries(registry_source),
        recipe_architectures,
        _parse_doc_arch_aliases(aliases_source),
    )
    return {
        support_log_path: _replace_generated_block(
            support_log, SUPPORT_LOG_START_MARKER, SUPPORT_LOG_END_MARKER, generated_support_log
        ),
        homepage_path: _replace_generated_block(
            homepage, HOMEPAGE_START_MARKER, HOMEPAGE_END_MARKER, generated_homepage
        ),
        overview_path: _replace_generated_block(
            overview, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated_registry
        ),
    }


def _sync_tables(repo_root: Path, *, check: bool) -> list[Path]:
    generated_documents = _generate_tables(repo_root)
    changed_paths = [
        path for path, generated in generated_documents.items() if path.read_text(encoding="utf-8") != generated
    ]
    if check and changed_paths:
        changed = ", ".join(str(path.relative_to(repo_root)) for path in changed_paths)
        raise ValueError(f"Generated model-coverage tables are stale: {changed}")
    if not check:
        for path in changed_paths:
            path.write_text(generated_documents[path], encoding="utf-8")
    return changed_paths


def main() -> None:
    """Generate model-coverage tables in the repository checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of writing when generated output is stale")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    _sync_tables(repo_root, check=args.check)


if __name__ == "__main__":
    main()
