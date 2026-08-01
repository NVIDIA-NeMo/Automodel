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
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

HOMEPAGE_ROW_COUNT = 9
HOMEPAGE_START_MARKER = "{/* BEGIN GENERATED LATEST MODEL SUPPORT */}"
HOMEPAGE_END_MARKER = "{/* END GENERATED LATEST MODEL SUPPORT */}"
REGISTRY_START_MARKER = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"
REGISTRY_END_MARKER = "{/* END GENERATED MODEL ARCHITECTURES */}"
RECIPE_LINK_PATTERN = re.compile(r"^\[[^]]+\]\((https://github\.com/NVIDIA-NeMo/Automodel/(?:blob|tree)/main/[^)]+)\)$")
HF_LINK_PATTERN = re.compile(r"\[[^]]+\]\((https://huggingface\.co/[^)]+)\)")


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


def _parse_release_rows(markdown: str) -> list[list[str]]:
    rows = [
        [cell.strip() for cell in line[1:-1].split("|")]
        for line in markdown.splitlines()
        if re.match(r"^\| \d{4}-\d{2}-\d{2} \|", line)
    ]
    if not rows:
        raise ValueError("No model release rows found")

    for index, row in enumerate(rows):
        if len(row) < 5:
            raise ValueError(f"Release row {index + 1} has {len(row)} columns; expected at least 5")
        if index > 0 and rows[index - 1][0] < row[0]:
            raise ValueError(f"Release log is not reverse chronological at {rows[index - 1][0]} then {row[0]}")
    return rows


def _render_homepage_table(rows: list[list[str]], repo_root: Path) -> str:
    runnable_rows = [row for row in rows if RECIPE_LINK_PATTERN.fullmatch(row[4])]
    if len(runnable_rows) < HOMEPAGE_ROW_COUNT:
        raise ValueError(f"Release log contains only {len(runnable_rows)} runnable recipes")

    table_rows = []
    for date, model, hf_model, modality, recipe, *_ in runnable_rows[:HOMEPAGE_ROW_COUNT]:
        hf_match = HF_LINK_PATTERN.search(hf_model)
        recipe_match = RECIPE_LINK_PATTERN.fullmatch(recipe)
        if hf_match is None:
            raise ValueError(f"HF Model ID for {model} is not a Markdown link: {hf_model}")
        if recipe_match is None:
            raise ValueError(f"Recipe for {model} is not a repository Markdown link: {recipe}")

        recipe_path = unquote(urlparse(recipe_match.group(1)).path).split("/main/", maxsplit=1)[-1]
        if not recipe_path or not (repo_root / recipe_path).is_file():
            raise ValueError(f"Recipe for {model} does not exist in this checkout: {recipe_path}")
        table_rows.append(f"| {date} | {modality} | [{model}]({hf_match.group(1)}) ({recipe}) |")

    return "\n".join(
        [
            HOMEPAGE_START_MARKER,
            "| Date | Modality | Model |",
            "|------|----------|-------|",
            *table_rows,
            HOMEPAGE_END_MARKER,
        ]
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


def _render_registry_table(entries: list[tuple[str, str, str]]) -> str:
    table_rows = [
        f"| `{architecture}` | `{module_path}.{class_name}` |" for architecture, module_path, class_name in entries
    ]
    return "\n".join(
        [
            REGISTRY_START_MARKER,
            "| Checkpoint Architecture | NeMo Implementation |",
            "|---|---|",
            *table_rows,
            REGISTRY_END_MARKER,
        ]
    )


def _generate_tables(repo_root: Path) -> dict[Path, str]:
    release_log_path = repo_root / "docs" / "model-coverage" / "latest-models.mdx"
    homepage_path = repo_root / "docs" / "index.mdx"
    overview_path = repo_root / "docs" / "model-coverage" / "overview.mdx"
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"

    release_log = release_log_path.read_text(encoding="utf-8")
    homepage = homepage_path.read_text(encoding="utf-8")
    overview = overview_path.read_text(encoding="utf-8")
    registry_source = registry_path.read_text(encoding="utf-8")

    generated_homepage = _render_homepage_table(_parse_release_rows(release_log), repo_root)
    generated_registry = _render_registry_table(_parse_registry_entries(registry_source))
    return {
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
    repo_root = Path(__file__).resolve().parents[1]
    _sync_tables(repo_root, check=args.check)


if __name__ == "__main__":
    main()
