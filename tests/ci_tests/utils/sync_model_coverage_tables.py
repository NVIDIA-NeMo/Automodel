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
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

TABLE_ROW_COUNT = 10
HOMEPAGE_START_MARKER = "{/* BEGIN GENERATED LATEST MODEL SUPPORT */}"
HOMEPAGE_END_MARKER = "{/* END GENERATED LATEST MODEL SUPPORT */}"
SUPPORT_LOG_START_MARKER = "{/* BEGIN GENERATED MODEL SUPPORT LOG */}"
SUPPORT_LOG_END_MARKER = "{/* END GENERATED MODEL SUPPORT LOG */}"
DIFFUSION_MODELS_START_MARKER = "{/* BEGIN GENERATED DIFFUSION MODELS */}"
DIFFUSION_MODELS_END_MARKER = "{/* END GENERATED DIFFUSION MODELS */}"
REGISTRY_START_MARKER = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"
REGISTRY_END_MARKER = "{/* END GENERATED MODEL ARCHITECTURES */}"
HF_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
DOCS_PAGE_PATTERN = re.compile(r"^/[a-z0-9][a-z0-9/-]*$")
MARKDOWN_UNSAFE_PATTERN = re.compile(r"[\[\]|<>`\r\n]")
REPOSITORY_URL = "https://github.com/NVIDIA-NeMo/Automodel/blob/main"
DATED_SUPPORT_TABLE_HEADER = "| Date | Type | Model |"
MODEL_TYPE_OVERVIEW_PATHS = (
    ("LLM", "llm/index.mdx"),
    ("VLM", "vlm/index.mdx"),
    ("Multimodal", "multimodal/index.mdx"),
    ("Omni", "omni/index.mdx"),
    ("dLLM", "dllm/index.mdx"),
    ("Diffusion", "diffusion/index.mdx"),
    ("Embedding", "embedding/index.mdx"),
    ("Reranking", "reranker/index.mdx"),
)
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


@dataclass(frozen=True)
class _ModelRelease:
    release_date: str
    hf_model_id: str
    docs_page: str
    model_type: str
    recipe: str


@dataclass(frozen=True)
class _ModelDoc:
    model_type: str
    docs_page: str
    architectures: tuple[str, ...]


@dataclass(frozen=True)
class _DiffusionModel:
    owner: str
    model: str
    docs_page: str
    task: str
    architecture: str


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


def _validate_dated_support_tables_are_generated(repo_root: Path) -> None:
    generated_marker_pairs = (
        (HOMEPAGE_START_MARKER, HOMEPAGE_END_MARKER),
        (SUPPORT_LOG_START_MARKER, SUPPORT_LOG_END_MARKER),
    )
    docs_root = repo_root / "docs"
    for path in docs_root.rglob("*.mdx"):
        if "fern" in path.relative_to(docs_root).parts:
            continue
        document = path.read_text(encoding="utf-8")
        ungenerated = document
        for start_marker, end_marker in generated_marker_pairs:
            start = ungenerated.find(start_marker)
            end = ungenerated.find(end_marker)
            if start != -1 and end != -1 and start < end:
                ungenerated = ungenerated[:start] + ungenerated[end + len(end_marker) :]
        if DATED_SUPPORT_TABLE_HEADER in ungenerated:
            raise ValueError(
                f"Dated model-support tables must be inside Python-generated markers: {path.relative_to(repo_root)}"
            )


def _run_git(repo_root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Could not inspect Git history: {error}") from error
    return result.stdout


def _load_recipe_addition_dates(repo_root: Path) -> dict[str, str]:
    if not (repo_root / ".git").exists():
        raise ValueError("Model support generation requires a Git checkout")

    shallow_commits = set()
    if _run_git(repo_root, ["rev-parse", "--is-shallow-repository"]).strip() == "true":
        shallow_path = Path(_run_git(repo_root, ["rev-parse", "--git-path", "shallow"]).strip())
        if not shallow_path.is_absolute():
            shallow_path = repo_root / shallow_path
        shallow_commits = set(shallow_path.read_text(encoding="utf-8").splitlines())

    history = _run_git(
        repo_root,
        [
            "log",
            "--reverse",
            "--no-renames",
            "--diff-filter=A",
            "--format=@@COMMIT@@%H %as",
            "--name-only",
            "--",
            "examples",
        ],
    )
    additions = {}
    commit = None
    commit_date: str | None = None
    for line in history.splitlines():
        if line.startswith("@@COMMIT@@"):
            commit, commit_date = line.removeprefix("@@COMMIT@@").split(" ", 1)
        elif line.endswith(".yaml") and commit_date is not None:
            additions.setdefault(line, (commit_date, commit))
    current_recipes = {str(path.relative_to(repo_root)) for path in (repo_root / "examples").rglob("*.yaml")}
    missing_recipes = current_recipes - additions.keys()
    if missing_recipes:
        raise ValueError(
            "Git history is incomplete for model support generation; use fetch-depth: 0. "
            f"Missing recipe additions: {', '.join(sorted(missing_recipes))}"
        )
    boundary_recipes = sorted(recipe for recipe in current_recipes if additions[recipe][1] in shallow_commits)
    if boundary_recipes:
        raise ValueError(
            "Git history is incomplete for model support generation; use fetch-depth: 0. "
            f"Recipe additions fall on a shallow boundary: {', '.join(boundary_recipes)}"
        )
    return {recipe: addition[0] for recipe, addition in additions.items()}


def _iter_pretrained_model_ids(value: object) -> set[str]:
    model_ids = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pretrained_model_name_or_path" and isinstance(child, str):
                if HF_MODEL_ID_PATTERN.fullmatch(child) and not child.startswith("checkpoints/"):
                    model_ids.add(child)
            model_ids.update(_iter_pretrained_model_ids(child))
    elif isinstance(value, list):
        for child in value:
            model_ids.update(_iter_pretrained_model_ids(child))
    return model_ids


def _model_type_for_recipe(recipe: Path, hf_model_id: str) -> str | None:
    top_level = recipe.parts[1]
    if top_level.startswith("audio_"):
        return "Omni"
    if top_level.startswith("diffusion"):
        return "Diffusion"
    if top_level.startswith("dllm_"):
        return "dLLM"
    if top_level.startswith("multimodal_"):
        return "Multimodal"
    if top_level == "retrieval":
        if "bi_encoder" in recipe.parts:
            return "Embedding"
        if "cross_encoder" in recipe.parts:
            return "Reranking"
        return None
    if top_level.startswith("vlm_"):
        return "VLM"
    if top_level.startswith("llm_") or top_level in {"convergence", "long_context_validation"}:
        if hf_model_id.startswith("google-t5/"):
            return "Encoder-Decoder"
        return "LLM"
    return None


def _load_model_docs(docs_root: Path) -> tuple[dict[str, list[_ModelDoc]], set[str]]:
    directory_types = {
        "llm": "LLM",
        "vlm": "VLM",
        "multimodal": "Multimodal",
        "omni": "Omni",
        "dllm": "dLLM",
        "diffusion": "Diffusion",
        "embedding": "Embedding",
        "reranker": "Reranking",
    }
    model_docs: dict[str, list[_ModelDoc]] = {}
    architectures = set()
    model_coverage_root = docs_root / "model-coverage"
    for directory, model_type in directory_types.items():
        for path in sorted((model_coverage_root / directory).glob("*/*.mdx")):
            document = path.read_text(encoding="utf-8")
            slug_match = re.search(r'^slug: "?([^"\r\n]+)"?$', document, flags=re.MULTILINE)
            if slug_match is None:
                raise ValueError(f"Model coverage page is missing a slug: {path}")
            docs_page = f"/{slug_match.group(1)}"
            if not DOCS_PAGE_PATTERN.fullmatch(docs_page):
                raise ValueError(f"Model coverage page has an invalid slug: {path}")
            architecture_match = re.search(r"^\| \*\*Architecture\*\* \| ([^|]+) \|$", document, flags=re.MULTILINE)
            page_architectures = (
                tuple(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", architecture_match.group(1)))
                if architecture_match
                else ()
            )
            page_architectures = tuple(
                architecture for architecture in page_architectures if architecture != "trust_remote_code"
            )
            architectures.update(page_architectures)
            page_model_type = (
                "Encoder-Decoder" if directory == "llm" and path.parent.name == "google-t5" else model_type
            )
            model_doc = _ModelDoc(page_model_type, docs_page, page_architectures)
            for hf_model_id in re.findall(r"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", document):
                docs = model_docs.setdefault(hf_model_id, [])
                if model_doc not in docs:
                    docs.append(model_doc)
    return model_docs, architectures


def _load_model_releases(repo_root: Path, model_docs: dict[str, list[_ModelDoc]]) -> list[_ModelRelease]:
    recipe_dates = _load_recipe_addition_dates(repo_root)
    releases_by_model_type: dict[tuple[str, str], _ModelRelease] = {}
    for path in sorted((repo_root / "examples").rglob("*.yaml")):
        recipe = path.relative_to(repo_root)
        recipe_text = path.read_text(encoding="utf-8")
        try:
            recipe_config = yaml.safe_load(recipe_text)
        except yaml.YAMLError as error:
            raise ValueError(f"Could not parse recipe {recipe}: {error}") from error
        model_config = recipe_config.get("model") if isinstance(recipe_config, dict) else None
        for hf_model_id in _iter_pretrained_model_ids(model_config):
            inferred_type = _model_type_for_recipe(recipe, hf_model_id)
            if inferred_type is None:
                continue
            matching_docs = model_docs.get(hf_model_id, [])
            documented_types = {model_doc.model_type for model_doc in matching_docs}
            model_type = inferred_type
            if inferred_type == "VLM" and len(documented_types) == 1:
                documented_type = next(iter(documented_types))
                if documented_type in {"Omni", "Multimodal"}:
                    model_type = documented_type
            docs_page = next(
                (model_doc.docs_page for model_doc in matching_docs if model_doc.model_type == model_type),
                f"https://huggingface.co/{hf_model_id}",
            )
            recipe_string = str(recipe)
            release_date = recipe_dates.get(recipe_string)
            if release_date is None:
                raise ValueError(f"Recipe is missing from Git addition history: {recipe_string}")
            release = _ModelRelease(release_date, hf_model_id, docs_page, model_type, recipe_string)
            key = (hf_model_id, model_type)
            previous = releases_by_model_type.get(key)
            if previous is None or (release.release_date, release.recipe) < (previous.release_date, previous.recipe):
                releases_by_model_type[key] = release

    releases = sorted(
        releases_by_model_type.values(),
        key=lambda release: (release.model_type.casefold(), release.hf_model_id.casefold()),
    )
    releases.sort(key=lambda release: release.release_date, reverse=True)
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


def _render_compact_release_table(releases: list[_ModelRelease]) -> str:
    return "\n\n".join(
        [
            '<div className="compact-model-tables">',
            COMPACT_TABLE_STYLE,
            _render_release_table(releases),
            "</div>",
        ]
    )


def _render_support_log_table(releases: list[_ModelRelease]) -> str:
    return "\n\n".join([SUPPORT_LOG_START_MARKER, _render_compact_release_table(releases), SUPPORT_LOG_END_MARKER])


def _render_homepage_table(releases: list[_ModelRelease]) -> str:
    if len(releases) < TABLE_ROW_COUNT:
        raise ValueError(f"Git history produced only {len(releases)} model releases")
    return "\n\n".join(
        [HOMEPAGE_START_MARKER, _render_compact_release_table(releases[:TABLE_ROW_COUNT]), HOMEPAGE_END_MARKER]
    )


def _require_diffusion_page_field(document: str, pattern: str, path: Path, field: str) -> str:
    match = re.search(pattern, document, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Diffusion model page {path} is missing generated-table field {field!r}")
    value = match.group(1).strip()
    if not value or MARKDOWN_UNSAFE_PATTERN.search(value):
        raise ValueError(f"Diffusion model page {path} has invalid generated-table field {field!r}")
    return value


def _format_hugging_face_org(owner: str) -> str:
    words = owner.replace("-", " ").split()
    return " ".join(word.upper() if word.casefold() == "ai" else word.capitalize() for word in words)


def _render_diffusion_models_table(diffusion_docs_dir: Path) -> str:
    models = []
    for path in sorted(diffusion_docs_dir.glob("*/*.mdx")):
        document = path.read_text(encoding="utf-8")
        title = _require_diffusion_page_field(document, r'^title: "([^"]+)"$', path, "title")
        slug = _require_diffusion_page_field(
            document,
            r"^slug: (model-coverage/diffusion/[a-z0-9][a-z0-9/-]+)$",
            path,
            "slug",
        )
        owner = _require_diffusion_page_field(
            document,
            r"^\| \*\*HF Org\*\* \| \[([^\]]+)\]\(https://huggingface\.co/[^)]+\) \|$",
            path,
            "HF Org",
        )
        task = _require_diffusion_page_field(
            document,
            r"^\| \*\*Tasks?\*\* \| ([^|]+) \|$",
            path,
            "Task",
        )
        architecture = _require_diffusion_page_field(
            document,
            r"^\| \*\*Architecture\*\* \| ([^|]+) \|$",
            path,
            "Architecture",
        )
        models.append(
            _DiffusionModel(
                owner=_format_hugging_face_org(owner),
                model=title,
                docs_page=f"/{slug}",
                task=task,
                architecture=architecture,
            )
        )

    if not models:
        raise ValueError(f"No diffusion model pages found under {diffusion_docs_dir}")

    rows = [
        f"| {model.owner} | [{model.model}]({model.docs_page}) | {model.task} | {model.architecture} |"
        for model in sorted(models, key=lambda model: (model.owner.casefold(), model.model.casefold()))
    ]

    return "\n".join(
        [
            DIFFUSION_MODELS_START_MARKER,
            "| Owner | Model | Task | Architecture |",
            "|---|---|---|---|",
            *rows,
            DIFFUSION_MODELS_END_MARKER,
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
    entries: list[tuple[str, str, str]], documented_architectures: set[str], aliases: dict[str, str]
) -> str:
    native_architectures = {architecture for architecture, _, _ in entries}
    native_rows = [
        f"| {_render_architecture_name(architecture, aliases)} | NeMo native | `{module_path}.{class_name}` |"
        for architecture, module_path, class_name in entries
        if architecture in documented_architectures
    ]
    hf_rows = [
        f"| {_render_architecture_name(architecture, aliases)} | Hugging Face | `transformers` |"
        for architecture in sorted(documented_architectures - native_architectures, key=str.casefold)
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
    _validate_dated_support_tables_are_generated(repo_root)
    docs_root = repo_root / "docs"
    diffusion_docs_dir = repo_root / "docs" / "model-coverage" / "diffusion"
    support_log_path = repo_root / "docs" / "model-coverage" / "latest-models.mdx"
    homepage_path = repo_root / "docs" / "index.mdx"
    overview_path = repo_root / "docs" / "model-coverage" / "overview.mdx"
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"
    aliases_path = repo_root / "tests" / "unit_tests" / "_transformers" / "test_doc_coverage.py"

    support_log = support_log_path.read_text(encoding="utf-8")
    typed_overview_paths = {
        model_type: repo_root / "docs" / "model-coverage" / relative_path
        for model_type, relative_path in MODEL_TYPE_OVERVIEW_PATHS
    }
    typed_overviews = {
        model_type: path.read_text(encoding="utf-8") for model_type, path in typed_overview_paths.items()
    }
    homepage = homepage_path.read_text(encoding="utf-8")
    overview = overview_path.read_text(encoding="utf-8")
    registry_source = registry_path.read_text(encoding="utf-8")
    aliases_source = aliases_path.read_text(encoding="utf-8")

    model_docs, documented_architectures = _load_model_docs(docs_root)
    releases = _load_model_releases(repo_root, model_docs)
    generated_diffusion_models = _render_diffusion_models_table(diffusion_docs_dir)
    typed_overviews["Diffusion"] = _replace_generated_block(
        typed_overviews["Diffusion"],
        DIFFUSION_MODELS_START_MARKER,
        DIFFUSION_MODELS_END_MARKER,
        generated_diffusion_models,
    )
    generated_support_log = _render_support_log_table(releases)
    generated_homepage = _render_homepage_table(releases)
    generated_registry = _render_registry_table(
        _parse_registry_entries(registry_source),
        documented_architectures,
        _parse_doc_arch_aliases(aliases_source),
    )
    generated_documents = {
        support_log_path: _replace_generated_block(
            support_log, SUPPORT_LOG_START_MARKER, SUPPORT_LOG_END_MARKER, generated_support_log
        ),
        **{
            typed_overview_paths[model_type]: _replace_generated_block(
                typed_overviews[model_type],
                SUPPORT_LOG_START_MARKER,
                SUPPORT_LOG_END_MARKER,
                _render_support_log_table([release for release in releases if release.model_type == model_type]),
            )
            for model_type, _ in MODEL_TYPE_OVERVIEW_PATHS
        },
        homepage_path: _replace_generated_block(
            homepage, HOMEPAGE_START_MARKER, HOMEPAGE_END_MARKER, generated_homepage
        ),
        overview_path: _replace_generated_block(
            overview, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated_registry
        ),
    }
    return generated_documents


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
