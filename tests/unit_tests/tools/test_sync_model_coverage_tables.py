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

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.ci_tests.utils.sync_model_coverage_tables import (
    DATED_MODEL_TABLE_HEADER,
    DATED_SUPPORT_TABLE_HEADER,
    HOMEPAGE_END_MARKER,
    HOMEPAGE_START_MARKER,
    MODEL_TYPE_OVERVIEW_PATHS,
    REGISTRY_END_MARKER,
    REGISTRY_START_MARKER,
    SUPPORT_LOG_END_MARKER,
    SUPPORT_LOG_START_MARKER,
    TABLE_ROW_COUNT,
    _generate_tables,
    _load_model_doc_catalog,
    _load_model_docs,
    _load_model_releases,
    _parse_doc_arch_aliases,
    _parse_registry_entries,
    _render_registry_table,
    _replace_generated_block,
    _strip_generated_tables,
    _sync_tables,
    _validate_dated_support_tables_are_generated,
    _validate_generated_tables_are_not_committed,
)


def _commit_recipes(repo_root: Path, timestamp: str = "2026-07-30T12:00:00Z") -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo_root)], check=True)
    subprocess.run(["git", "-C", str(repo_root), "add", "examples"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "add model recipes",
            f"--date={timestamp}",
        ],
        check=True,
        env={**os.environ, "GIT_COMMITTER_DATE": timestamp},
    )


def _write_typed_overview_templates(repo_root: Path) -> list[Path]:
    paths = []
    for model_type, relative_path in MODEL_TYPE_OVERVIEW_PATHS:
        path = repo_root / "docs" / "model-coverage" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"before\n{SUPPORT_LOG_START_MARKER}\nstale\n{SUPPORT_LOG_END_MARKER}\nafter\n",
            encoding="utf-8",
        )
        paths.append(path)
    diffusion_model_path = repo_root / "docs" / "model-coverage" / "diffusion" / "test" / "model.mdx"
    diffusion_model_path.parent.mkdir(parents=True)
    diffusion_model_path.write_text(
        """---
title: "Test Diffusion Model"
slug: model-coverage/diffusion/test/model
---

<Info>

| | |
|---|---|
| **Task** | Text-to-Image |
| **Architecture** | DiT (Flow Matching) |
| **HF Org** | [Test-Owner](https://huggingface.co/Test-Owner) |

</Info>
""",
        encoding="utf-8",
    )
    llm_model_path = repo_root / "docs" / "model-coverage" / "llm" / "test" / "model.mdx"
    llm_model_path.parent.mkdir(parents=True, exist_ok=True)
    llm_model_path.write_text(
        """---
title: "Test LLM"
slug: model-coverage/large-language-models/test/model
---

| | |
|---|---|
| **Architecture** | `NewModel` |
| **HF Org** | [org](https://huggingface.co/org) |

[`org/model-0`](https://huggingface.co/org/model-0)
""",
        encoding="utf-8",
    )
    return paths


def _fern_slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _frontmatter_slug(path: Path) -> str | None:
    document = path.read_text(encoding="utf-8")
    if not document.startswith("---\n"):
        return None
    closing = document.find("\n---\n", 4)
    if closing == -1:
        return None
    frontmatter = yaml.safe_load(document[4:closing]) or {}
    slug = frontmatter.get("slug")
    return str(slug).strip("/") if slug else None


def _collect_fern_routes(
    items: list[dict[str, object]],
    parents: tuple[str, ...] = (),
    config_dir: Path | None = None,
) -> set[str]:
    routes = set()
    for item in items:
        if "section" in item:
            slug = str(item.get("slug", _fern_slug(str(item["section"]))))
            routes.update(_collect_fern_routes(item.get("contents", []), (*parents, slug), config_dir))
        elif "page" in item:
            page_path = item.get("path")
            frontmatter_slug = None
            if config_dir is not None and isinstance(page_path, str):
                frontmatter_slug = _frontmatter_slug((config_dir / page_path).resolve())
            if frontmatter_slug:
                routes.add("/" + frontmatter_slug)
            else:
                slug = str(item.get("slug", _fern_slug(str(item["page"]))))
                routes.add("/" + "/".join((*parents, slug)))
    return routes


def test_registry_table_is_generated_from_mapping_entries():
    source = """
from collections import OrderedDict

MODEL_ARCH_MAPPING = OrderedDict(
    [
        ("ZuluForCausalLM", ("nemo_automodel.components.models.zulu.model", "ZuluForCausalLM")),
        (
            "AlphaModel",
            ("nemo_automodel.components.models.alpha.model", "AlphaModel", {"retrieval"}),
        ),
    ]
)
"""

    entries = _parse_registry_entries(source)
    generated = _render_registry_table(entries, {"AlphaModel", "ExternalModel", "ZuluForCausalLM"}, {})

    assert entries == [
        ("AlphaModel", "nemo_automodel.components.models.alpha.model", "AlphaModel"),
        ("ZuluForCausalLM", "nemo_automodel.components.models.zulu.model", "ZuluForCausalLM"),
    ]
    assert "| `AlphaModel` | NeMo native | `nemo_automodel.components.models.alpha.model.AlphaModel` |" in generated
    assert (
        "| `ZuluForCausalLM` | NeMo native | `nemo_automodel.components.models.zulu.model.ZuluForCausalLM` |"
        in generated
    )
    assert "| `ExternalModel` | Hugging Face | `transformers` |" in generated
    assert generated.count("`ZuluForCausalLM`") == 1


def test_registry_table_uses_documentation_aliases_for_native_models():
    generated = _render_registry_table(
        [
            (
                "NativeArchitecture",
                "nemo_automodel.components.models.native.model",
                "NativeArchitecture",
            )
        ],
        {"NativeArchitecture"},
        {"NativeArchitecture": "DocumentedArchitecture"},
    )

    assert (
        "| `DocumentedArchitecture` (`NativeArchitecture`) | NeMo native | "
        "`nemo_automodel.components.models.native.model.NativeArchitecture` |"
    ) in generated


def test_doc_arch_aliases_are_parsed_from_the_coverage_test():
    source = '_DOC_ARCH_ALIASES = {"NativeArchitecture": "DocumentedArchitecture"}\n'

    assert _parse_doc_arch_aliases(source) == {"NativeArchitecture": "DocumentedArchitecture"}


def test_generated_registry_table_replaces_the_marked_block():
    document = f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n"
    generated = _render_registry_table([("NewModel", "models.new", "NewModel")], {"NewModel"}, {})

    updated = _replace_generated_block(document, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated)

    assert "stale" not in updated
    assert "| `NewModel` | NeMo native | `models.new.NewModel` |" in updated
    assert updated.startswith("before\n")
    assert updated.endswith("\nafter\n")


def test_model_release_docs_pages_exist_in_nightly_navigation():
    repo_root = Path(__file__).parents[3]
    model_docs, _ = _load_model_docs(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    internal_pages = {release.docs_page for release in releases if release.docs_page.startswith("/")}

    missing_pages = sorted(internal_pages - _collect_fern_routes(navigation, config_dir=config_path.parent))

    assert not missing_pages, f"Model release docs pages missing from nightly navigation: {missing_pages}"


def test_embedding_and_reranking_releases_are_discovered_from_recipes():
    repo_root = Path(__file__).parents[3]
    model_docs, _ = _load_model_docs(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    models_by_type = {
        model_type: {release.hf_model_id for release in releases if release.model_type == model_type}
        for model_type in ("Embedding", "Reranking")
    }

    assert "meta-llama/Llama-3.2-1B" in models_by_type["Embedding"]
    assert "mistralai/Ministral-3-3B-Instruct-2512-BF16" in models_by_type["Embedding"]
    assert models_by_type["Reranking"] == {"meta-llama/Llama-3.2-1B"}


def test_generated_model_coverage_tables_are_not_committed():
    _validate_generated_tables_are_not_committed(Path(__file__).parents[3])


def test_model_type_overviews_use_one_dated_supported_models_table():
    repo_root = Path(__file__).parents[3]
    for _, relative_path in MODEL_TYPE_OVERVIEW_PATHS:
        document = (repo_root / "docs" / "model-coverage" / relative_path).read_text(encoding="utf-8")
        assert document.count("## Supported Models") == 1
        assert "## Model Support Log" not in document
        assert document.count(SUPPORT_LOG_START_MARKER) == 1
        assert document.count(SUPPORT_LOG_END_MARKER) == 1


def test_strip_generated_tables_preserves_empty_markers(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    generated_path = docs_dir / "generated.mdx"
    generated_path.write_text(
        f"before\n{SUPPORT_LOG_START_MARKER}\n| generated |\n{SUPPORT_LOG_END_MARKER}\nafter\n",
        encoding="utf-8",
    )

    assert _strip_generated_tables(tmp_path) == [generated_path]
    assert generated_path.read_text(encoding="utf-8") == (
        f"before\n{SUPPORT_LOG_START_MARKER}\n{SUPPORT_LOG_END_MARKER}\nafter\n"
    )
    _validate_generated_tables_are_not_committed(tmp_path)


def test_committed_generated_tables_are_rejected(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "generated.mdx").write_text(
        f"{SUPPORT_LOG_START_MARKER}\n| generated |\n{SUPPORT_LOG_END_MARKER}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Generated model-coverage tables must not be committed"):
        _validate_generated_tables_are_not_committed(tmp_path)


def test_model_coverage_pages_use_org_slugs_without_nesting_sidebar():
    repo_root = Path(__file__).parents[3]
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    offenders: list[str] = []

    def visit(items: list[dict[str, object]], parent_slugs: tuple[str, ...] = ()) -> None:
        for item in items:
            if "section" in item:
                section_slug = str(item.get("slug", _fern_slug(str(item["section"]))))
                visit(item.get("contents", []), (*parent_slugs, section_slug))
                continue

            path = item.get("path")
            if not isinstance(path, str):
                continue
            parts = Path(path).parts
            if "model-coverage" not in parts:
                continue
            relative_parts = parts[parts.index("model-coverage") + 1 :]
            if len(relative_parts) != 3:
                continue

            category, expected_org = relative_parts[:2]
            frontmatter_slug = _frontmatter_slug((config_path.parent / path).resolve())
            actual_parent = parent_slugs[-1] if parent_slugs else "<none>"
            if frontmatter_slug is None or f"/{expected_org}/" not in f"/{frontmatter_slug}/":
                offenders.append(f"{path}: frontmatter slug does not include organization {expected_org!r}")
            if actual_parent == expected_org:
                offenders.append(f"{path}: organization {expected_org!r} must not be a sidebar section")
            if category == "llm" and (
                frontmatter_slug is None or not frontmatter_slug.startswith("model-coverage/large-language-models/")
            ):
                offenders.append(f"{path}: unexpected LLM slug {frontmatter_slug!r}")

    visit(navigation)

    assert not offenders, "Model coverage organization URL/sidebar violations:\n" + "\n".join(
        f"  - {offender}" for offender in offenders
    )


def test_internal_model_coverage_links_resolve_to_nightly_routes():
    repo_root = Path(__file__).parents[3]
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    broken_links: list[tuple[Path, str]] = []

    for page in (repo_root / "docs").rglob("*.mdx"):
        if "fern/versions" in page.relative_to(repo_root).as_posix():
            continue
        document = page.read_text(encoding="utf-8")
        for link in re.findall(r"\]\((/model-coverage/[^)#?]+)", document):
            # Fern serves a generated llms.txt index at every navigation level;
            # it is a virtual endpoint rather than an entry in nightly.yml.
            if link == "/model-coverage/llms.txt":
                continue
            if link.rstrip("/") not in routes:
                broken_links.append((page.relative_to(repo_root), link))

    assert not broken_links, "Model coverage links missing from nightly routes:\n" + "\n".join(
        f"  - {page}: {link}" for page, link in broken_links
    )


def test_sync_tables_writes_support_log_homepage_and_registry(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "tests" / "unit_tests" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    for index in range(10):
        recipe_path = tmp_path / "examples" / "llm_finetune" / f"model_{index}.yaml"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            f"model:\n  pretrained_model_name_or_path: org/model-{index}\n",
            encoding="utf-8",
        )
    vlm_recipe = tmp_path / "examples" / "vlm_finetune" / "model.yaml"
    vlm_recipe.parent.mkdir(parents=True)
    vlm_recipe.write_text("model:\n  pretrained_model_name_or_path: org/vlm-model\n", encoding="utf-8")
    diffusion_recipe = tmp_path / "examples" / "diffusion" / "finetune" / "model.yaml"
    diffusion_recipe.parent.mkdir(parents=True)
    diffusion_recipe.write_text(
        "model:\n  pretrained_model_name_or_path: Test-Owner/test-diffusion-model\n",
        encoding="utf-8",
    )
    _commit_recipes(tmp_path)
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"before\n{SUPPORT_LOG_START_MARKER}\nstale\n{SUPPORT_LOG_END_MARKER}\nafter\n", encoding="utf-8"
    )
    typed_overview_paths = _write_typed_overview_templates(tmp_path)
    (tmp_path / "docs" / "index.mdx").write_text(
        f"before\n{HOMEPAGE_START_MARKER}\nstale\n{HOMEPAGE_END_MARKER}\nafter\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "model-coverage" / "overview.mdx").write_text(
        f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n", encoding="utf-8"
    )
    (tmp_path / "nemo_automodel" / "_transformers" / "registry.py").write_text(
        'MODEL_ARCH_MAPPING = OrderedDict([("NewModel", ("models.new", "NewModel"))])\n', encoding="utf-8"
    )
    (tmp_path / "tests" / "unit_tests" / "_transformers" / "test_doc_coverage.py").write_text(
        "_DOC_ARCH_ALIASES = {}\n", encoding="utf-8"
    )

    model_docs, _ = _load_model_docs(tmp_path / "docs")
    releases = _load_model_releases(tmp_path, model_docs)

    changed_paths = _sync_tables(tmp_path, check=False)

    assert changed_paths == [
        tmp_path / "docs" / "model-coverage" / "latest-models.mdx",
        *typed_overview_paths,
        tmp_path / "docs" / "index.mdx",
        tmp_path / "docs" / "model-coverage" / "overview.mdx",
    ]
    support_log = (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").read_text(encoding="utf-8")
    assert (
        "| 2026-07-30 | VLM | "
        "[Vlm-model](https://huggingface.co/org/vlm-model) "
        "([model.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/model.yaml)) |"
    ) in support_log
    for (model_type, _), typed_overview_path in zip(MODEL_TYPE_OVERVIEW_PATHS, typed_overview_paths):
        typed_overview = typed_overview_path.read_text(encoding="utf-8")
        typed_rows = [line for line in typed_overview.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]
        assert len(typed_rows) == len([release for release in releases if release.model_type == model_type])
        assert DATED_MODEL_TABLE_HEADER in typed_overview
        assert all(f"| {model_type} |" not in row for row in typed_rows)
        assert "nth-child(-n + 1)" in typed_overview
        assert "nth-child(-n + 2)" not in typed_overview
        assert "<Tabs>" not in typed_overview
    homepage = (tmp_path / "docs" / "index.mdx").read_text(encoding="utf-8")
    for document in (support_log, homepage):
        assert document.count('<div className="compact-model-tables">') == 1
        assert document.count(".compact-model-tables .fern-table-root") == 1
        assert "width: 100% !important;" in document
        assert ".compact-model-tables .fern-table td:last-child" in document
        assert document.count("|:-----|:-----|:-----|") == 1
        assert "<Tabs>" not in document
        assert "<Tab " not in document
        assert "Documentation only" not in document
    assert len([line for line in support_log.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]) == len(
        releases
    )
    assert (
        len([line for line in homepage.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]) == TABLE_ROW_COUNT
    )
    assert "[Model-0](/model-coverage/large-language-models/test/model)" in homepage
    assert "| `NewModel` | NeMo native | `models.new.NewModel` |" in (
        tmp_path / "docs" / "model-coverage" / "overview.mdx"
    ).read_text(encoding="utf-8")
    assert _sync_tables(tmp_path, check=True) == []


def test_model_release_uses_first_recipe_addition_date(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    recipe_dir = tmp_path / "examples" / "llm_finetune"
    recipe_dir.mkdir(parents=True)
    first_recipe_path = "examples/llm_finetune/first.yaml"
    second_recipe_path = "examples/llm_finetune/second.yaml"
    recipe_body = "model:\n  pretrained_model_name_or_path: org/model\n"
    (tmp_path / first_recipe_path).write_text(recipe_body, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", first_recipe_path], check=True)

    def commit(message: str, timestamp: str) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test User",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-am",
                message,
                f"--date={timestamp}",
            ],
            check=True,
            env={**os.environ, "GIT_COMMITTER_DATE": timestamp},
        )

    commit("add first model recipe", "2026-07-29T12:00:00Z")
    (tmp_path / second_recipe_path).write_text(recipe_body, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", second_recipe_path], check=True)
    commit("add second model recipe", "2026-07-30T12:00:00Z")

    releases = _load_model_releases(tmp_path, {})

    assert len(releases) == 1
    assert releases[0].release_date == "2026-07-29"
    assert releases[0].recipe == first_recipe_path


def test_model_release_uses_model_introduction_date_when_recipe_changes(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    recipe_path = tmp_path / "examples" / "llm_finetune" / "model.yaml"
    recipe_path.parent.mkdir(parents=True)

    def commit(model_id: str, message: str, timestamp: str) -> None:
        recipe_path.write_text(f"model:\n  pretrained_model_name_or_path: {model_id}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", str(recipe_path.relative_to(tmp_path))], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test User",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                message,
                f"--date={timestamp}",
            ],
            check=True,
            env={**os.environ, "GIT_COMMITTER_DATE": timestamp},
        )

    commit("org/old-model", "add recipe", "2026-07-29T12:00:00Z")
    commit("org/current-model", "update checkpoint", "2026-08-07T12:00:00Z")

    releases = _load_model_releases(tmp_path, {})

    assert len(releases) == 1
    assert releases[0].hf_model_id == "org/current-model"
    assert releases[0].release_date == "2026-08-07"


def test_typed_support_tables_include_every_documented_model_family():
    repo_root = Path(__file__).parents[3]
    _, _, documented_models = _load_model_doc_catalog(repo_root / "docs")
    generated = _generate_tables(repo_root)
    overview_paths = {
        model_type: repo_root / "docs" / "model-coverage" / relative_path
        for model_type, relative_path in MODEL_TYPE_OVERVIEW_PATHS
    }
    overview_paths["Encoder-Decoder"] = overview_paths["LLM"]

    missing = [
        (model.model_type, model.docs_page)
        for model in documented_models
        if model.docs_page not in generated[overview_paths[model.model_type]]
    ]

    assert not missing, f"Documented model families missing from generated support tables: {missing}"


def test_model_release_ignores_tokenizer_and_teacher_checkpoints(tmp_path):
    recipe_path = tmp_path / "examples" / "llm_kd" / "model.yaml"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(
        """model:
  pretrained_model_name_or_path: org/student
teacher_model:
  pretrained_model_name_or_path: org/teacher
tokenizer:
  pretrained_model_name_or_path: org/tokenizer
""",
        encoding="utf-8",
    )
    _commit_recipes(tmp_path)

    releases = _load_model_releases(tmp_path, {})

    assert [release.hf_model_id for release in releases] == ["org/student"]


def test_model_release_rejects_recipe_additions_on_shallow_boundary(tmp_path):
    recipe_path = tmp_path / "examples" / "llm_finetune" / "model.yaml"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text("model:\n  pretrained_model_name_or_path: org/model\n", encoding="utf-8")
    _commit_recipes(tmp_path)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / ".git" / "shallow").write_text(f"{head}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Recipe additions fall on a shallow boundary"):
        _load_model_releases(tmp_path, {})


def test_model_release_discovery_allows_same_hf_model_for_different_types(tmp_path):
    recipe_paths = (
        "examples/llm_finetune/model.yaml",
        "examples/vlm_finetune/model.yaml",
        "examples/retrieval/bi_encoder/model.yaml",
        "examples/retrieval/cross_encoder/model.yaml",
    )
    for recipe_path in recipe_paths:
        path = tmp_path / recipe_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model:\n  pretrained_model_name_or_path: org/model\n", encoding="utf-8")
    _commit_recipes(tmp_path)

    releases = _load_model_releases(tmp_path, {})

    assert {release.model_type for release in releases} == {"Embedding", "LLM", "Reranking", "VLM"}


def test_sync_tables_check_rejects_stale_generated_support_log(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "tests" / "unit_tests" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    for index in range(TABLE_ROW_COUNT):
        recipe_path = tmp_path / "examples" / "llm_finetune" / f"model_{index}.yaml"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            f"model:\n  pretrained_model_name_or_path: org/model-{index}\n",
            encoding="utf-8",
        )
    _commit_recipes(tmp_path)
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"{SUPPORT_LOG_START_MARKER}\nstale\n{SUPPORT_LOG_END_MARKER}\n", encoding="utf-8"
    )
    _write_typed_overview_templates(tmp_path)
    (tmp_path / "docs" / "index.mdx").write_text(
        f"{HOMEPAGE_START_MARKER}\nstale\n{HOMEPAGE_END_MARKER}\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "model-coverage" / "overview.mdx").write_text(
        f"{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\n", encoding="utf-8"
    )
    (tmp_path / "nemo_automodel" / "_transformers" / "registry.py").write_text(
        'MODEL_ARCH_MAPPING = OrderedDict([("NewModel", ("models.new", "NewModel"))])\n', encoding="utf-8"
    )
    (tmp_path / "tests" / "unit_tests" / "_transformers" / "test_doc_coverage.py").write_text(
        "_DOC_ARCH_ALIASES = {}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="latest-models.mdx"):
        _sync_tables(tmp_path, check=True)


@pytest.mark.parametrize("table_header", [DATED_SUPPORT_TABLE_HEADER, DATED_MODEL_TABLE_HEADER])
def test_dated_support_tables_must_be_generated(tmp_path, table_header):
    docs_dir = tmp_path / "docs" / "model-coverage"
    docs_dir.mkdir(parents=True)
    generated_path = docs_dir / "generated.mdx"
    generated_path.write_text(
        f"{SUPPORT_LOG_START_MARKER}\n{table_header}\n{SUPPORT_LOG_END_MARKER}\n",
        encoding="utf-8",
    )

    _validate_dated_support_tables_are_generated(tmp_path)

    manual_path = docs_dir / "manual.mdx"
    manual_path.write_text(f"{table_header}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manual.mdx"):
        _validate_dated_support_tables_are_generated(tmp_path)
