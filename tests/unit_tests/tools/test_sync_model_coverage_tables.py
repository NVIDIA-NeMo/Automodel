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

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.ci_tests.utils.sync_model_coverage_tables import (
    HOMEPAGE_END_MARKER,
    HOMEPAGE_START_MARKER,
    REGISTRY_END_MARKER,
    REGISTRY_START_MARKER,
    SUPPORT_LOG_END_MARKER,
    SUPPORT_LOG_START_MARKER,
    TABLE_ROW_COUNT,
    _load_model_releases,
    _parse_doc_arch_aliases,
    _parse_registry_entries,
    _render_registry_table,
    _replace_generated_block,
    _sync_tables,
)


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


def _tab_table_rows(document: str, title: str) -> list[str]:
    tab_content = document.split(f'<Tab title="{title}">', 1)[1].split("</Tab>", 1)[0]
    return [line for line in tab_content.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]


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
    releases = json.loads((repo_root / "docs" / "model-coverage" / "model-releases.json").read_text(encoding="utf-8"))
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]

    missing_pages = sorted(
        {release["docs_page"] for release in releases} - _collect_fern_routes(navigation, config_dir=config_path.parent)
    )

    assert not missing_pages, f"Model release docs pages missing from nightly navigation: {missing_pages}"


def test_model_release_architectures_match_cached_hf_configs():
    """Validate checked-in architecture metadata without accessing the Hub."""
    from transformers import PretrainedConfig

    repo_root = Path(__file__).parents[3]
    releases = json.loads((repo_root / "docs" / "model-coverage" / "model-releases.json").read_text())
    expected_by_model: dict[str, set[str]] = {}
    for release in releases:
        expected_by_model.setdefault(release["hf_model_id"], set()).update(release["architectures"])

    checked = 0
    mismatches: list[tuple[str, set[str], set[str]]] = []
    for model_id, expected in sorted(expected_by_model.items()):
        try:
            config_dict, _ = PretrainedConfig.get_config_dict(model_id, local_files_only=True)
        except (OSError, ValueError):
            continue
        checked += 1
        actual = set(config_dict.get("architectures") or [])
        if actual != expected:
            mismatches.append((model_id, expected, actual))

    if not checked:
        pytest.skip("No Hugging Face configs are available in the local offline cache")

    assert not mismatches, "Cached Hugging Face architectures differ from model-releases.json:\n" + "\n".join(
        f"  - {model_id}: catalog={sorted(expected)}, cached={sorted(actual)}"
        for model_id, expected, actual in mismatches
    )


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
            if link.rstrip("/") not in routes:
                broken_links.append((page.relative_to(repo_root), link))

    assert not broken_links, "Model coverage links missing from nightly routes:\n" + "\n".join(
        f"  - {page}: {link}" for page, link in broken_links
    )


def test_embedding_and_reranking_catalog_lists_documented_models():
    repo_root = Path(__file__).parents[3]
    releases = json.loads((repo_root / "docs" / "model-coverage" / "model-releases.json").read_text(encoding="utf-8"))
    models_by_type = {
        model_type: {release["hf_model_id"] for release in releases if release["type"] == model_type}
        for model_type in ("Embedding", "Reranking")
    }

    assert models_by_type == {
        "Embedding": {
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-1B",
            "mistralai/Ministral-3-3B-Base-2512",
            "mistralai/Ministral-3-3B-Instruct-2512",
            "nvidia/llama-embed-nemotron-8b",
            "nvidia/llama-nemotron-embed-1b-v2",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
        },
        "Reranking": {
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-1B",
            "nvidia/llama-nemotron-rerank-1b-v2",
        },
    }


def test_sync_tables_writes_support_log_homepage_and_registry(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "tests" / "unit_tests" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    shared_recipe = "examples/shared.yaml"
    (tmp_path / shared_recipe).write_text("model: test\n", encoding="utf-8")

    releases = [
        {
            "date": "2026-07-31",
            "model": "Documentation Model",
            "hf_model_id": "org/documentation-model",
            "architectures": ["DocumentationForConditionalGeneration", "NewModel"],
            "docs_page": "/model-coverage/vision-language-models/documentation-model",
            "type": "VLM",
            "recipe": shared_recipe,
            "brev_status": "available",
            "brev_url": "https://brev.nvidia.com/launchable/deploy/now?launchableID=test",
        }
    ]
    for model_type in ("Omni", "dLLM", "Multimodal", "Diffusion", "Encoder-Decoder", "Embedding", "Reranking"):
        type_slug = model_type.lower().replace(" ", "-")
        releases.append(
            {
                "date": "2026-07-31",
                "model": f"{model_type} Model",
                "hf_model_id": f"org/{type_slug}-model",
                "architectures": [f"{type_slug.title().replace('-', '')}Model"],
                "docs_page": f"/model-coverage/{type_slug}/model",
                "type": model_type,
                "recipe": shared_recipe,
                "brev_status": "planned",
            }
        )
    for index in range(10):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        releases.append(
            {
                "date": f"2026-07-{30 - index:02d}",
                "model": f"Model {index}",
                "hf_model_id": f"org/model-{index}",
                "architectures": [f"Model{index}ForCausalLM"],
                "docs_page": f"/model-coverage/large-language-models/model-{index}",
                "type": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"before\n{SUPPORT_LOG_START_MARKER}\nstale\n{SUPPORT_LOG_END_MARKER}\nafter\n", encoding="utf-8"
    )
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

    changed_paths = _sync_tables(tmp_path, check=False)

    assert changed_paths == [
        tmp_path / "docs" / "model-coverage" / "latest-models.mdx",
        tmp_path / "docs" / "index.mdx",
        tmp_path / "docs" / "model-coverage" / "overview.mdx",
    ]
    support_log = (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").read_text(encoding="utf-8")
    assert (
        "| 2026-07-31 | VLM | "
        "[Documentation-model](/model-coverage/vision-language-models/documentation-model) "
        "([shared.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/shared.yaml)) |"
    ) in support_log
    expected_tabs = [
        "All",
        "LLM",
        "VLM",
        "Omni",
        "dLLM",
        "Multimodal",
        "Diffusion",
        "Encoder-Decoder",
        "Embedding",
        "Reranking",
    ]
    assert re.findall(r'<Tab title="([^"]+)">', support_log) == expected_tabs
    homepage = (tmp_path / "docs" / "index.mdx").read_text(encoding="utf-8")
    assert re.findall(r'<Tab title="([^"]+)">', homepage) == expected_tabs
    assert len(_tab_table_rows(support_log, "All")) == len(releases)
    assert len(_tab_table_rows(support_log, "LLM")) == 10
    for tab in expected_tabs:
        assert len(_tab_table_rows(homepage, tab)) <= TABLE_ROW_COUNT
    for document in (support_log, homepage):
        assert document.count('<div className="compact-model-tables">') == 1
        assert document.count(".compact-model-tables .fern-table-root") == 1
        assert "width: 100% !important;" in document
        assert ".compact-model-tables .fern-table td:last-child" in document
        assert document.count("|:-----|:-----|:-----|") == len(expected_tabs)
        assert "Documentation only" not in document
    assert len(_tab_table_rows(homepage, "All")) == TABLE_ROW_COUNT
    assert len(_tab_table_rows(homepage, "LLM")) == TABLE_ROW_COUNT
    llm_tab = support_log.split('<Tab title="LLM">', 1)[1].split("</Tab>", 1)[0]
    vlm_tab = support_log.split('<Tab title="VLM">', 1)[1].split("</Tab>", 1)[0]
    assert "[Model-0](/model-coverage/large-language-models/model-0)" in llm_tab
    assert "[Documentation-model]" not in llm_tab
    assert "[Documentation-model](/model-coverage/vision-language-models/documentation-model)" in vlm_tab
    assert "[Model-0]" not in vlm_tab
    assert "Documentation Model" not in support_log
    assert "Try on Brev" not in support_log
    assert "brev.nvidia.com" not in support_log
    assert "[Model-0](/model-coverage/large-language-models/model-0)" in homepage
    assert "Documentation Model" not in homepage
    assert "| `NewModel` | NeMo native | `models.new.NewModel` |" in (
        tmp_path / "docs" / "model-coverage" / "overview.mdx"
    ).read_text(encoding="utf-8")
    overview = (tmp_path / "docs" / "model-coverage" / "overview.mdx").read_text(encoding="utf-8")
    assert "| `DocumentationForConditionalGeneration` | Hugging Face | `transformers` |" in overview
    assert _sync_tables(tmp_path, check=True) == []


def test_model_release_catalog_rejects_non_chronological_entries(tmp_path):
    (tmp_path / "examples").mkdir()
    recipe_path = "examples/model.yaml"
    (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
    catalog = [
        {
            "date": "2026-07-29",
            "model": "Older Model",
            "hf_model_id": "org/older-model",
            "architectures": ["OlderModel"],
            "docs_page": "/model-coverage/large-language-models/older-model",
            "type": "LLM",
            "recipe": recipe_path,
            "brev_status": "unavailable",
        },
        {
            "date": "2026-07-30",
            "model": "Newer Model",
            "hf_model_id": "org/newer-model",
            "architectures": ["NewerModel"],
            "docs_page": "/model-coverage/large-language-models/newer-model",
            "type": "LLM",
            "recipe": recipe_path,
            "brev_status": "planned",
        },
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="not reverse chronological"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_requires_iso_date_format(tmp_path):
    catalog = [
        {
            "date": "20260730",
            "model": "Model",
            "hf_model_id": "org/model",
            "architectures": ["ModelForCausalLM"],
            "docs_page": "/model-coverage/large-language-models/model",
            "type": "LLM",
            "recipe": None,
            "brev_status": "planned",
        }
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_requires_version_agnostic_docs_page(tmp_path):
    catalog = [
        {
            "date": "2026-07-30",
            "model": "Model",
            "hf_model_id": "org/model",
            "architectures": ["ModelForCausalLM"],
            "docs_page": "https://docs.nvidia.com/nemo/automodel/nightly/model",
            "type": "LLM",
            "recipe": None,
            "brev_status": "planned",
        }
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="version-agnostic docs page"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_requires_recipe(tmp_path):
    catalog = [
        {
            "date": "2026-07-30",
            "model": "Model",
            "hf_model_id": "org/model",
            "architectures": ["ModelForCausalLM"],
            "docs_page": "/model-coverage/large-language-models/model",
            "type": "LLM",
            "recipe": None,
            "brev_status": "planned",
        }
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="field 'recipe' must be a non-empty Markdown-safe string"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_uses_first_yaml_for_model_id(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("test repository\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
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
            "initialize repository",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "-b", "feature"], check=True)

    (tmp_path / "examples").mkdir()
    first_recipe_path = "examples/first.yaml"
    selected_recipe_path = "examples/selected.yaml"
    (tmp_path / first_recipe_path).write_text("model: org/model\n", encoding="utf-8")
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
    (tmp_path / selected_recipe_path).write_text("model: org/model\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", selected_recipe_path], check=True)
    commit("add selected model recipe", "2026-07-30T12:00:00Z")

    catalog = [
        {
            "date": "2026-07-30",
            "model": "Model",
            "hf_model_id": "org/model",
            "architectures": ["ModelForCausalLM"],
            "docs_page": "/model-coverage/model",
            "type": "LLM",
            "recipe": selected_recipe_path,
            "brev_status": "planned",
        }
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="first YAML was created on 2026-07-29"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_allows_same_hf_model_for_different_types(tmp_path):
    (tmp_path / "examples").mkdir()
    recipe_path = "examples/model.yaml"
    (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
    catalog = [
        {
            "date": "2026-07-30",
            "model": f"Model {model_type}",
            "hf_model_id": "org/model",
            "architectures": ["ModelForCausalLM"],
            "docs_page": "/model-coverage/model",
            "type": model_type,
            "recipe": recipe_path,
            "brev_status": "planned",
        }
        for model_type in ("LLM", "Embedding", "Reranking")
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    releases = _load_model_releases(catalog_path, tmp_path)

    assert [release.model_type for release in releases] == ["LLM", "Embedding", "Reranking"]


def test_sync_tables_check_rejects_stale_generated_support_log(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "tests" / "unit_tests" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    releases = []
    for index in range(TABLE_ROW_COUNT):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        releases.append(
            {
                "date": f"2026-07-{30 - index:02d}",
                "model": f"Model {index}",
                "hf_model_id": f"org/model-{index}",
                "architectures": [f"Model{index}ForCausalLM"],
                "docs_page": f"/model-coverage/large-language-models/model-{index}",
                "type": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"{SUPPORT_LOG_START_MARKER}\nstale\n{SUPPORT_LOG_END_MARKER}\n", encoding="utf-8"
    )
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
