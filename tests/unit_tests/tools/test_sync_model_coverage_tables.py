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
    RELEASE_LOG_END_MARKER,
    RELEASE_LOG_START_MARKER,
    _load_model_releases,
    _parse_registry_entries,
    _render_registry_table,
    _replace_generated_block,
    _sync_tables,
)


def _fern_slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _collect_fern_routes(items: list[dict[str, object]], parents: tuple[str, ...] = ()) -> set[str]:
    routes = set()
    for item in items:
        if "section" in item:
            slug = str(item.get("slug", _fern_slug(str(item["section"]))))
            routes.update(_collect_fern_routes(item.get("contents", []), (*parents, slug)))
        elif "page" in item:
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
    generated = _render_registry_table(entries)

    assert entries == [
        ("AlphaModel", "nemo_automodel.components.models.alpha.model", "AlphaModel"),
        ("ZuluForCausalLM", "nemo_automodel.components.models.zulu.model", "ZuluForCausalLM"),
    ]
    assert "| `AlphaModel` | `nemo_automodel.components.models.alpha.model.AlphaModel` |" in generated
    assert "| `ZuluForCausalLM` | `nemo_automodel.components.models.zulu.model.ZuluForCausalLM` |" in generated


def test_generated_registry_table_replaces_the_marked_block():
    document = f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n"
    generated = _render_registry_table([("NewModel", "models.new", "NewModel")])

    updated = _replace_generated_block(document, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated)

    assert "stale" not in updated
    assert "| `NewModel` | `models.new.NewModel` |" in updated
    assert updated.startswith("before\n")
    assert updated.endswith("\nafter\n")


def test_model_release_docs_pages_exist_in_nightly_navigation():
    repo_root = Path(__file__).parents[3]
    releases = json.loads((repo_root / "docs" / "model-coverage" / "model-releases.json").read_text(encoding="utf-8"))
    navigation = yaml.safe_load((repo_root / "docs" / "fern" / "versions" / "nightly.yml").read_text(encoding="utf-8"))[
        "navigation"
    ]

    missing_pages = sorted({release["docs_page"] for release in releases} - _collect_fern_routes(navigation))

    assert not missing_pages, f"Model release docs pages missing from nightly navigation: {missing_pages}"


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


def test_sync_tables_writes_release_log_homepage_and_registry(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    shared_recipe = "examples/shared.yaml"
    (tmp_path / shared_recipe).write_text("model: test\n", encoding="utf-8")

    releases = [
        {
            "date": "2026-07-31",
            "model": "Documentation Model",
            "hf_model_id": "org/documentation-model",
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
                "docs_page": f"/model-coverage/large-language-models/model-{index}",
                "type": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"before\n{RELEASE_LOG_START_MARKER}\nstale\n{RELEASE_LOG_END_MARKER}\nafter\n", encoding="utf-8"
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

    changed_paths = _sync_tables(tmp_path, check=False)

    assert changed_paths == [
        tmp_path / "docs" / "model-coverage" / "latest-models.mdx",
        tmp_path / "docs" / "index.mdx",
        tmp_path / "docs" / "model-coverage" / "overview.mdx",
    ]
    release_log = (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").read_text(encoding="utf-8")
    assert (
        "| 2026-07-31 | VLM | "
        "[Documentation-model](/model-coverage/vision-language-models/documentation-model) "
        "([shared.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/shared.yaml)) |"
    ) in release_log
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
    assert re.findall(r'<Tab title="([^"]+)">', release_log) == expected_tabs
    homepage = (tmp_path / "docs" / "index.mdx").read_text(encoding="utf-8")
    assert re.findall(r'<Tab title="([^"]+)">', homepage) == expected_tabs
    assert len(_tab_table_rows(release_log, "All")) == len(releases)
    assert len(_tab_table_rows(release_log, "LLM")) == 10
    for tab in expected_tabs:
        assert len(_tab_table_rows(homepage, tab)) <= 9
    for document in (release_log, homepage):
        assert document.count("<CompactModelTables>") == 1
        assert document.count("</CompactModelTables>") == 1
        assert document.count("|:-----|:-----|:-----|") == len(expected_tabs)
        assert "Documentation only" not in document
    assert len(_tab_table_rows(homepage, "All")) == 9
    assert len(_tab_table_rows(homepage, "LLM")) == 9
    llm_tab = release_log.split('<Tab title="LLM">', 1)[1].split("</Tab>", 1)[0]
    vlm_tab = release_log.split('<Tab title="VLM">', 1)[1].split("</Tab>", 1)[0]
    assert "[Model-0](/model-coverage/large-language-models/model-0)" in llm_tab
    assert "[Documentation-model]" not in llm_tab
    assert "[Documentation-model](/model-coverage/vision-language-models/documentation-model)" in vlm_tab
    assert "[Model-0]" not in vlm_tab
    assert "Documentation Model" not in release_log
    assert "Try on Brev" not in release_log
    assert "brev.nvidia.com" not in release_log
    assert "[Model-0](/model-coverage/large-language-models/model-0)" in homepage
    assert "Documentation Model" not in homepage
    assert "| `NewModel` | `models.new.NewModel` |" in (
        tmp_path / "docs" / "model-coverage" / "overview.mdx"
    ).read_text(encoding="utf-8")
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
            "docs_page": "/model-coverage/large-language-models/older-model",
            "type": "LLM",
            "recipe": recipe_path,
            "brev_status": "unavailable",
        },
        {
            "date": "2026-07-30",
            "model": "Newer Model",
            "hf_model_id": "org/newer-model",
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


def test_sync_tables_check_rejects_stale_generated_release_log(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    releases = []
    for index in range(9):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        releases.append(
            {
                "date": f"2026-07-{30 - index:02d}",
                "model": f"Model {index}",
                "hf_model_id": f"org/model-{index}",
                "docs_page": f"/model-coverage/large-language-models/model-{index}",
                "type": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"{RELEASE_LOG_START_MARKER}\nstale\n{RELEASE_LOG_END_MARKER}\n", encoding="utf-8"
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

    with pytest.raises(ValueError, match="latest-models.mdx"):
        _sync_tables(tmp_path, check=True)
