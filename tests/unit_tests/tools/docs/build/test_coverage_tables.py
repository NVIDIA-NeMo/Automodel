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

import io
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest
import yaml

from tools.docs.build.coverage_tables import (
    _generate_tables,
    _load_model_doc_catalog,
    _load_model_docs,
    _load_model_releases,
    _ModelDoc,
    _ModelRelease,
    _parse_registry_entries,
    _read_documented_recipe,
    _render_registry_table,
    _replace_generated_block,
    _resolve_model_card_routes,
    _strip_generated_tables,
    _sync_tables,
    _validate_dated_support_tables_are_generated,
    _validate_generated_tables_are_not_committed,
)
from tools.docs.catalog import (
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
)

pytestmark = pytest.mark.timeout(70)


def _commit_recipes(repo_root: Path, timestamp: str = "2026-07-30T12:00:00Z") -> None:
    # Disposable recipe-history commits must not run contributor hooks on fixture files.
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
            "-c",
            f"core.hooksPath={os.devnull}",
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


def test_generated_registry_table_replaces_the_marked_block():
    document = f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n"
    generated = _render_registry_table([("NewModel", "models.new", "NewModel")], {"NewModel"}, {})

    updated = _replace_generated_block(document, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated)

    assert "stale" not in updated
    assert "| `NewModel` | NeMo native | `models.new.NewModel` |" in updated
    assert updated.startswith("before\n")
    assert updated.endswith("\nafter\n")


def test_recipe_model_id_can_redirect_to_one_shared_model_card(tmp_path):
    docs_config_path = tmp_path / "docs" / "fern" / "docs.yml"
    docs_config_path.parent.mkdir(parents=True)
    docs_config_path.write_text(
        yaml.safe_dump(
            {
                "redirects": [
                    {
                        "source": "/nemo/automodel/model-coverage/large-language-models/org/model",
                        "destination": "/nemo/automodel/model-coverage/large-language-models/org/model-instruct",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model_doc = _ModelDoc(
        "LLM",
        "/model-coverage/large-language-models/org/model-instruct",
        (),
    )
    release = _ModelRelease("2026-01-01", "org/model", "https://huggingface.co/org/model", "LLM", "recipe.yaml")

    routes = _resolve_model_card_routes(
        tmp_path,
        [release],
        {"org/model-instruct": [model_doc]},
        [model_doc],
    )

    assert routes == {("org/model", "LLM"): model_doc.docs_page}


def test_recipe_model_redirects_are_case_insensitive(tmp_path):
    docs_config_path = tmp_path / "docs" / "fern" / "docs.yml"
    docs_config_path.parent.mkdir(parents=True)
    docs_config_path.write_text(
        yaml.safe_dump(
            {
                "redirects": [
                    {
                        "source": "/nemo/automodel/model-coverage/large-language-models/org/model",
                        "destination": "/nemo/automodel/model-coverage/large-language-models/org/Canonical-Model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model_doc = _ModelDoc("LLM", "/model-coverage/large-language-models/org/Canonical-Model", ())
    release = _ModelRelease("2026-01-01", "org/Model", "https://huggingface.co/org/Model", "LLM", "recipe.yaml")

    routes = _resolve_model_card_routes(tmp_path, [release], {"org/Canonical-Model": [model_doc]}, [model_doc])

    assert routes == {("org/Model", "LLM"): model_doc.docs_page}


def test_case_insensitive_redirect_cannot_shadow_a_canonical_card(tmp_path):
    docs_config_path = tmp_path / "docs" / "fern" / "docs.yml"
    docs_config_path.parent.mkdir(parents=True)
    docs_config_path.write_text(
        yaml.safe_dump(
            {
                "redirects": [
                    {
                        "source": "/nemo/automodel/model-coverage/large-language-models/org/model",
                        "destination": "/nemo/automodel/model-coverage/large-language-models/org",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model_doc = _ModelDoc("LLM", "/model-coverage/large-language-models/org/Model", ())

    with pytest.raises(ValueError, match="collide case-insensitively with canonical model-card routes"):
        _resolve_model_card_routes(tmp_path, [], {"org/Model": [model_doc]}, [model_doc])


def test_recipe_model_id_without_model_card_is_rejected(tmp_path):
    docs_config_path = tmp_path / "docs" / "fern" / "docs.yml"
    docs_config_path.parent.mkdir(parents=True)
    docs_config_path.write_text("redirects: []\n", encoding="utf-8")
    model_doc = _ModelDoc(
        "LLM",
        "/model-coverage/large-language-models/org/other-model",
        (),
    )
    release = _ModelRelease(
        "2026-01-01",
        "org/missing-model",
        "https://huggingface.co/org/missing-model",
        "LLM",
        "recipe.yaml",
    )

    with pytest.raises(ValueError, match=r"missing: LLM: org/missing-model"):
        _resolve_model_card_routes(
            tmp_path,
            [release],
            {"org/other-model": [model_doc]},
            [model_doc],
        )


def test_recipe_model_id_with_multiple_model_cards_is_rejected(tmp_path):
    docs_config_path = tmp_path / "docs" / "fern" / "docs.yml"
    docs_config_path.parent.mkdir(parents=True)
    docs_config_path.write_text("redirects: []\n", encoding="utf-8")
    first_doc = _ModelDoc("LLM", "/model-coverage/large-language-models/first/model", ())
    second_doc = _ModelDoc("LLM", "/model-coverage/large-language-models/second/model", ())
    release = _ModelRelease("2026-01-01", "org/model", "https://huggingface.co/org/model", "LLM", "recipe.yaml")

    with pytest.raises(ValueError, match=r"ambiguous: LLM: org/model"):
        _resolve_model_card_routes(
            tmp_path,
            [release],
            {"org/first": [first_doc], "org/second": [second_doc]},
            [first_doc, second_doc],
        )


def test_embedding_and_reranking_releases_are_discovered_from_recipes(tmp_path):
    repo_root = Path(__file__).parents[5]
    model_docs, _ = _load_model_docs(repo_root / "docs")
    # Keep real recipe discovery while making Git history independent of checkout size.
    shutil.copytree(repo_root / "examples" / "retrieval", tmp_path / "examples" / "retrieval")
    _commit_recipes(tmp_path)
    releases = _load_model_releases(tmp_path, model_docs)
    models_by_type = {
        model_type: {release.hf_model_id for release in releases if release.model_type == model_type}
        for model_type in ("Embedding", "Reranking")
    }

    assert "meta-llama/Llama-3.2-1B" in models_by_type["Embedding"]
    assert "mistralai/Ministral-3-3B-Instruct-2512-BF16" in models_by_type["Embedding"]
    assert models_by_type["Reranking"] == {
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen3-Reranker-4B",
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    }


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


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_trusted_catalog_resolves_cards_in_checkouts_before_and_after_a_rename(tmp_path, canonical_exists):
    alias = "docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx"
    canonical = "docs/model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base.mdx"
    document = (
        '---\ntitle: "flux-3-image-klein-base"\n'
        "slug: model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base\n---\n"
        "[Checkpoint](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)\n"
    )
    alias_path = tmp_path / alias
    alias_path.parent.mkdir(parents=True)
    alias_path.write_text(document, encoding="utf-8")
    if canonical_exists:
        (tmp_path / canonical).write_text(document, encoding="utf-8")
    config_path = tmp_path / "docs/fern/docs.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("redirects: []\n", encoding="utf-8")

    navigation = config_path.parent / "versions/nightly.yml"
    navigation.parent.mkdir()
    selected = canonical if canonical_exists else alias
    navigation.write_text(f"navigation:\n  - page: Model\n    path: ../../{selected.removeprefix('docs/')}\n")

    model_docs, _, catalog = _load_model_doc_catalog(tmp_path / "docs")
    release = _ModelRelease(
        "2026-10-22", "black-forest-labs/flux-3-image-klein-base", "", "Diffusion", "examples/flux3.yaml"
    )
    routes = _resolve_model_card_routes(tmp_path, [release], model_docs, catalog)

    assert routes == {
        (
            "black-forest-labs/flux-3-image-klein-base",
            "Diffusion",
        ): "/model-coverage/diffusion/black-forest-labs/flux-3-image-klein-base"
    }
    assert [model.source_path for model in catalog] == [canonical if canonical_exists else alias]


def test_documented_recipe_reads_current_checkout(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("model: {}\n", encoding="utf-8")
    assert _read_documented_recipe(tmp_path, "recipe.yaml", "") == "model: {}\n"


def test_documented_recipe_rejects_permalink_for_checked_in_recipe(tmp_path):
    revision = "a" * 40
    recipe_path = "examples/model.yaml"
    (tmp_path / "examples").mkdir()
    (tmp_path / recipe_path).write_text("model: {}\n", encoding="utf-8")
    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{revision}/{recipe_path})"
    with pytest.raises(ValueError, match="replace its commit permalink with.*blob/main/examples/model.yaml"):
        _read_documented_recipe(tmp_path, recipe_path, document)


def test_documented_recipe_reads_permalink_when_absent_from_checkout(tmp_path, monkeypatch):
    revision = "a" * 40
    recipe_path = "examples/model.yaml"
    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{revision}/{recipe_path})"

    def open_recipe(url, *, timeout):
        assert url == f"https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/{revision}/{recipe_path}"
        assert timeout == 10
        return io.BytesIO(b"model: {pretrained_model_name_or_path: org/model}\n")

    monkeypatch.setattr(urllib.request, "urlopen", open_recipe)
    assert "org/model" in _read_documented_recipe(tmp_path, recipe_path, document)


@pytest.mark.parametrize("revision,linked_path", [("main", "examples/model.yaml"), ("a" * 40, "examples/other.yaml")])
def test_missing_recipe_requires_matching_commit_permalink(tmp_path, revision, linked_path):
    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{revision}/{linked_path})"
    with pytest.raises(FileNotFoundError):
        _read_documented_recipe(tmp_path, "examples/model.yaml", document)


def test_sync_tables_writes_support_log_homepage_and_registry(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "tools" / "docs").mkdir(parents=True)
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
    (tmp_path / "tools" / "docs" / "catalog.py").write_text("_DOC_ARCH_ALIASES = {}\n", encoding="utf-8")

    model_docs, _ = _load_model_docs(tmp_path / "docs")
    releases = _load_model_releases(tmp_path, model_docs)

    changed_paths = _sync_tables(tmp_path, check=False, validate_model_cards=False)

    assert changed_paths == [
        tmp_path / "docs" / "model-coverage" / "latest-models.mdx",
        *typed_overview_paths,
        tmp_path / "docs" / "index.mdx",
        tmp_path / "docs" / "model-coverage" / "overview.mdx",
    ]
    support_log = (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").read_text(encoding="utf-8")
    assert (
        "| 2026-07-30 | VLM | "
        "[Vlm-model](https://huggingface.co/org/vlm-model) | "
        "[recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/model.yaml) |"
    ) in support_log
    for (model_type, _), typed_overview_path in zip(MODEL_TYPE_OVERVIEW_PATHS, typed_overview_paths):
        typed_overview = typed_overview_path.read_text(encoding="utf-8")
        typed_rows = [line for line in typed_overview.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]
        assert len(typed_rows) == len([release for release in releases if release.model_type == model_type])
        assert DATED_MODEL_TABLE_HEADER in typed_overview
        assert all(f"| {model_type} |" not in row for row in typed_rows)
        assert ".compact-model-tables .fern-table th:first-child" in typed_overview
        assert ".compact-model-tables .fern-table th:nth-last-child(2)" in typed_overview
        assert "<Tabs>" not in typed_overview
    homepage = (tmp_path / "docs" / "index.mdx").read_text(encoding="utf-8")
    for document in (support_log, homepage):
        assert document.count('<div className="compact-model-tables">') == 1
        assert document.count(".compact-model-tables .fern-table-root") == 1
        assert "width: 100% !important;" in document
        assert ".compact-model-tables .fern-table td:last-child" in document
        assert document.count("|:-----|:-----|:-----|:-----|") == 1
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
    assert _sync_tables(tmp_path, check=True, validate_model_cards=False) == []

    # A newly published card must enter every table before a recipe is merged.
    new_card = tmp_path / "docs/model-coverage/llm/org/new.mdx"
    new_card.parent.mkdir(parents=True)
    new_card.write_text(
        '---\ntitle: "New"\nslug: model-coverage/large-language-models/org/new\n---\n'
        "[org/new](https://huggingface.co/org/new)\n"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", str(new_card)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "-c",
            f"core.hooksPath={os.devnull}",
            "commit",
            "-qm",
            "publish new model card",
            "--date=2026-10-15T12:00:00Z",
        ],
        check=True,
        env={**os.environ, "GIT_COMMITTER_DATE": "2026-10-15T12:00:00Z"},
    )
    generated = _generate_tables(tmp_path, validate_model_cards=False)
    for relative in ("docs/index.mdx", "docs/model-coverage/latest-models.mdx", "docs/model-coverage/llm/index.mdx"):
        rows = [line for line in generated[tmp_path / relative].splitlines() if line.startswith("| 2026-")]
        assert "[New](/model-coverage/large-language-models/org/new)" in rows[0]
        assert "2026-10-15" in rows[0]
        assert rows[0].endswith("|  |")


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
                "-c",
                f"core.hooksPath={os.devnull}",
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
                "-c",
                f"core.hooksPath={os.devnull}",
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
    (tmp_path / "tools" / "docs").mkdir(parents=True)
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
    (tmp_path / "tools" / "docs" / "catalog.py").write_text("_DOC_ARCH_ALIASES = {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="latest-models.mdx"):
        _sync_tables(tmp_path, check=True, validate_model_cards=False)


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


def test_recipe_pr_requires_main_links_when_its_recipe_arrives(tmp_path, monkeypatch):
    from tools.docs.build import coverage_tables as coverage

    recipe_path = "examples/llm_finetune/new.yaml"
    permalink = f"https://github.com/NVIDIA-NeMo/Automodel/blob/{'b' * 40}/{recipe_path}"
    main_url = f"{coverage.REPOSITORY_URL}/{recipe_path}"
    document = f"[Recipe]({permalink})\n"
    monkeypatch.setattr(
        coverage, "_read_remote_recipe", lambda url: "model: {pretrained_model_name_or_path: org/new}\n"
    )
    # The docs PR can publish a validated recipe from another PR.
    assert coverage._validate_recipe_links(tmp_path, document) == permalink
    with pytest.raises(ValueError, match="recipe is missing"):
        coverage._validate_recipe_links(tmp_path, document + f"[Main]({main_url})")

    # The implementation PR contains the recipe, so it must fix every pinned link.
    local_recipe = tmp_path / recipe_path
    local_recipe.parent.mkdir(parents=True)
    local_recipe.write_text("model: {pretrained_model_name_or_path: org/new}\n")

    def unexpected_remote(url):
        pytest.fail("A checked-in recipe must never be read from a permalink")

    monkeypatch.setattr(coverage, "_read_remote_recipe", unexpected_remote)
    with pytest.raises(ValueError, match="exists in this checkout"):
        coverage._validate_recipe_links(tmp_path, document)
    # Adding a main link does not excuse a leftover permalink in the same card.
    with pytest.raises(ValueError, match="exists in this checkout"):
        coverage._validate_recipe_links(tmp_path, document + f"[Main]({main_url})")
    assert coverage._validate_recipe_links(tmp_path, document.replace(permalink, main_url)) == main_url


@pytest.mark.parametrize("source", ["[", "[]"])
def test_linked_recipe_yaml_is_validated(tmp_path, monkeypatch, source):
    from tools.docs.build import coverage_tables as coverage

    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'c' * 40}/examples/new.yaml)"
    monkeypatch.setattr(coverage, "_read_remote_recipe", lambda url: source)
    with pytest.raises(ValueError, match="Invalid recipe links"):
        coverage._validate_recipe_links(tmp_path, document)


def test_missing_permalink_target_fails(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    from tools.docs.build import coverage_tables as coverage

    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'d' * 40}/examples/missing.yaml)"

    def missing(url):
        raise HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(coverage, "_read_remote_recipe", missing)
    with pytest.raises(ValueError, match="404"):
        coverage._validate_recipe_links(tmp_path, document)


def test_recent_support_uses_permalink_for_documented_card(tmp_path, monkeypatch):
    from tools.docs.build import coverage_tables as coverage

    model = coverage._ModelDoc("LLM", "/models/new", (), "New", "docs/model-coverage/llm/org/new.mdx")
    recipe_url = f"https://github.com/NVIDIA-NeMo/Automodel/blob/{'a' * 40}/examples/llm_finetune/new.yaml"
    monkeypatch.setattr(coverage, "_load_file_addition_date", lambda root, path: "2026-10-15")
    old_releases = [
        coverage._ModelRelease("2026-10-01", f"org/old-{i}", f"/models/old-{i}", "LLM", f"examples/old-{i}.yaml")
        for i in range(10)
    ]
    entries = coverage._collect_support_entries(tmp_path, old_releases, [model], {model.docs_page: recipe_url})
    for output in (
        coverage._render_homepage_table(entries),
        coverage._render_support_log_table(entries),
        coverage._render_support_log_table(entries, include_type=False),
    ):
        assert "[New](/models/new)" in output
        assert f"[recipe]({recipe_url})" in output
    assert coverage._render_homepage_table(entries).count("| 2026-") == 10
    assert coverage._render_support_log_table(entries).count("| 2026-") == 11
    assert entries[0].date == "2026-10-15"

    merged = coverage._ModelRelease("2026-10-16", "org/new", model.docs_page, "LLM", "examples/llm_finetune/new.yaml")
    after = coverage._collect_support_entries(tmp_path, [*old_releases, merged], [model], {})
    assert len(after) == len(entries)
    assert "/blob/main/examples/llm_finetune/new.yaml" in after[0].recipe_link

    without_recipe = coverage._collect_support_entries(tmp_path, old_releases, [model], {})
    assert without_recipe[0].recipe_link == ""
