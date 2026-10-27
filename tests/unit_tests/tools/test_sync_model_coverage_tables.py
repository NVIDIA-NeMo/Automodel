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

import base64
import io
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest
import yaml

from tests.ci_tests.utils.sync_model_coverage_tables import (
    DATED_MODEL_TABLE_HEADER,
    DATED_SUPPORT_TABLE_HEADER,
    HOMEPAGE_END_MARKER,
    HOMEPAGE_START_MARKER,
    MODEL_CARD_PATH_ALIASES,
    MODEL_TYPE_OVERVIEW_PATHS,
    REGISTRY_END_MARKER,
    REGISTRY_START_MARKER,
    SUPPORT_LOG_END_MARKER,
    SUPPORT_LOG_START_MARKER,
    TABLE_ROW_COUNT,
    _generate_tables,
    _iter_pretrained_model_ids,
    _load_model_doc_catalog,
    _load_model_docs,
    _load_model_releases,
    _ModelDoc,
    _ModelRelease,
    _parse_doc_arch_aliases,
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
from tools.sync_fern_provider_icons import MAX_TOTAL_ENCODED_BYTES, PROVIDER_ORGS

# Over the default 5s budget on purpose: this module drives git through subprocesses over throwaway repositories.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(70)


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


def _model_card_paths(repo_root: Path) -> list[Path]:
    coverage_root = repo_root / "docs" / "model-coverage"
    return sorted(
        path
        for path in coverage_root.rglob("*.mdx")
        if path.name != "index.mdx" and len(path.relative_to(coverage_root).parts) == 3
    )


def _collect_fern_routes(
    items: list[dict[str, object]],
    parents: tuple[str, ...] = (),
    config_dir: Path | None = None,
) -> set[str]:
    routes = set()
    for item in items:
        if "section" in item:
            slug = str(item.get("slug", _fern_slug(str(item["section"]))))
            section_path = item.get("path")
            if config_dir is not None and isinstance(section_path, str):
                frontmatter_slug = _frontmatter_slug((config_dir / section_path).resolve())
                if frontmatter_slug:
                    routes.add("/" + frontmatter_slug)
            child_parents = parents if item.get("skip-slug") else (*parents, slug)
            routes.update(_collect_fern_routes(item.get("contents", []), child_parents, config_dir))
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


def test_regrouped_training_pages_preserve_published_urls():
    config_path = Path(__file__).parents[3] / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)

    assert {
        "/recipes-e2e-examples/minimax-m3",
        "/recipes-e2e-examples/databricks",
        "/recipes-e2e-examples/eagle-speculative-decoding",
        "/recipes-e2e-examples/diffusion-fine-tuning",
        "/tutorials/overview",
    } <= routes
    assert "/job-launchers/databricks" not in routes


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
    repo_root = Path(__file__).parents[3]
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


@pytest.mark.parametrize("canonical_exists", [False, True])
def test_trusted_catalog_resolves_cards_in_checkouts_before_and_after_a_rename(tmp_path, canonical_exists):
    alias = "docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx"
    canonical = MODEL_CARD_PATH_ALIASES[alias]
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


def test_model_card_path_aliases_preserve_published_paths():
    repo_root = Path(__file__).parents[3]
    _, _, catalog = _load_model_doc_catalog(repo_root / "docs")
    source_paths = [model.source_path for model in catalog]
    for alias, canonical in MODEL_CARD_PATH_ALIASES.items():
        assert (repo_root / alias).is_file(), f"Published source URL must remain available: {alias}"
        assert (repo_root / canonical).is_file(), f"Published source URL must remain available: {canonical}"
        assert (repo_root / alias).read_bytes() == (repo_root / canonical).read_bytes(), (
            f"Compatibility card must stay synchronized with {canonical}: {alias}"
        )
        assert _frontmatter_slug(repo_root / alias) == _frontmatter_slug(repo_root / canonical)
        assert alias not in source_paths
        assert source_paths.count(canonical) == 1


def _assert_model_card_recipe_identity(repo_root: Path, card: Path) -> None:
    document = card.read_text(encoding="utf-8")
    recipe_ids: set[str] = set()
    for match in re.finditer(
        r"https://github\.com/NVIDIA-NeMo/Automodel/blob/(?:main|[0-9a-f]{40})/"
        r"(examples/[A-Za-z0-9_./-]+\.yaml)",
        document,
    ):
        recipe = yaml.safe_load(_read_documented_recipe(repo_root, match[1], match[0]))
        assert isinstance(recipe, dict), f"{match[1]}: recipe must be a YAML mapping"
        recipe_ids.update(_iter_pretrained_model_ids(recipe.get("model", {})))
    metadata = yaml.safe_load(document.split("---", 2)[1])
    title = metadata["title"]
    if recipe_ids:
        names = {model_id.split("/", 1)[1] for model_id in recipe_ids}
        assert title in names, f"{card}: title {title!r} must match a linked recipe model ID: {sorted(recipe_ids)}"
    else:
        # Reference-only cards identify the checkpoint in their first HF model link.
        hf_ids = re.findall(r"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", document)
        assert hf_ids, f"{card}: reference card must link its Hugging Face checkpoint"
        assert title == hf_ids[0].split("/", 1)[1], f"{card}: title must match the primary HF checkpoint"
        recipe_ids = {hf_ids[0]}
    filename = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") + ".mdx"
    assert card.name == filename, f"{card}: filename must be {filename!r} for checkpoint {title!r}"
    matching_ids = {model_id for model_id in recipe_ids if model_id.split("/", 1)[1] == title}
    assert any(f"https://huggingface.co/{model_id}" in document for model_id in matching_ids), (
        f"{card}: link the exact Hugging Face checkpoint from the recipe"
    )


def test_changed_model_cards_match_linked_recipe_identity():
    repo_root = Path(__file__).parents[3]
    base = subprocess.check_output(["git", "merge-base", "HEAD", "origin/main"], cwd=repo_root, text=True).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "--", "docs/model-coverage"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    changed += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "docs/model-coverage"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    # Keep the original Step-5 regression covered after the PR is merged.
    changed += [f"docs/model-coverage/{kind}/stepfun-ai/step-5-preview-bf16.mdx" for kind in ("llm", "vlm")]
    canonical_paths = {MODEL_CARD_PATH_ALIASES.get(relative, relative) for relative in changed}
    for relative in sorted(canonical_paths):
        card = repo_root / relative
        if (
            len(card.relative_to(repo_root / "docs/model-coverage").parts) == 3
            and card.is_file()
            and card.suffix == ".mdx"
            and card.name != "index.mdx"
        ):
            _assert_model_card_recipe_identity(repo_root, card)


@pytest.mark.parametrize(
    ("title", "filename", "message"),
    [
        ("Step-5-Preview", "step-5-preview.mdx", "title .* must match a linked recipe model ID"),
        ("Step-5-Preview-BF16", "step-5-preview.mdx", "filename must be 'step-5-preview-bf16.mdx'"),
    ],
)
def test_model_card_identity_rejects_recipe_mismatch(tmp_path: Path, title: str, filename: str, message: str):
    recipe = tmp_path / "examples/llm_finetune/step5.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("model:\n  pretrained_model_name_or_path: stepfun-ai/Step-5-Preview-BF16\n")
    card = tmp_path / filename
    card.write_text(
        f'---\ntitle: "{title}"\n---\n'
        "[Old checkpoint](https://huggingface.co/stepfun-ai/Step-5-Preview)\n"
        "[BF16](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16)\n"
        "[SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/step5.yaml)\n"
    )
    with pytest.raises(AssertionError, match=message):
        _assert_model_card_recipe_identity(tmp_path, card)


@pytest.mark.parametrize("title,valid", [("Step-5-Preview", True), ("Step-5-Preview-BF16", False)])
def test_reference_card_identifies_fp8_checkpoint_without_bf16_recipe(tmp_path: Path, title: str, valid: bool):
    card = tmp_path / "step-5-preview.mdx"
    card.write_text(
        f'---\ntitle: "{title}"\n---\n'
        "[FP8](https://huggingface.co/stepfun-ai/Step-5-Preview)\n"
        "[BF16](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16)\n"
    )
    if valid:
        _assert_model_card_recipe_identity(tmp_path, card)
    else:
        with pytest.raises(AssertionError, match="primary HF checkpoint"):
            _assert_model_card_recipe_identity(tmp_path, card)


def test_model_coverage_pages_use_provider_sections_and_checkpoint_slugs():
    repo_root = Path(__file__).parents[3]
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {item["source"]: item["destination"] for item in docs_config["redirects"]}
    assert docs_config.get("theme", {}).get("sidebar") == "default", "Provider icons require the default sidebar"
    assert docs_config.get("layout", {}).get("breadcrumbs", {}).get("current-page") is True
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    model_coverage = next(item for item in navigation if item.get("section") == "Model Coverage")
    category_directories = {
        "large-language-models": "llm",
        "vision-language-models": "vlm",
        "multimodal": "multimodal",
        "omni": "omni",
        "dllm": "dllm",
        "diffusion": "diffusion",
        "embedding-models": "embedding",
        "reranking-models": "reranker",
    }
    offenders: list[str] = []
    navigated_model_pages: list[Path] = []
    provider_slugs: set[str] = set()
    provider_icons: dict[str, set[str]] = {}
    total_provider_icon_bytes = 0
    provider_models: dict[tuple[str, str], list[str]] = {}

    for category in (item for item in model_coverage["contents"] if "section" in item):
        category_slug = str(category.get("slug"))
        category_directory = category_directories[category_slug]
        provider_sections = category.get("contents", [])[1:]
        provider_labels = [str(provider.get("section")) for provider in provider_sections]
        if provider_labels != sorted(provider_labels, key=str.casefold):
            offenders.append(f"{category_slug}: provider sections are not alphabetized")
        for provider in provider_sections:
            if "section" not in provider:
                offenders.append(f"{category_slug}: expected provider section, found {provider}")
                continue
            provider_slug = str(provider.get("slug"))
            expected_index = f"../../model-coverage/{category_directory}/{provider_slug}/index.mdx"
            if provider.get("path") != expected_index:
                offenders.append(f"{provider_slug}: expected provider index {expected_index!r}")
            index_path = (config_path.parent / expected_index).resolve()
            expected_provider_route = f"model-coverage/{category_slug}/{provider_slug}"
            if not index_path.is_file() or _frontmatter_slug(index_path) != expected_provider_route:
                offenders.append(f"{provider_slug}: invalid provider index route")
            else:
                index_document = index_path.read_text(encoding="utf-8")
                for label, href in re.findall(r"^- \[`([^`]+)`\]\(([^)]+)\)$", index_document, re.MULTILINE):
                    checkpoint_name = href.rstrip("/").rsplit("/", 1)[-1]
                    # A corrected checkpoint name can redirect to its stable published URL.
                    label_route = href.rsplit("/", 1)[0] + "/" + label
                    declared_alias = redirects.get("/nemo/automodel" + label_route) == "/nemo/automodel" + href
                    if label.casefold() != checkpoint_name.casefold() and not declared_alias:
                        offenders.append(
                            f"{provider_slug}: provider index label {label!r} does not match {checkpoint_name!r}"
                        )

            icon = provider.get("icon")
            if not isinstance(icon, str):
                offenders.append(f"{provider_slug}: missing provider icon")
            else:
                icon_match = re.search(r"data:image/webp;base64,([A-Za-z0-9+/=]+)", icon)
                if icon_match is None:
                    offenders.append(f"{provider_slug}: provider icon does not embed a WebP logo")
                else:
                    encoded = icon_match.group(1)
                    provider_icons.setdefault(provider_slug, set()).add(encoded)
                    total_provider_icon_bytes += len(encoded)
                    provider_icon = base64.b64decode(encoded, validate=True)
                    if not (provider_icon.startswith(b"RIFF") and provider_icon[8:12] == b"WEBP"):
                        offenders.append(f"{provider_slug}: embedded provider icon is not WebP")
            provider_slugs.add(provider_slug)
            model_labels = [str(page.get("page")) for page in provider.get("contents", [])]
            provider_models[(category_slug, provider_slug)] = model_labels
            if model_labels != sorted(model_labels, key=str.casefold):
                offenders.append(f"{category_slug}/{provider_slug}: model entries are not alphabetized")

            for page in provider.get("contents", []):
                path = page.get("path")
                if not isinstance(path, str):
                    offenders.append(f"{provider_slug}: model entry has no path")
                    continue
                model_path = (config_path.parent / path).resolve()
                navigated_model_pages.append(model_path)
                relative_parts = model_path.relative_to(repo_root / "docs" / "model-coverage").parts
                if relative_parts[:2] != (category_directory, provider_slug):
                    offenders.append(f"{path}: nested under the wrong provider section")
                frontmatter_slug = _frontmatter_slug(model_path)
                if frontmatter_slug is None or not frontmatter_slug.startswith(expected_provider_route + "/"):
                    offenders.append(f"{path}: invalid model route {frontmatter_slug!r}")
                    continue
                model_id = frontmatter_slug.rsplit("/", 1)[1]
                document = model_path.read_text(encoding="utf-8")
                title_match = re.search(r'^title: "?([^"\r\n]+)"?$', document, flags=re.MULTILINE)
                if title_match is not None:
                    corrected_name = title_match.group(1)
                    corrected_route = frontmatter_slug.rsplit("/", 1)[0] + "/" + corrected_name
                    if redirects.get("/nemo/automodel/" + corrected_route) == "/nemo/automodel/" + frontmatter_slug:
                        model_id = corrected_name
                sidebar_label = page.get("page")
                if not isinstance(sidebar_label, str) or not sidebar_label:
                    offenders.append(f"{path}: model entry has no sidebar label")
                elif sidebar_label.casefold() != model_id.casefold():
                    offenders.append(f"{path}: sidebar label {sidebar_label!r} does not match {model_id!r}")
                if title_match is None or title_match.group(1).casefold() != model_id.casefold():
                    title = title_match.group(1) if title_match is not None else None
                    offenders.append(f"{path}: page title {title!r} does not match {model_id!r}")
                hf_checkpoints = re.findall(
                    r"https://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
                    document,
                )
                canonical_publishers = {owner for owner, checkpoint in hf_checkpoints if checkpoint == model_id}
                if not canonical_publishers:
                    offenders.append(f"{path}: URL model ID {model_id!r} is not linked by the card")
                expected_publisher = PROVIDER_ORGS.get(provider_slug)
                if expected_publisher is None or all(
                    owner.casefold() != expected_publisher.casefold() for owner in canonical_publishers
                ):
                    offenders.append(
                        f"{path}: checkpoint publisher {sorted(canonical_publishers)!r} does not match provider "
                        f"{provider_slug!r}"
                    )

    if "Llama" in provider_models.get(("large-language-models", "meta"), []):
        offenders.append("large-language-models/meta: catch-all Llama label remains")
    for category_slug in ("large-language-models", "vision-language-models"):
        if "Muse-Glimmer-30B" not in provider_models.get((category_slug, "meta-models"), []):
            offenders.append(f"{category_slug}/meta-models: Muse-Glimmer-30B is not grouped under Meta Models")
        if (category_slug, "muse") in provider_models:
            offenders.append(f"{category_slug}: Muse remains a standalone provider")

    expected_model_pages = {
        path.resolve()
        for category_directory in category_directories.values()
        for path in (repo_root / "docs" / "model-coverage" / category_directory).glob("*/*.mdx")
        if path.name != "index.mdx" and path.relative_to(repo_root).as_posix() not in MODEL_CARD_PATH_ALIASES
    }
    if set(navigated_model_pages) != expected_model_pages:
        offenders.append("nightly navigation does not contain every model card exactly once")
    if len(navigated_model_pages) != len(set(navigated_model_pages)):
        offenders.append("nightly navigation contains duplicate model cards")
    inconsistent_icons = {provider: len(icons) for provider, icons in provider_icons.items() if len(icons) != 1}
    if inconsistent_icons:
        offenders.append(f"providers use inconsistent icons across categories: {inconsistent_icons}")
    if total_provider_icon_bytes > MAX_TOTAL_ENCODED_BYTES:
        offenders.append(
            f"provider icon payload is {total_provider_icon_bytes} bytes; expected at most {MAX_TOTAL_ENCODED_BYTES}"
        )
    legacy_icons = list((repo_root / "docs" / "fern" / "assets" / "providers").glob("*.png"))
    if legacy_icons:
        offenders.append("individual provider icons remain alongside the embedded sprite")
    if (repo_root / "docs" / "fern" / "assets" / "provider-sprite.svg").exists():
        offenders.append("provider icons must be embedded in navigation rather than stored as an external asset")

    assert not offenders, "Model coverage provider hierarchy violations:\n" + "\n".join(
        f"  - {offender}" for offender in offenders
    )


def test_model_card_descriptions_meet_seo_contract():
    repo_root = Path(__file__).parents[3]
    offenders: list[str] = []

    for path in _model_card_paths(repo_root):
        document = path.read_text(encoding="utf-8")
        closing = document.find("\n---\n", 4)
        frontmatter = yaml.safe_load(document[4:closing]) or {}
        title = str(frontmatter.get("title", ""))
        description = str(frontmatter.get("description", ""))
        reasons = []
        if not 120 <= len(description) <= 170:
            reasons.append(f"length {len(description)} is outside 120-170 characters")
        if title not in description:
            reasons.append(f"does not contain the exact title {title!r}")
        if "NeMo AutoModel" not in description:
            reasons.append("does not contain 'NeMo AutoModel'")
        if not description.isascii():
            reasons.append("contains non-ASCII characters")
        if reasons:
            offenders.append(f"{path.relative_to(repo_root)}: {', '.join(reasons)}")

    assert not offenders, "Invalid model-card descriptions:\n" + "\n".join(f"  - {item}" for item in offenders)


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


def _is_local_checkpoint_override(value: str, document: str) -> bool:
    """Recognize literal local paths and paths exported earlier in the example."""
    value = value.strip("\"'")
    variable = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if variable:
        assignment = re.search(rf"^export {variable.group(1)}=([^\r\n]+)$", document, re.MULTILINE)
        if assignment is None:
            return False
        value = assignment.group(1).strip().strip("\"'")
    return Path(value).is_absolute()


@pytest.mark.parametrize(
    "value,document,expected",
    [
        ("/data/model", "", True),
        ('"/data/model"', "", True),
        ('"${MODEL}"', "export MODEL=/data/model", True),
        ('"${MODEL}"', "", False),
        ('"${MODEL}"', "export MODEL=other/model", False),
        ("other/model", "", False),
    ],
)
def test_local_checkpoint_override(value, document, expected):
    assert _is_local_checkpoint_override(value, document) is expected


def _model_card_command_violations(repo_root: Path) -> list[str]:
    offenders: list[str] = []

    for path in _model_card_paths(repo_root):
        document = path.read_text(encoding="utf-8")
        lines = document.splitlines()
        documented_model_ids = set(
            re.findall(
                r"https://huggingface\.co/([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)",
                document,
            )
        )
        for index, line in enumerate(lines):
            if re.search(r"(?:^|\s)(?:uv run )?(?:automodel|torchrun)\s", line) is None:
                continue
            command_lines = [line]
            while command_lines[-1].rstrip().endswith("\\") and index + len(command_lines) < len(lines):
                command_lines.append(lines[index + len(command_lines)])
            command = "\n".join(command_lines)
            if "<recipe-path>" in command:
                offenders.append(f"{path.relative_to(repo_root)}: contains a recipe placeholder")
                continue
            checkpoint_override = re.search(r"--model\.pretrained_model_name_or_path\s+(\S+)", command)
            if checkpoint_override and not _is_local_checkpoint_override(checkpoint_override.group(1), document):
                offenders.append(f"{path.relative_to(repo_root)}: substitutes a checkpoint into a different recipe")
            recipe_paths = re.findall(r"examples/[A-Za-z0-9_./-]+\.yaml", command)
            if not recipe_paths and re.search(r"-c /[A-Za-z0-9_./-]+\.yaml", command):
                # A local edited config must name its checked-in source recipe in the instructions.
                recipe_paths = re.findall(
                    r"Copy `(examples/[A-Za-z0-9_./-]+\.yaml)` to a local configuration", document
                )
            if len(recipe_paths) != 1:
                offenders.append(
                    f"{path.relative_to(repo_root)}: command must reference exactly one recipe, found {recipe_paths}"
                )
                continue
            try:
                recipe_source = _read_documented_recipe(repo_root, recipe_paths[0], document)
                recipe = yaml.safe_load(recipe_source)
                if not isinstance(recipe, dict):
                    raise ValueError("Recipe must be a YAML mapping")
            except (OSError, ValueError, yaml.YAMLError) as error:
                offenders.append(f"{path.relative_to(repo_root)}: cannot read recipe {recipe_paths[0]}: {error}")
                continue
            recipe_model_ids = _iter_pretrained_model_ids(recipe.get("model", {}))
            documented_model_ids_by_case = {model_id.casefold() for model_id in documented_model_ids}
            recipe_model_ids_by_case = {model_id.casefold() for model_id in recipe_model_ids}
            if recipe_model_ids and documented_model_ids_by_case.isdisjoint(recipe_model_ids_by_case):
                offenders.append(
                    f"{path.relative_to(repo_root)}: {recipe_paths[0]} uses "
                    f"{', '.join(sorted(recipe_model_ids))}, but the card documents "
                    f"{', '.join(sorted(documented_model_ids)) or 'no Hugging Face model IDs'}"
                )

    return offenders


def test_model_card_commands_use_checked_in_recipes():
    offenders = _model_card_command_violations(Path(__file__).parents[3])
    assert not offenders, "Invalid model-card commands:\n" + "\n".join(f"  - {item}" for item in offenders)


def _model_size_key(hf_model_id: str) -> tuple[str, str]:
    organization, model_name = hf_model_id.split("/", 1)
    size_pattern = re.compile(r"(?i)(?:^|[-_])(?:A?\d+(?:\.\d+)?[BMT])(?=$|[-_])")
    size_matches = list(size_pattern.finditer(model_name))
    if size_matches:
        model_stem = model_name[: size_matches[-1].end()]
    else:
        variant_suffixes = re.compile(
            r"(?i)(?:[-_](?:instruct|chat|base|thinking|reasoning|pt|it|bf16|fp8|flash|preview|deep|hf))+$"
        )
        model_stem = variant_suffixes.sub("", model_name)
    normalized_stem = model_stem.casefold().replace("_", "-")
    if organization.casefold() == "meta-llama":
        normalized_stem = re.sub(r"^meta-", "", normalized_stem)
    # These Hub IDs omit a parameter count, but both checkpoints use the same 310B-scale MiMo-V2 backbone.
    size_aliases = {("xiaomimimo", "mimo-v2.6-flash-rl"): "mimo-v2"}
    normalized_stem = size_aliases.get((organization.casefold(), normalized_stem), normalized_stem)
    return organization.casefold(), normalized_stem


def test_recipe_backed_model_sizes_have_exact_index_routes_and_one_card():
    repo_root = Path(__file__).parents[3]
    model_docs, _, documented_models = _load_model_doc_catalog(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    model_card_routes = _resolve_model_card_routes(repo_root, releases, model_docs, documented_models)

    index_routes: dict[str, set[str]] = {}
    duplicates: list[tuple[str, str]] = []
    for index_path in (repo_root / "docs" / "model-coverage").glob("*/*/index.mdx"):
        provider_route = _frontmatter_slug(index_path)
        assert provider_route is not None
        provider_href = f"/{provider_route}"
        document = index_path.read_text(encoding="utf-8")
        labels = set()
        for label, href in re.findall(r"^- \[`([^`]+)`\]\((/model-coverage/[^)]+)\)$", document, re.MULTILINE):
            normalized_label = label.casefold()
            if normalized_label in labels:
                duplicates.append((provider_href, label))
            labels.add(normalized_label)
            index_routes.setdefault(provider_href, set()).add(href)
    assert not duplicates, f"Duplicate model names within one provider index: {duplicates}"

    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}

    destinations_by_group: dict[tuple[str, tuple[str, str]], set[str]] = {}
    destination_groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    missing = []
    for release in releases:
        destination = model_card_routes[(release.hf_model_id, release.model_type)]
        provider_href = destination.rsplit("/", 1)[0]
        if destination not in index_routes.get(provider_href, set()):
            missing.append((release.model_type, release.hf_model_id, f"provider index card {destination}"))
            continue
        group = (release.model_type, _model_size_key(release.hf_model_id))
        destinations_by_group.setdefault(group, set()).add(destination)
        destination_groups.setdefault((release.model_type, destination), set()).add(group[1])

    split_groups = {
        group: destinations for group, destinations in destinations_by_group.items() if len(destinations) != 1
    }
    merged_groups = {key: groups for key, groups in destination_groups.items() if len(groups) != 1}
    assert not missing, f"Recipe-backed checkpoints without exact model-card routes: {missing}"
    assert not split_groups, f"Same-size checkpoint variants resolve to different cards: {split_groups}"
    assert not merged_groups, f"Different model sizes resolve to the same card: {merged_groups}"

    meta_aliases = (
        "/model-coverage/large-language-models/meta/Meta-Llama-3.1-8B",
        "/model-coverage/large-language-models/meta/Meta-Llama-3.1-8B-Instruct",
    )
    meta_destinations = {redirects[f"/nemo/automodel{route}"].removeprefix("/nemo/automodel") for route in meta_aliases}
    assert meta_destinations == {"/model-coverage/large-language-models/meta/Llama-3.1-8B"}


def test_retired_model_routes_redirect_to_provider_indexes():
    repo_root = Path(__file__).parents[3]
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = docs_config["redirects"]
    sources = [redirect["source"] for redirect in redirects]
    folded_sources = [source.casefold() for source in sources]
    assert len(folded_sources) == len(set(folded_sources)), "Fern redirects contain case-insensitive duplicate sources"
    assert all(redirect["source"].casefold() != redirect["destination"].casefold() for redirect in redirects)

    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    provider_routes = {
        route
        for route in routes
        if re.fullmatch(
            r"/model-coverage/(?:large-language-models|vision-language-models|multimodal|omni|dllm|diffusion|embedding-models|reranking-models)/[^/]+",
            route,
        )
    }

    relevant_redirects = [
        redirect
        for redirect in redirects
        if redirect["source"].startswith("/nemo/automodel/nightly/model-coverage/")
        or redirect["source"].startswith("/nemo/automodel/model-coverage/")
    ]
    for redirect in relevant_redirects:
        destination = redirect["destination"].removeprefix("/nemo/automodel/nightly")
        destination = destination.removeprefix("/nemo/automodel")
        assert destination in routes, redirect

    canonical_model_sources = {
        f"/nemo/automodel{route}" for route in routes - provider_routes if route.startswith("/model-coverage/")
    }
    canonical_model_sources |= {
        source.replace("/nemo/automodel/", "/nemo/automodel/nightly/", 1) for source in canonical_model_sources
    }
    canonical_model_sources = {source.casefold() for source in canonical_model_sources}
    assert not canonical_model_sources.intersection(folded_sources), (
        "A canonical model URL must not also be a case-insensitive redirect source"
    )

    expected_moonlight_redirects = {
        "/nemo/automodel/model-coverage/large-language-models/moonshotai/moonlight": (
            "/nemo/automodel/model-coverage/large-language-models/moonshotai"
        ),
        "/nemo/automodel/nightly/model-coverage/large-language-models/moonshotai/moonlight": (
            "/nemo/automodel/nightly/model-coverage/large-language-models/moonshotai"
        ),
    }
    redirects_by_source = {redirect["source"]: redirect["destination"] for redirect in redirects}
    for source, destination in expected_moonlight_redirects.items():
        assert redirects_by_source.get(source) == destination


def test_internal_links_from_model_coverage_resolve_to_nightly_routes():
    repo_root = Path(__file__).parents[3]
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    broken_links: list[tuple[Path, str]] = []

    for page in (repo_root / "docs" / "model-coverage").rglob("*.mdx"):
        document = page.read_text(encoding="utf-8")
        for link in re.findall(r"\]\((/[^)#?]+)", document):
            # Fern serves a generated llms.txt index at every navigation level;
            # it is a virtual endpoint rather than an entry in nightly.yml.
            if link == "/model-coverage/llms.txt":
                continue
            route = link.rstrip("/")
            redirected = redirects.get(f"/nemo/automodel{route}", "").removeprefix("/nemo/automodel")
            if route not in routes and redirected not in routes:
                broken_links.append((page.relative_to(repo_root), link))

    assert not broken_links, "Model coverage pages link to missing nightly routes:\n" + "\n".join(
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
    from tests.ci_tests.utils import sync_model_coverage_tables as coverage

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
    from tests.ci_tests.utils import sync_model_coverage_tables as coverage

    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'c' * 40}/examples/new.yaml)"
    monkeypatch.setattr(coverage, "_read_remote_recipe", lambda url: source)
    with pytest.raises(ValueError, match="Invalid recipe links"):
        coverage._validate_recipe_links(tmp_path, document)


def test_missing_permalink_target_fails(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    from tests.ci_tests.utils import sync_model_coverage_tables as coverage

    document = f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'d' * 40}/examples/missing.yaml)"

    def missing(url):
        raise HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(coverage, "_read_remote_recipe", missing)
    with pytest.raises(ValueError, match="404"):
        coverage._validate_recipe_links(tmp_path, document)


def test_recent_support_uses_permalink_for_documented_card(tmp_path, monkeypatch):
    from tests.ci_tests.utils import sync_model_coverage_tables as coverage

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


def test_recipe_link_errors_include_every_link_card_and_compatibility_copy(tmp_path):
    from tests.ci_tests.utils import sync_model_coverage_tables as coverage

    models = []
    source_paths = ["docs/model-coverage/llm/first.mdx", "docs/model-coverage/llm/second.mdx"]
    alias = next(iter(MODEL_CARD_PATH_ALIASES))
    expected = []
    for index, source_path in enumerate([*source_paths, alias]):
        path = tmp_path / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        links = []
        for suffix in ("a", "b"):
            recipe_path = f"examples/model_{index}_{suffix}.yaml"
            recipe = tmp_path / recipe_path
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text("model: {}\n")
            links.append(f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{'a' * 40}/{recipe_path})")
            expected.append(recipe_path)
        path.write_text("\n".join(links))
        if source_path in source_paths:
            models.append(coverage._ModelDoc("LLM", f"/models/{index}", (), "Model", source_path))

    with pytest.raises(ValueError) as caught:
        coverage._collect_documented_recipe_urls(tmp_path, models)

    message = str(caught.value)
    for source_path in [*source_paths, alias]:
        assert source_path in message
    for recipe_path in expected:
        assert f"replace its commit permalink with {coverage.REPOSITORY_URL}/{recipe_path}" in message


def test_command_validation_collects_link_and_yaml_errors_across_cards(tmp_path):
    recipe_sources = {"pinned": "model: {}\n", "syntax": "[", "mapping": "[]"}
    for name, source in recipe_sources.items():
        recipe = tmp_path / "examples" / f"{name}.yaml"
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(source)
    for name in [*recipe_sources, "missing"]:
        card = tmp_path / "docs" / "model-coverage" / "llm" / "test" / f"{name}.mdx"
        card.parent.mkdir(parents=True, exist_ok=True)
        revision = "a" * 40 if name == "pinned" else "main"
        card.write_text(
            f"[Recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/{revision}/examples/{name}.yaml)\n"
            f"```bash\nuv run automodel -c examples/{name}.yaml\n```\n"
        )

    offenders = _model_card_command_violations(tmp_path)

    assert len(offenders) == 4
    for name in [*recipe_sources, "missing"]:
        assert any(f"{name}.mdx: cannot read recipe examples/{name}.yaml:" in item for item in offenders)
