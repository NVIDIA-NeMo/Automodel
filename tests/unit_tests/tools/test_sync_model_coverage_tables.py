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

# Over the default 5s budget on purpose: this module drives git through subprocesses over throwaway repositories.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


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
            section_path = item.get("path")
            if config_dir is not None and isinstance(section_path, str):
                frontmatter_slug = _frontmatter_slug((config_dir / section_path).resolve())
                if frontmatter_slug:
                    routes.add("/" + frontmatter_slug)
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


def test_model_coverage_pages_use_provider_sections_and_checkpoint_slugs():
    repo_root = Path(__file__).parents[3]
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
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
    provider_sprites: set[str] = set()

    for category in (item for item in model_coverage["contents"] if "section" in item):
        category_slug = str(category.get("slug"))
        category_directory = category_directories[category_slug]
        provider_sections = category.get("contents", [])[1:]
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

            icon = provider.get("icon")
            if not isinstance(icon, str):
                offenders.append(f"{provider_slug}: missing provider icon")
            else:
                sprite_match = re.search(r"data:image/webp;base64,([A-Za-z0-9+/=]+)", icon)
                if sprite_match is None:
                    offenders.append(f"{provider_slug}: provider icon does not embed the shared sprite")
                else:
                    provider_sprites.add(sprite_match.group(1))
            provider_slugs.add(provider_slug)

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
                sidebar_label = page.get("page")
                if not isinstance(sidebar_label, str) or not sidebar_label:
                    offenders.append(f"{path}: model entry has no sidebar label")
                elif sidebar_label[0].islower():
                    offenders.append(f"{path}: sidebar label must use human-readable capitalization")
                document = model_path.read_text(encoding="utf-8")
                hf_models = re.findall(
                    r"https://huggingface\.co/[A-Za-z0-9_.-]+/([A-Za-z0-9_.-]+)",
                    document,
                )
                if model_id not in hf_models:
                    offenders.append(f"{path}: URL model ID {model_id!r} is not linked by the card")

    expected_model_pages = {
        path.resolve()
        for category_directory in category_directories.values()
        for path in (repo_root / "docs" / "model-coverage" / category_directory).glob("*/*.mdx")
        if path.name != "index.mdx"
    }
    if set(navigated_model_pages) != expected_model_pages:
        offenders.append("nightly navigation does not contain every model card exactly once")
    if len(navigated_model_pages) != len(set(navigated_model_pages)):
        offenders.append("nightly navigation contains duplicate model cards")
    if len(provider_sprites) != 1:
        offenders.append(f"provider icons must share one base64 sprite, found {len(provider_sprites)}")
    else:
        sprite = base64.b64decode(next(iter(provider_sprites)), validate=True)
        if not (sprite.startswith(b"RIFF") and sprite[8:12] == b"WEBP"):
            offenders.append("embedded provider sprite is not WebP")
    legacy_icons = list((repo_root / "docs" / "fern" / "assets" / "providers").glob("*.png"))
    if legacy_icons:
        offenders.append("individual provider icons remain alongside the embedded sprite")
    if (repo_root / "docs" / "fern" / "assets" / "provider-sprite.svg").exists():
        offenders.append("provider sprite must be embedded in navigation rather than stored as an external asset")

    assert not offenders, "Model coverage provider hierarchy violations:\n" + "\n".join(
        f"  - {offender}" for offender in offenders
    )


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
    return organization.casefold(), normalized_stem


def test_recipe_backed_model_sizes_have_exact_index_routes_and_one_card():
    repo_root = Path(__file__).parents[3]
    model_docs, _ = _load_model_docs(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    unavailable_hf_models = {
        # The checked-in recipe currently names an unpublished Hugging Face repository.
        "nvidia/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained",
    }
    category_by_type = {
        "LLM": "large-language-models",
        "Encoder-Decoder": "large-language-models",
        "VLM": "vision-language-models",
        "Multimodal": "multimodal",
        "Omni": "omni",
        "dLLM": "dllm",
        "Diffusion": "diffusion",
        "Embedding": "embedding-models",
        "Reranking": "reranking-models",
    }

    index_routes: dict[str, set[str]] = {}
    providers_by_owner: dict[tuple[str, str], set[str]] = {}
    duplicates: list[tuple[str, str]] = []
    for index_path in (repo_root / "docs" / "model-coverage").glob("*/*/index.mdx"):
        provider_route = _frontmatter_slug(index_path)
        assert provider_route is not None
        provider_href = f"/{provider_route}"
        category = provider_route.split("/")[1]
        document = index_path.read_text(encoding="utf-8")
        labels = set()
        for label, href in re.findall(r"^- \[`([^`]+)`\]\((/model-coverage/[^)]+)\)$", document, re.MULTILINE):
            normalized_label = label.casefold()
            if normalized_label in labels:
                duplicates.append((provider_href, label))
            labels.add(normalized_label)
            index_routes.setdefault(provider_href, set()).add(href)
        for card_path in index_path.parent.glob("*.mdx"):
            if card_path.name == "index.mdx":
                continue
            card = card_path.read_text(encoding="utf-8")
            for owner in re.findall(r"https://huggingface\.co/([A-Za-z0-9_.-]+)/[A-Za-z0-9_.-]+", card):
                providers_by_owner.setdefault((category, owner.casefold()), set()).add(provider_href)
    assert not duplicates, f"Duplicate model names within one provider index: {duplicates}"

    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    navigated_routes = _collect_fern_routes(navigation, config_dir=config_path.parent)

    destinations_by_group: dict[tuple[str, tuple[str, str]], set[str]] = {}
    destination_groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    missing = []
    for release in releases:
        if release.hf_model_id in unavailable_hf_models:
            continue
        category = category_by_type[release.model_type]
        owner, model_name = release.hf_model_id.split("/", 1)
        model_name = model_name.replace("_", "-")
        provider_hrefs = {
            provider_href
            for provider_href in providers_by_owner.get((category, owner.casefold()), set())
            if f"{provider_href}/{model_name}" in navigated_routes
            or f"/nemo/automodel{provider_href}/{model_name}" in redirects
        }
        if len(provider_hrefs) != 1:
            missing.append((release.model_type, release.hf_model_id, f"provider mapping {sorted(provider_hrefs)}"))
            continue
        provider_href = next(iter(provider_hrefs))
        exact_href = f"{provider_href}/{model_name}"
        source = f"/nemo/automodel{exact_href}"
        destination = redirects.get(source, source).removeprefix("/nemo/automodel")
        if destination not in navigated_routes:
            missing.append((release.model_type, release.hf_model_id, f"navigated card {destination}"))
            continue
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
    assert len(sources) == len(set(sources)), "Fern redirects contain duplicate sources"
    assert all(redirect["source"] != redirect["destination"] for redirect in redirects)

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
    assert not canonical_model_sources.intersection(sources), "A canonical model URL must not also be a redirect source"

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


def test_internal_model_coverage_links_resolve_to_nightly_routes():
    repo_root = Path(__file__).parents[3]
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}
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
            route = link.rstrip("/")
            redirected = redirects.get(f"/nemo/automodel{route}", "").removeprefix("/nemo/automodel")
            if route not in routes and redirected not in routes:
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
