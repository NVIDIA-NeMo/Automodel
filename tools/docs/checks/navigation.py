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

"""Validate navigation, provider hierarchy, routes, redirects, and links."""

import base64
import re
from pathlib import Path
from typing import TypedDict

import yaml

from tools.docs.build.coverage_tables import (
    _load_model_doc_catalog,
    _load_model_docs,
    _load_model_releases,
    _resolve_model_card_routes,
)
from tools.docs.cards import _frontmatter_slug, _model_card_aliases, _model_card_paths
from tools.docs.catalog import (
    MAX_TOTAL_ENCODED_BYTES,
    PROVIDER_ORGS,
)

_NavigationItem = TypedDict(
    "_NavigationItem",
    {"page": str, "section": str, "path": str, "slug": str, "skip-slug": bool, "contents": list["_NavigationItem"]},
    total=False,
)


def _fern_slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    return re.sub("[^a-z0-9]+", "-", value).strip("-")


def _collect_fern_routes(
    items: list[_NavigationItem], parents: tuple[str, ...] = (), config_dir: Path | None = None
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


def _model_size_key(hf_model_id: str) -> tuple[str, str]:
    (organization, model_name) = hf_model_id.split("/", 1)
    size_pattern = re.compile("(?i)(?:^|[-_])(?:A?\\d+(?:\\.\\d+)?[BMT])(?=$|[-_])")
    size_matches = list(size_pattern.finditer(model_name))
    if size_matches:
        model_stem = model_name[: size_matches[-1].end()]
    else:
        variant_suffixes = re.compile(
            "(?i)(?:[-_](?:instruct|chat|base|thinking|reasoning|pt|it|bf16|fp8|flash|preview|deep|hf))+$"
        )
        model_stem = variant_suffixes.sub("", model_name)
    normalized_stem = model_stem.casefold().replace("_", "-")
    if organization.casefold() == "meta-llama":
        normalized_stem = re.sub("^meta-", "", normalized_stem)
    size_aliases = {("xiaomimimo", "mimo-v2.6-flash-rl"): "mimo-v2"}
    normalized_stem = size_aliases.get((organization.casefold(), normalized_stem), normalized_stem)
    return (organization.casefold(), normalized_stem)


def _validate_published_training_routes(repo_root: Path) -> None:
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    expected = {
        "/recipes-e2e-examples/minimax-m3",
        "/recipes-e2e-examples/databricks",
        "/recipes-e2e-examples/eagle-speculative-decoding",
        "/recipes-e2e-examples/diffusion-fine-tuning",
        "/tutorials/overview",
    }
    missing = sorted(expected - routes)
    if missing:
        raise ValueError(f"Published training routes missing from nightly navigation: {missing}")
    if "/job-launchers/databricks" in routes:
        raise ValueError("Databricks must use /recipes-e2e-examples/databricks instead of /job-launchers/databricks")


def _validate_release_navigation(repo_root: Path) -> None:
    (model_docs, _) = _load_model_docs(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    internal_pages = {release.docs_page for release in releases if release.docs_page.startswith("/")}
    missing_pages = sorted(internal_pages - _collect_fern_routes(navigation, config_dir=config_path.parent))
    if missing_pages:
        raise ValueError(f"Model release docs pages missing from nightly navigation: {missing_pages}")


def _validate_provider_sections(repo_root: Path) -> None:
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {item["source"]: item["destination"] for item in docs_config["redirects"]}
    if docs_config.get("theme", {}).get("sidebar") != "default":
        raise ValueError("Provider icons require the default sidebar")
    if docs_config.get("layout", {}).get("breadcrumbs", {}).get("current-page") is not True:
        raise ValueError("Provider breadcrumbs require layout.breadcrumbs.current-page: true")
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    model_coverage = next((item for item in navigation if item.get("section") == "Model Coverage"))
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
                for label, href in re.findall("^- \\[`([^`]+)`\\]\\(([^)]+)\\)$", index_document, re.MULTILINE):
                    checkpoint_name = href.rstrip("/").rsplit("/", 1)[-1]
                    label_route = href.rsplit("/", 1)[0] + "/" + label
                    declared_alias = redirects.get("/nemo/automodel" + label_route) == "/nemo/automodel" + href
                    if label.casefold() != checkpoint_name.casefold() and (not declared_alias):
                        offenders.append(
                            f"{provider_slug}: provider index label {label!r} does not match {checkpoint_name!r}"
                        )
            icon = provider.get("icon")
            if not isinstance(icon, str):
                offenders.append(f"{provider_slug}: missing provider icon")
            else:
                icon_match = re.search("data:image/webp;base64,([A-Za-z0-9+/=]+)", icon)
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
            provider_models[category_slug, provider_slug] = model_labels
            if model_labels != sorted(model_labels, key=str.casefold):
                offenders.append(f"{category_slug}/{provider_slug}: model entries are not alphabetized")
            for page in provider.get("contents", []):
                path = page.get("path")
                if not isinstance(path, str):
                    offenders.append(f"{provider_slug}: model entry has no path")
                    continue
                model_path = (config_path.parent / path).resolve()
                navigated_model_pages.append(model_path)
                frontmatter_slug = _frontmatter_slug(model_path)
                if frontmatter_slug is None or not frontmatter_slug.startswith(expected_provider_route + "/"):
                    offenders.append(f"{path}: invalid model route {frontmatter_slug!r}")
                    continue
                model_id = frontmatter_slug.rsplit("/", 1)[1]
                document = model_path.read_text(encoding="utf-8")
                title_match = re.search('^title: "?([^"\\r\\n]+)"?$', document, flags=re.MULTILINE)
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
                hf_checkpoints = re.findall("https://huggingface\\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", document)
                canonical_publishers = {owner for (owner, checkpoint) in hf_checkpoints if checkpoint == model_id}
                if not canonical_publishers:
                    offenders.append(f"{path}: URL model ID {model_id!r} is not linked by the card")
                expected_publisher = PROVIDER_ORGS.get(provider_slug)
                if expected_publisher is None or all(
                    (owner.casefold() != expected_publisher.casefold() for owner in canonical_publishers)
                ):
                    offenders.append(
                        f"{path}: checkpoint publisher {sorted(canonical_publishers)!r} does not match provider {provider_slug!r}"
                    )
    if "Llama" in provider_models.get(("large-language-models", "meta"), []):
        offenders.append("large-language-models/meta: catch-all Llama label remains")
    for category_slug in ("large-language-models", "vision-language-models"):
        if "Muse-Glimmer-30B" not in provider_models.get((category_slug, "meta-models"), []):
            offenders.append(f"{category_slug}/meta-models: Muse-Glimmer-30B is not grouped under Meta Models")
        if (category_slug, "muse") in provider_models:
            offenders.append(f"{category_slug}: Muse remains a standalone provider")
    aliases = _model_card_aliases(repo_root)
    expected_model_pages = {path.resolve() for path in _model_card_paths(repo_root) if path not in aliases}
    if set(navigated_model_pages) != expected_model_pages:
        offenders.append("nightly navigation does not contain every model card exactly once")
    if len(navigated_model_pages) != len(set(navigated_model_pages)):
        offenders.append("nightly navigation contains duplicate model cards")
    inconsistent_icons = {provider: len(icons) for (provider, icons) in provider_icons.items() if len(icons) != 1}
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
    if offenders:
        raise ValueError(
            "Model coverage provider hierarchy violations:\n" + "\n".join((f"  - {offender}" for offender in offenders))
        )


def _validate_recipe_routes(repo_root: Path) -> None:
    (model_docs, _, documented_models) = _load_model_doc_catalog(repo_root / "docs")
    releases = _load_model_releases(repo_root, model_docs)
    model_card_routes = _resolve_model_card_routes(repo_root, releases, model_docs, documented_models)
    index_routes: dict[str, set[str]] = {}
    duplicates: list[tuple[str, str]] = []
    for index_path in (repo_root / "docs" / "model-coverage").glob("*/*/index.mdx"):
        provider_route = _frontmatter_slug(index_path)
        if provider_route is None:
            raise ValueError(f"{index_path.relative_to(repo_root)}: provider index must declare a frontmatter slug")
        provider_href = f"/{provider_route}"
        document = index_path.read_text(encoding="utf-8")
        labels = set()
        for label, href in re.findall("^- \\[`([^`]+)`\\]\\((/model-coverage/[^)]+)\\)$", document, re.MULTILINE):
            normalized_label = label.casefold()
            if normalized_label in labels:
                duplicates.append((provider_href, label))
            labels.add(normalized_label)
            index_routes.setdefault(provider_href, set()).add(href)
    if duplicates:
        raise ValueError(f"Duplicate model names within one provider index: {duplicates}")
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}
    destinations_by_group: dict[tuple[str, tuple[str, str]], set[str]] = {}
    destination_groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    missing = []
    for release in releases:
        destination = model_card_routes[release.hf_model_id, release.model_type]
        provider_href = destination.rsplit("/", 1)[0]
        if destination not in index_routes.get(provider_href, set()):
            missing.append((release.model_type, release.hf_model_id, f"provider index card {destination}"))
            continue
        group = (release.model_type, _model_size_key(release.hf_model_id))
        destinations_by_group.setdefault(group, set()).add(destination)
        destination_groups.setdefault((release.model_type, destination), set()).add(group[1])
    split_groups = {
        group: destinations for (group, destinations) in destinations_by_group.items() if len(destinations) != 1
    }
    merged_groups = {key: groups for (key, groups) in destination_groups.items() if len(groups) != 1}
    if missing:
        raise ValueError(f"Recipe-backed checkpoints without exact model-card routes: {missing}")
    if split_groups:
        raise ValueError(f"Same-size checkpoint variants resolve to different cards: {split_groups}")
    if merged_groups:
        raise ValueError(f"Different model sizes resolve to the same card: {merged_groups}")
    meta_aliases = (
        "/model-coverage/large-language-models/meta/Meta-Llama-3.1-8B",
        "/model-coverage/large-language-models/meta/Meta-Llama-3.1-8B-Instruct",
    )
    meta_destinations = {redirects[f"/nemo/automodel{route}"].removeprefix("/nemo/automodel") for route in meta_aliases}
    if meta_destinations != {"/model-coverage/large-language-models/meta/Llama-3.1-8B"}:
        raise ValueError(
            f"Meta Llama 3.1 aliases must redirect to the canonical Llama-3.1-8B card: {meta_destinations}"
        )


def _validate_redirects(repo_root: Path) -> None:
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = docs_config["redirects"]
    sources = [redirect["source"] for redirect in redirects]
    folded_sources = [source.casefold() for source in sources]
    if len(folded_sources) != len(set(folded_sources)):
        raise ValueError("Fern redirects contain case-insensitive duplicate sources")
    if not all((redirect["source"].casefold() != redirect["destination"].casefold() for redirect in redirects)):
        raise ValueError("Fern redirects must not point to themselves, including case-only differences")
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    provider_routes = {
        route
        for route in routes
        if re.fullmatch(
            "/model-coverage/(?:large-language-models|vision-language-models|multimodal|omni|dllm|diffusion|embedding-models|reranking-models)/[^/]+",
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
        if destination not in routes:
            raise ValueError(redirect)
    canonical_model_sources = {
        f"/nemo/automodel{route}" for route in routes - provider_routes if route.startswith("/model-coverage/")
    }
    canonical_model_sources |= {
        source.replace("/nemo/automodel/", "/nemo/automodel/nightly/", 1) for source in canonical_model_sources
    }
    canonical_model_sources = {source.casefold() for source in canonical_model_sources}
    if canonical_model_sources.intersection(folded_sources):
        raise ValueError("A canonical model URL must not also be a case-insensitive redirect source")
    expected_moonlight_redirects = {
        "/nemo/automodel/model-coverage/large-language-models/moonshotai/moonlight": "/nemo/automodel/model-coverage/large-language-models/moonshotai",
        "/nemo/automodel/nightly/model-coverage/large-language-models/moonshotai/moonlight": "/nemo/automodel/nightly/model-coverage/large-language-models/moonshotai",
    }
    redirects_by_source = {redirect["source"]: redirect["destination"] for redirect in redirects}
    for source, destination in expected_moonlight_redirects.items():
        if redirects_by_source.get(source) != destination:
            raise ValueError(f"Published Moonlight route {source} must redirect to {destination}")


def _validate_internal_links(repo_root: Path) -> None:
    docs_config = yaml.safe_load((repo_root / "docs" / "fern" / "docs.yml").read_text(encoding="utf-8"))
    redirects = {redirect["source"]: redirect["destination"] for redirect in docs_config["redirects"]}
    config_path = repo_root / "docs" / "fern" / "versions" / "nightly.yml"
    navigation = yaml.safe_load(config_path.read_text(encoding="utf-8"))["navigation"]
    routes = _collect_fern_routes(navigation, config_dir=config_path.parent)
    broken_links: list[tuple[Path, str]] = []
    pages = set((repo_root / "docs" / "model-coverage").rglob("*.mdx")) | set(_model_card_paths(repo_root))
    for page in sorted(pages):
        document = page.read_text(encoding="utf-8")
        for link in re.findall("\\]\\((/[^)#?]+)", document):
            if link == "/model-coverage/llms.txt":
                continue
            route = link.rstrip("/")
            redirected = redirects.get(f"/nemo/automodel{route}", "").removeprefix("/nemo/automodel")
            if route not in routes and redirected not in routes:
                broken_links.append((page.relative_to(repo_root), link))
    if broken_links:
        raise ValueError(
            "Model coverage pages link to missing nightly routes:\n"
            + "\n".join((f"  - {page}: {link}" for (page, link) in broken_links))
        )
