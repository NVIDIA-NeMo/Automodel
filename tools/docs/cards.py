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

"""Discover model cards by route, preserving their published source paths."""

import re
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote, urlsplit

import yaml

from tools.docs.catalog import MODEL_TYPE_ROUTE_CATEGORIES


def checkpoint_links(document: str) -> set[str]:
    """Extract exact checkpoint identities from Hugging Face URLs.

    Args:
        document: Markdown text containing HTTPS Hugging Face links.

    Returns:
        Decoded publisher/repository IDs from complete URL path segments.
        Query strings, fragments, and paths inside the repository do not alter
        its identity. Invalid or encoded path separators are not accepted.
    """
    checkpoints = set()
    for url in re.findall(r"https://huggingface\.co/[^\s<>\"'`\])}]+", document):
        segments = [unquote(segment) for segment in urlsplit(url).path.split("/")[1:]]
        if len(segments) < 2 or any(segment in {".", ".."} for segment in segments):
            continue
        owner, repository = segments[:2]
        if re.fullmatch(r"[A-Za-z0-9_.-]+", owner) and re.fullmatch(r"[A-Za-z0-9_.-]+", repository):
            checkpoints.add(f"{owner}/{repository}")
    return checkpoints


def family_checkpoints(path: Path) -> set[str]:
    """Read the explicit checkpoint membership of a model family page.

    Args:
        path: Model-card MDX file to inspect.

    Returns:
        Declared checkpoint IDs, or an empty set when no family is declared.

    Raises:
        ValueError: Declared checkpoints are invalid, duplicated, inconsistent
            with the family title, unlinked, or published by different owners.
    """
    document = path.read_text(encoding="utf-8")
    if not document.startswith("---\n"):
        return set()
    closing = document.find("\n---\n", 4)
    if closing == -1:
        return set()
    metadata = yaml.safe_load(document[4:closing]) or {}
    if "checkpoints" not in metadata:
        return set()
    checkpoints = metadata["checkpoints"]
    title = metadata.get("title", "")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not all(isinstance(item, str) for item in checkpoints)
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise ValueError(f"{path}: checkpoints must be a nonempty list of unique Hugging Face model IDs")
    linked = checkpoint_links(document)
    for checkpoint in checkpoints:
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", checkpoint) is None
            or not checkpoint.split("/", 1)[1].startswith(f"{title}-")
            or checkpoint not in linked
        ):
            raise ValueError(f"{path}: family checkpoint {checkpoint!r} must match the title and have an exact HF link")
    if len({checkpoint.split("/", 1)[0] for checkpoint in checkpoints}) != 1:
        raise ValueError(f"{path}: family checkpoints must share one publisher")
    return set(checkpoints)


def _frontmatter_slug(path: Path) -> str | None:
    document = path.read_text(encoding="utf-8")
    if not document.startswith("---\n"):
        return None
    closing = document.find("\n---\n", 4)
    if closing == -1:
        return None
    metadata = yaml.safe_load(document[4:closing]) or {}
    slug = metadata.get("slug")
    return slug.strip().strip("/") if isinstance(slug, str) and slug.strip().strip("/") else None


def _model_card_paths(repo_root: Path) -> list[Path]:
    paths = []
    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.mdx")):
        relative = path.relative_to(docs_root)
        if relative.parts[0] == "fern":
            continue
        slug = _frontmatter_slug(path)
        route = slug.split("/") if slug else []
        is_model_route = (
            len(route) == 4 and route[0] == "model-coverage" and route[1] in MODEL_TYPE_ROUTE_CATEGORIES.values()
        )
        # Retain validation for cards missing metadata in the conventional tree.
        if is_model_route or (
            relative.parts[0] == "model-coverage" and len(relative.parts) >= 4 and path.name != "index.mdx"
        ):
            paths.append(path)
    return paths


def _model_card_aliases(repo_root: Path) -> dict[Path, Path]:
    """Resolve shared slugs to the one source file selected by nightly navigation."""
    config = repo_root / "docs/fern/versions/nightly.yml"
    navigated: list[Path] = []
    if config.is_file():
        pending = list(yaml.safe_load(config.read_text(encoding="utf-8"))["navigation"])
        while pending:
            item = pending.pop()
            if "page" in item and "path" in item:
                navigated.append((config.parent / item["path"]).resolve())
            pending.extend(item.get("contents", []))
    by_slug: dict[str, list[Path]] = {}
    for path in _model_card_paths(repo_root):
        slug = _frontmatter_slug(path)
        if slug is not None:
            by_slug.setdefault(slug, []).append(path)
    aliases = {}
    for slug, paths in by_slug.items():
        if len(paths) == 1:
            continue
        selected = [path for path in paths if path.resolve() in navigated]
        if len(selected) != 1 or navigated.count(selected[0].resolve()) != 1:
            raise ValueError(
                f"Shared model-card slug {slug!r} must select exactly one file in nightly navigation: {paths}"
            )
        canonical = selected[0]
        aliases.update((path, canonical) for path in paths if path != canonical)
    return aliases


FernNavigationItem = TypedDict(
    "FernNavigationItem",
    {"page": str, "section": str, "path": str, "slug": str, "skip-slug": bool, "contents": list["FernNavigationItem"]},
    total=False,
)


def _fern_slug(value: str) -> str:
    return re.sub("[^a-z0-9]+", "-", value.lower().replace("&", "and")).strip("-")


def fern_page_paths(
    items: list[FernNavigationItem], *, parents: tuple[str, ...] = (), config_dir: Path | None = None
) -> dict[str, set[Path]]:
    """Collect Fern routes and the source files selected by navigation.

    Args:
        items: Parsed Fern navigation entries.
        parents: Slugs inherited from enclosing sections.
        config_dir: Directory containing the navigation YAML. Without it, only
            inferred routes are returned, with empty source-file sets.

    Returns:
        Route-to-source mappings. Multiple source files remain visible so callers
        can reject ambiguous destinations rather than borrow another file's identity.
    """
    routes: dict[str, set[Path]] = {}
    for item in items:
        source = None
        frontmatter_slug = None
        if config_dir is not None and isinstance(item.get("path"), str):
            source = (config_dir / item["path"]).resolve()
            frontmatter_slug = _frontmatter_slug(source)
        if "section" in item:
            slug = str(item.get("slug", _fern_slug(str(item["section"]))))
            if frontmatter_slug and source is not None:
                routes.setdefault("/" + frontmatter_slug, set()).add(source)
            child_parents = parents if item.get("skip-slug") else (*parents, slug)
            for route, paths in fern_page_paths(
                item.get("contents", []), parents=child_parents, config_dir=config_dir
            ).items():
                routes.setdefault(route, set()).update(paths)
        elif "page" in item:
            slug = str(item.get("slug", _fern_slug(str(item["page"]))))
            route = "/" + frontmatter_slug if frontmatter_slug else "/" + "/".join((*parents, slug))
            paths = routes.setdefault(route, set())
            if source is not None:
                paths.add(source)
    return routes
