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

from pathlib import Path

import yaml

from tools.docs.catalog import MODEL_TYPE_ROUTE_CATEGORIES


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
