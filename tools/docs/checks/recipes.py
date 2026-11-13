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

"""Recipe model discovery and documentation coverage validation.

The default validators inspect checked-in files only. Architecture resolution
is optional and requires Transformers, the training registry, and HF configs.
"""

import logging
import pathlib

import yaml

from tools.docs.cards import _frontmatter_slug, _model_card_paths
from tools.docs.catalog import (
    _DOC_ARCH_ALIASES,
)


def _collect_recipe_model_ids(examples_dir: pathlib.Path) -> set[str]:
    """Return the set of HF-style model IDs (``org/name``) referenced in every
    example YAML via ``pretrained_model_name_or_path``.

    Filters to strings that look like HF hub IDs (contain ``/``) and excludes
    local-path markers like ``/path/to``.
    """
    model_ids: set[str] = set()
    for yaml_path in examples_dir.rglob("*.yaml"):
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue

        for k, v in _iter_path_value_pairs(data):
            if k != "pretrained_model_name_or_path":
                continue
            if not isinstance(v, str) or "/" not in v:
                continue
            if v.startswith("/") or v.startswith("./") or " " in v:
                continue
            if any(tok in v.lower() for tok in ("path_to", "your", "<", "placeholder")):
                continue
            model_ids.add(v)
    return model_ids


def _iter_path_value_pairs(node, key=None):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_path_value_pairs(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_path_value_pairs(v, key)
    else:
        yield key, node


def _resolve_architectures(model_id: str) -> list[str] | None:
    """Resolve ``model_id`` → list of architecture class names via AutoConfig.

    Fetches only ``config.json`` from the Hugging Face hub — no weights, no
    tensors allocated, safe on CPU-only workers. Re-uses any cached config at
    ``~/.cache/huggingface/hub`` so developers with a warm cache can run this
    fully offline.

    Returns ``None`` on failure (network, gated repo, missing cache, etc.).
    ``trust_remote_code`` is intentionally ``False`` to keep the test hermetic;
    custom NeMo configs are registered at import time via
    ``_register_custom_configs`` so HF-hosted configs matching our registered
    ``model_type`` values still resolve without remote code.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        return None
    archs = getattr(cfg, "architectures", None)
    return list(archs) if archs else None


def _arch_is_documented(arch: str, md_contents: list[str]) -> bool:
    needle = _DOC_ARCH_ALIASES.get(arch, arch)
    return any(needle in c for c in md_contents)


# HF org → doc-dir slug, for orgs whose docs/model-coverage/ subdirectory name
# differs from a simple lowercase of the HF name. Most orgs map identity /
# lowercase; list here only the deliberate deviations (rebrands, shared-family
# pages, etc.).
_HF_ORG_TO_DOC_SLUG = {
    "CohereForAI": "cohere",
    "diffusers": "lightricks",  # LTX-2.3 conversion shares the Lightricks family page
    "ibm-granite": "ibm",
    "meta-llama": "meta",
    "MiniMaxAI": "MiniMaxAI",  # Keep the publisher's exact case in directories and routes.
    "zai-org": "thudm",  # zai-org (née THUDM) publishes GLM-4+
}

# HF orgs that are personal mirrors / temp upload accounts and should not be
# expected to have a dedicated doc subdirectory.
_IGNORED_HF_ORGS = {
    "akoumpa",  # personal mirror of Devstral-Small-2
}


def _expected_doc_slug(hf_org: str) -> str:
    return _HF_ORG_TO_DOC_SLUG.get(hf_org, hf_org.lower())


def validate_recipe_architectures(repo_root: pathlib.Path) -> None:
    """Resolve recipe architectures from HF configs and validate their documentation.

    This optional check requires the training environment. Missing or inaccessible
    configs are skipped, matching the historical opt-in recipe check.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: A resolved architecture has no corresponding model card.
    """
    # Import NeMo registry so custom model_type → config mappings register.
    import nemo_automodel._transformers.registry  # noqa: F401

    root = repo_root
    examples_dir = root / "examples"
    docs_dir = root / "docs" / "model-coverage"
    if not examples_dir.is_dir():
        raise ValueError(f"examples/ not found at {examples_dir}")
    if not docs_dir.is_dir():
        raise ValueError(f"docs/model-coverage/ not found at {docs_dir}")

    model_ids = _collect_recipe_model_ids(examples_dir)
    if not model_ids:
        raise ValueError("No pretrained_model_name_or_path values found in examples/*.yaml")

    md_contents = [p.read_text(encoding="utf-8") for p in _model_card_paths(root)]

    resolved: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for mid in sorted(model_ids):
        archs = _resolve_architectures(mid)
        if archs is None:
            unresolved.append(mid)
        else:
            resolved[mid] = archs

    if not resolved:
        logging.getLogger(__name__).warning(
            "Could not resolve any recipe model ID via AutoConfig (HF hub "
            f"unreachable and cache empty). Unresolved: {len(unresolved)} IDs."
        )

        return

    missing: list[tuple[str, str]] = []
    for mid, archs in resolved.items():
        for arch in archs:
            if not _arch_is_documented(arch, md_contents):
                missing.append((mid, arch))

    if missing:
        details = "\n".join(f"  - {arch} (from recipe using {mid})" for mid, arch in missing)
        raise ValueError(
            "The following recipe architectures have no model card in "
            "docs/model-coverage/:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. Adding a new .mdx file under docs/model-coverage/, or\n"
            "  2. Updating an existing .mdx file to mention the arch name, or\n"
            "  3. Adding an entry to _DOC_ARCH_ALIASES in tools/docs/catalog.py "
            "with a comment explaining the mismatch."
        )


def validate_recipe_publishers(repo_root: pathlib.Path) -> None:
    """Validate that documented recipe checkpoints have cards under their publishers.

    Publisher aliases allow rebrands and shared family pages. Cross-references
    elsewhere are allowed when at least one card uses the expected provider slug.
    This check only reads checked-in files and makes no network requests.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: A documented checkpoint has no card under its publisher.
    """
    root = repo_root
    recipe_model_ids = _collect_recipe_model_ids(root / "examples")
    docs_dir = root / "docs" / "model-coverage"
    if not docs_dir.is_dir():
        raise ValueError("Missing docs input")

    if not recipe_model_ids:
        raise ValueError("YAML scanner returned no HF model IDs")

    # Slugs identify publishers even when shared source URLs use another directory.
    md_texts = [(md, md.read_text(encoding="utf-8")) for md in _model_card_paths(root)]

    offenders: list[tuple[str, str, str]] = []
    for mid in sorted(recipe_model_ids):
        hf_org = mid.split("/", 1)[0]
        if hf_org in _IGNORED_HF_ORGS:
            continue
        expected_slug = _expected_doc_slug(hf_org)

        mentioning = [md for md, text in md_texts if mid in text]
        if not mentioning:
            # Undocumented; surfaced by optional architecture resolution.
            continue

        # Cross-references are allowed when a primary card has the expected publisher route.
        if any(
            (slug := _frontmatter_slug(md)) is not None
            and len(slug.split("/")) == 4
            and slug.split("/")[2] == expected_slug
            for md in mentioning
        ):
            continue

        wrong_paths = ", ".join(str(p.relative_to(root)) for p in mentioning)
        offenders.append((mid, expected_slug, wrong_paths))

    if offenders:
        details = "\n".join(
            f"  - {mid}  (expected provider slug {slug!r}; currently mentioned in: {paths})"
            for mid, slug, paths in offenders
        )
        raise ValueError(
            "The following recipe HF model IDs are documented outside their publisher routes:\n"
            f"{details}\n\n"
            "Update the card's frontmatter slug and navigation provider section, keeping its source path stable. "
            "For a rebrand or shared-page case, add a documented publisher alias to _HF_ORG_TO_DOC_SLUG."
        )


def validate_recipe_publisher_aliases(repo_root: pathlib.Path) -> None:
    """Validate that publisher alias targets exist in the documentation tree.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: An alias points to a missing publisher directory.
    """
    docs_dir = repo_root / "docs" / "model-coverage"
    existing_slugs = {p.name for p in docs_dir.rglob("*") if p.is_dir()}
    missing = [(hf_org, slug) for hf_org, slug in _HF_ORG_TO_DOC_SLUG.items() if slug not in existing_slugs]
    if missing:
        raise ValueError(
            "_HF_ORG_TO_DOC_SLUG points at non-existent doc directories: "
            + ", ".join(f"{hf_org}→{slug}" for hf_org, slug in missing)
        )
