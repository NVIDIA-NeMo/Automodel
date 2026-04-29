# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Recipe-level doc-coverage check.

Scans every example YAML under ``examples/``, extracts the HF model IDs from
``pretrained_model_name_or_path`` fields, resolves each ID to an architecture
via ``AutoConfig.from_pretrained`` (config.json only — no GPU allocation, no
weight download), then asserts every resolved architecture is mentioned in at
least one ``docs/model-coverage/*.md`` file.

This complements the ``Doc coverage`` GitHub Actions workflow (which only
covers archs registered in ``MODEL_ARCH_MAPPING``). Many recipes fine-tune
HF-native archs (e.g.,
``Olmo2ForCausalLM``) that never get added to the registry, and still need
documentation.

Offline / CI behavior
---------------------
Arch resolution hits the Hugging Face hub for ``config.json`` (a few KB per
model — no weights, no GPU allocation). CI workers run with the hub disabled
(``HF_HUB_OFFLINE=1``) and would otherwise just accumulate lookup errors for
every recipe, so the resolution step is **opt-in**: set
``NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1`` to enable it. Without the env var the
test only sanity-checks the YAML scanner.

Developers can run the full check in two modes:
  * **Online:** ``NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1 pytest …`` — configs are
    fetched from the hub and cached under ``~/.cache/huggingface/hub``.
  * **Offline (with a warm cache):** same env var; ``AutoConfig`` re-uses the
    cached ``config.json`` files, so no network is needed. Per-model cache
    misses are skipped rather than failing the test.
"""

import os
import pathlib
from typing import Iterable

import pytest
import yaml

from tests.unit_tests._transformers._doc_coverage_data import _DOC_ARCH_ALIASES


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _iter_strings(node) -> Iterable[str]:
    """Yield every string value reachable from a nested YAML structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_strings(v)


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


_RESOLVE_ENV_VAR = "NEMO_DOC_COVERAGE_RESOLVE_ARCHS"


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


def test_recipe_archs_have_doc_coverage():
    """Every arch referenced by an example YAML recipe must appear in at least
    one ``docs/model-coverage/*.md`` file.

    Disabled by default — CI workers run with ``HF_HUB_OFFLINE=1`` and cannot
    fetch ``config.json`` from the hub, so this test would just emit a long
    list of lookup errors. Opt in by setting
    ``NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1``; the resolver uses the HF cache
    when available, so it works fully offline once warmed.
    """
    if os.environ.get(_RESOLVE_ENV_VAR, "0") != "1":
        pytest.skip(
            f"Arch resolution disabled (set {_RESOLVE_ENV_VAR}=1 to enable). "
            "CI workers run HF offline so the config.json fetch is opt-in; "
            "developers can enable it online or with a warm HF cache."
        )

    # Import NeMo registry so custom model_type → config mappings register.
    import nemo_automodel._transformers.registry  # noqa: F401

    root = _repo_root()
    examples_dir = root / "examples"
    docs_dir = root / "docs" / "model-coverage"
    assert examples_dir.is_dir(), f"examples/ not found at {examples_dir}"
    assert docs_dir.is_dir(), f"docs/model-coverage/ not found at {docs_dir}"

    model_ids = _collect_recipe_model_ids(examples_dir)
    assert model_ids, "No pretrained_model_name_or_path values found in examples/*.yaml"

    md_contents = [p.read_text(encoding="utf-8") for p in docs_dir.rglob("*.md")]

    resolved: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for mid in sorted(model_ids):
        archs = _resolve_architectures(mid)
        if archs is None:
            unresolved.append(mid)
        else:
            resolved[mid] = archs

    if not resolved:
        pytest.skip(
            "Could not resolve any recipe model ID via AutoConfig (HF hub "
            f"unreachable and cache empty). Unresolved: {len(unresolved)} IDs."
        )

    missing: list[tuple[str, str]] = []
    for mid, archs in resolved.items():
        for arch in archs:
            if not _arch_is_documented(arch, md_contents):
                missing.append((mid, arch))

    if missing:
        details = "\n".join(f"  - {arch} (from recipe using {mid})" for mid, arch in missing)
        raise AssertionError(
            "The following recipe architectures have no model card in "
            "docs/model-coverage/:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. Adding a new .md file under docs/model-coverage/, or\n"
            "  2. Updating an existing .md file to mention the arch name, or\n"
            "  3. Adding an entry to _DOC_ARCH_ALIASES in _doc_coverage_data.py "
            "with a comment explaining the mismatch."
        )


def test_yaml_recipe_scan_finds_model_ids():
    """Sanity-check the YAML scanner: examples/ must yield at least a few HF
    model IDs. Protects against regressions in the scanner itself (e.g., a
    YAML structure change that silently makes ``_collect_recipe_model_ids``
    return nothing).
    """
    root = _repo_root()
    model_ids = _collect_recipe_model_ids(root / "examples")
    assert len(model_ids) >= 10, (
        f"YAML scanner found only {len(model_ids)} model IDs in examples/ — expected many more. Scanner may be broken."
    )


# HF org → doc-dir slug, for orgs whose docs/model-coverage/ subdirectory name
# differs from a simple lowercase of the HF name. Most orgs map identity /
# lowercase; list here only the deliberate deviations (rebrands, shared-family
# pages, etc.).
_HF_ORG_TO_DOC_SLUG = {
    "CohereForAI": "cohere",
    "ibm-granite": "ibm",
    "meta-llama": "meta",
    "MiniMaxAI": "minimax",
    "OpenGVLab": "internlm",  # InternVL docs live on internlm/ alongside InternLM
    "openai-community": "openai",  # gpt2 mirror
    "zai-org": "thudm",  # zai-org (née THUDM) publishes GLM-4+
}

# HF orgs that are personal mirrors / temp upload accounts and should not be
# expected to have a dedicated doc subdirectory.
_IGNORED_HF_ORGS = {
    "akoumpa",  # personal mirror of Devstral-Small-2
}


def _expected_doc_slug(hf_org: str) -> str:
    return _HF_ORG_TO_DOC_SLUG.get(hf_org, hf_org.lower())


def test_recipe_model_ids_live_under_publishing_org_dir():
    """When a recipe's HF model ID is documented in ``docs/model-coverage/``,
    the hosting ``.md`` file must live under the HF publisher's org
    subdirectory (``docs/model-coverage/<modality>/<org_slug>/<model>.md``).

    ``org_slug`` is the HF org prefix lowercased, or mapped via
    ``_HF_ORG_TO_DOC_SLUG`` for rebrands / legitimate shared-family pages.
    This bakes the publishing-org directory convention into CI — if a future
    PR (re-)adds, say, ``lmms-lab``'s LLaVA-OneVision card under ``llava-hf/``
    instead of ``lmms-lab/``, the test fails and points at the right slug.

    Deliberately does NOT require every model ID to be mentioned by name:
    docs typically list a canonical HF ID per family (e.g., ``Llama-3.2-1B``)
    rather than enumerating every version recipe uses
    (``Llama-3.1-70B``, ``Llama-3.2-3B-Instruct``, etc.). Missing-doc cases
    are caught by ``test_recipe_archs_have_doc_coverage``.

    Purely filesystem/string work, no network — safe on offline CI workers.
    """
    root = _repo_root()
    docs_dir = root / "docs" / "model-coverage"
    assert docs_dir.is_dir()

    model_ids = _collect_recipe_model_ids(root / "examples")
    assert model_ids, "YAML scanner returned no HF model IDs"

    # Only consider model-card files at depth <modality>/<org>/<name>.md.
    # Navigation pages (top-level ``latest-models.md``, per-modality
    # ``index.md``, etc.) legitimately cross-reference HF IDs from many orgs.
    md_texts: list[tuple[pathlib.Path, str]] = []
    for md in docs_dir.rglob("*.md"):
        if len(md.relative_to(docs_dir).parts) != 3:
            continue
        md_texts.append((md, md.read_text(encoding="utf-8")))

    offenders: list[tuple[str, str, str]] = []
    for mid in sorted(model_ids):
        hf_org = mid.split("/", 1)[0]
        if hf_org in _IGNORED_HF_ORGS:
            continue
        expected_slug = _expected_doc_slug(hf_org)

        mentioning = [md for md, text in md_texts if mid in text]
        if not mentioning:
            # Undocumented; surfaced by test_recipe_archs_have_doc_coverage.
            continue

        # Pass if AT LEAST ONE mention is under the expected org slug —
        # cross-references on sibling pages (e.g., Moonlight on deepseek-v3.md
        # for shared-arch context) are fine as long as a properly-placed
        # primary card exists.
        if any(md.parent.name == expected_slug for md in mentioning):
            continue

        wrong_paths = ", ".join(str(p.relative_to(docs_dir)) for p in mentioning)
        offenders.append((mid, expected_slug, wrong_paths))

    if offenders:
        details = "\n".join(
            f"  - {mid}  (belongs under docs/model-coverage/*/{slug}/ — currently mentioned in: {paths})"
            for mid, slug, paths in offenders
        )
        raise AssertionError(
            "The following recipe HF model IDs are documented outside the "
            "org directory that matches their HF publisher:\n"
            f"{details}\n\n"
            "Fix by either:\n"
            "  1. Moving the model card to "
            "docs/model-coverage/<modality>/<org_slug>/<model>.md (preferred), "
            "or\n"
            "  2. Adding a ``<HF_ORG>: <doc_slug>`` entry to "
            "_HF_ORG_TO_DOC_SLUG with a comment explaining the rebrand / "
            "shared-page case."
        )


def test_hf_org_to_doc_slug_targets_exist():
    """Every target slug in ``_HF_ORG_TO_DOC_SLUG`` must correspond to an
    actual subdirectory under ``docs/model-coverage/``. Prevents the mapping
    from rotting if a doc directory gets renamed or removed.
    """
    docs_dir = _repo_root() / "docs" / "model-coverage"
    existing_slugs = {p.name for p in docs_dir.rglob("*") if p.is_dir()}
    missing = [(hf_org, slug) for hf_org, slug in _HF_ORG_TO_DOC_SLUG.items() if slug not in existing_slugs]
    assert not missing, "_HF_ORG_TO_DOC_SLUG points at non-existent doc directories: " + ", ".join(
        f"{hf_org}→{slug}" for hf_org, slug in missing
    )
