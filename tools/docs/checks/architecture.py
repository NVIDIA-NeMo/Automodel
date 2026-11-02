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

"""Guard against the gemma4-style regression where a new model architecture
lands in ``MODEL_ARCH_MAPPING`` without any corresponding page under
``docs/model-coverage/``.
"""

from pathlib import Path

from tools.docs.build.coverage_tables import (
    _parse_registry_entries,
)
from tools.docs.cards import _model_card_paths
from tools.docs.catalog import (
    _DOC_ARCH_ALIASES,
)


def _model_card_documents(repo_root: Path) -> list[tuple[Path, str]]:
    """Return rendered model-card documents, excluding section landing pages."""
    return [(path, path.read_text(encoding="utf-8")) for path in _model_card_paths(repo_root)]


def _without_generated_registry_table(content: str) -> str:
    start_marker = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"
    end_marker = "{/* END GENERATED MODEL ARCHITECTURES */}"
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start == -1 or end == -1:
        return content
    return content[:start] + content[end + len(end_marker) :]


def validate_architecture_reference_tables(repo_root: Path) -> None:
    """Validate the shared architecture section on every rendered model card.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: A card is absent or violates the architecture section contract.
    """
    documents = _model_card_documents(repo_root)
    if not documents:
        raise ValueError("No rendered model cards found under docs/model-coverage/")
    invalid = []
    for path, content in documents:
        reasons = []
        if "## Model Reference" not in content:
            reasons.append("missing '## Model Reference'")
        if "### Model Architecture" not in content:
            reasons.append("missing '### Model Architecture'")
        else:
            architecture_section = content.split("### Model Architecture", maxsplit=1)[1]
            architecture_section = architecture_section.split("\n##", maxsplit=1)[0]
            if "| Property | Value |" not in architecture_section:
                reasons.append("missing architecture Property/Value table")
        if "### Model Summary" in content:
            reasons.append("uses legacy '### Model Summary' heading")
        if "### Available Checkpoints" in content:
            reasons.append("uses legacy '### Available Checkpoints' heading")
        if not content.isascii():
            reasons.append("contains non-ASCII characters")
        if reasons:
            invalid.append(f"  - {path.relative_to(repo_root)}: {', '.join(reasons)}")
    if invalid:
        raise ValueError("Model cards that violate the shared page contract:\n" + "\n".join(invalid))


def validate_registered_architectures(repo_root: Path) -> None:
    """Every architecture in ``MODEL_ARCH_MAPPING`` must be mentioned in at
    least one ``docs/model-coverage/*.mdx`` file, either by its own name or by
    a mapped alias in ``_DOC_ARCH_ALIASES``.

    This guards against the regression where a new arch (e.g. gemma4) is
    registered but never gets a corresponding model card in the docs.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: An architecture has no corresponding model card.
    """
    registry_entries = _parse_registry_entries(
        (repo_root / "nemo_automodel/_transformers/registry.py").read_text(encoding="utf-8")
    )
    docs_dir = repo_root / "docs" / "model-coverage"
    if not docs_dir.is_dir():
        raise ValueError(f"docs/model-coverage/ not found at {docs_dir}")
    md_contents = [
        _without_generated_registry_table(p.read_text(encoding="utf-8")) for p in _model_card_paths(repo_root)
    ]
    if not md_contents:
        raise ValueError("No .mdx files found under docs/model-coverage/")
    missing = []
    for arch_name, _, _ in registry_entries:
        needle = _DOC_ARCH_ALIASES.get(arch_name, arch_name)
        if not any((needle in content for content in md_contents)):
            missing.append((arch_name, needle))
    if missing:
        details = "\n".join((f"  - {arch} (looked for {repr(needle)})" for (arch, needle) in missing))
        raise ValueError(
            f"The following registered architectures have no model card in docs/model-coverage/:\n{details}\n\nFix by either:\n  1. Adding a new .mdx file under docs/model-coverage/ (preferred for new architectures — e.g., docs/model-coverage/vlm/google/gemma4.mdx), or\n  2. Updating an existing .mdx file to mention the arch name, or\n  3. Adding an entry to _DOC_ARCH_ALIASES in tools/docs/catalog.py with a comment explaining the mismatch."
        )


def validate_architecture_aliases(repo_root: Path) -> None:
    """Every value in ``_DOC_ARCH_ALIASES`` must literally appear in some
    ``docs/model-coverage/*.mdx`` file.

    Prevents aliases from pointing at strings that never existed or got
    removed — if the target string is missing, the aliased arch is silently
    undocumented and the doc-coverage check becomes a no-op for that entry.

    Args:
        repo_root: AutoModel repository root.

    Raises:
        ValueError: An alias target is absent from the model cards.
    """
    md_contents = [
        _without_generated_registry_table(p.read_text(encoding="utf-8")) for p in _model_card_paths(repo_root)
    ]
    bad = []
    for arch, needle in _DOC_ARCH_ALIASES.items():
        if not any((needle in content for content in md_contents)):
            bad.append((arch, needle))
    if bad:
        raise ValueError(
            "_DOC_ARCH_ALIASES entries pointing at strings absent from the docs:\n"
            + "\n".join((f"  - {arch} -> {needle!r}" for (arch, needle) in bad))
        )
