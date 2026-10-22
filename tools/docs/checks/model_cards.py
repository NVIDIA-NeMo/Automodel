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

"""Validate model-card content, checkpoint identity, and compatibility copies."""

import re
import subprocess
from pathlib import Path

import yaml

from tools.docs.build.coverage_tables import (
    _iter_pretrained_model_ids,
    _load_model_doc_catalog,
    _read_documented_recipe,
)
from tools.docs.catalog import (
    MODEL_CARD_PATH_ALIASES,
)


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
        (
            path
            for path in coverage_root.rglob("*.mdx")
            if path.name != "index.mdx" and len(path.relative_to(coverage_root).parts) == 3
        )
    )


def _validate_model_card_recipe_identity(repo_root: Path, card: Path) -> None:
    document = card.read_text(encoding="utf-8")
    recipe_ids: set[str] = set()
    for match in re.finditer(
        "https://github\\.com/NVIDIA-NeMo/Automodel/blob/(?:main|[0-9a-f]{40})/(examples/[A-Za-z0-9_./-]+\\.yaml)",
        document,
    ):
        recipe = yaml.safe_load(_read_documented_recipe(repo_root, match[1], match[0]))
        if not isinstance(recipe, dict):
            raise ValueError(f"{match[1]}: recipe must be a YAML mapping")
        recipe_ids.update(_iter_pretrained_model_ids(recipe.get("model", {})))
    metadata = yaml.safe_load(document.split("---", 2)[1])
    title = metadata["title"]
    if recipe_ids:
        names = {model_id.split("/", 1)[1] for model_id in recipe_ids}
        if title not in names:
            raise ValueError(f"{card}: title {title!r} must match a linked recipe model ID: {sorted(recipe_ids)}")
    else:
        hf_ids = re.findall("https://huggingface\\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", document)
        if not hf_ids:
            raise ValueError(f"{card}: reference card must link its Hugging Face checkpoint")
        if title != hf_ids[0].split("/", 1)[1]:
            raise ValueError(f"{card}: title must match the primary HF checkpoint")
        recipe_ids = {hf_ids[0]}
    filename = re.sub("[^a-z0-9]+", "-", title.casefold()).strip("-") + ".mdx"
    if card.name != filename:
        raise ValueError(f"{card}: filename must be {filename!r} for checkpoint {title!r}")
    matching_ids = {model_id for model_id in recipe_ids if model_id.split("/", 1)[1] == title}
    if not any((f"https://huggingface.co/{model_id}" in document for model_id in matching_ids)):
        raise ValueError(f"{card}: link the exact Hugging Face checkpoint from the recipe")


def _is_local_checkpoint_override(value: str, document: str) -> bool:
    """Recognize literal local paths and paths exported earlier in the example."""
    value = value.strip("\"'")
    variable = re.fullmatch("\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}", value)
    if variable:
        assignment = re.search(f"^export {variable.group(1)}=([^\\r\\n]+)$", document, re.MULTILINE)
        if assignment is None:
            return False
        value = assignment.group(1).strip().strip("\"'")
    return Path(value).is_absolute()


def _model_card_command_violations(repo_root: Path) -> list[str]:
    offenders: list[str] = []
    for path in _model_card_paths(repo_root):
        document = path.read_text(encoding="utf-8")
        lines = document.splitlines()
        documented_model_ids = set(
            re.findall("https://huggingface\\.co/([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)", document)
        )
        for index, line in enumerate(lines):
            if re.search("(?:^|\\s)(?:uv run )?(?:automodel|torchrun)\\s", line) is None:
                continue
            command_lines = [line]
            while command_lines[-1].rstrip().endswith("\\") and index + len(command_lines) < len(lines):
                command_lines.append(lines[index + len(command_lines)])
            command = "\n".join(command_lines)
            if "<recipe-path>" in command:
                offenders.append(f"{path.relative_to(repo_root)}: contains a recipe placeholder")
                continue
            checkpoint_override = re.search("--model\\.pretrained_model_name_or_path\\s+(\\S+)", command)
            if checkpoint_override and (not _is_local_checkpoint_override(checkpoint_override.group(1), document)):
                offenders.append(f"{path.relative_to(repo_root)}: substitutes a checkpoint into a different recipe")
            recipe_paths = re.findall("examples/[A-Za-z0-9_./-]+\\.yaml", command)
            if not recipe_paths and re.search("-c /[A-Za-z0-9_./-]+\\.yaml", command):
                recipe_paths = re.findall(
                    "Copy `(examples/[A-Za-z0-9_./-]+\\.yaml)` to a local configuration", document
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
                    f"{path.relative_to(repo_root)}: {recipe_paths[0]} uses {', '.join(sorted(recipe_model_ids))}, but the card documents {', '.join(sorted(documented_model_ids)) or 'no Hugging Face model IDs'}"
                )
    return offenders


def _validate_card_aliases(repo_root: Path) -> None:
    (_, _, catalog) = _load_model_doc_catalog(repo_root / "docs")
    source_paths = [model.source_path for model in catalog]
    for alias, canonical in MODEL_CARD_PATH_ALIASES.items():
        if not (repo_root / alias).is_file():
            raise ValueError(f"Published source URL must remain available: {alias}")
        if not (repo_root / canonical).is_file():
            raise ValueError(f"Published source URL must remain available: {canonical}")
        if (repo_root / alias).read_bytes() != (repo_root / canonical).read_bytes():
            raise ValueError(f"Compatibility card must stay synchronized with {canonical}: {alias}")
        if _frontmatter_slug(repo_root / alias) != _frontmatter_slug(repo_root / canonical):
            raise ValueError(f"Compatibility card {alias} must use the same slug as {canonical}")
        if alias in source_paths:
            raise ValueError(f"Compatibility card must be excluded from the model catalog: {alias}")
        if source_paths.count(canonical) != 1:
            raise ValueError(f"Canonical card must appear exactly once in the model catalog: {canonical}")


def _validate_changed_card_identity(repo_root: Path) -> None:
    base = subprocess.check_output(["git", "merge-base", "HEAD", "origin/main"], cwd=repo_root, text=True).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "--", "docs/model-coverage"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    changed += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "docs/model-coverage"], cwd=repo_root, text=True
    ).splitlines()
    changed += [f"docs/model-coverage/{kind}/stepfun-ai/step-5-preview-bf16.mdx" for kind in ("llm", "vlm")]
    canonical_paths = {MODEL_CARD_PATH_ALIASES.get(relative, relative) for relative in changed}
    for relative in sorted(canonical_paths):
        card = repo_root / relative
        if (
            len(card.relative_to(repo_root / "docs/model-coverage").parts) == 3
            and card.is_file()
            and (card.suffix == ".mdx")
            and (card.name != "index.mdx")
        ):
            _validate_model_card_recipe_identity(repo_root, card)


def _validate_descriptions(repo_root: Path) -> None:
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
    if offenders:
        raise ValueError("Invalid model-card descriptions:\n" + "\n".join((f"  - {item}" for item in offenders)))


def _validate_commands(repo_root: Path) -> None:
    offenders = _model_card_command_violations(repo_root)
    if offenders:
        raise ValueError("Invalid model-card commands:\n" + "\n".join((f"  - {item}" for item in offenders)))
