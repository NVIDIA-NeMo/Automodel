#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Hub dataset cards, recipe links, navigation, and direct example coverage.

This check is offline: it does not import training code, download datasets, or
certify upstream facts. Authors verify those against the linked Hub sources.
"""

import argparse
import json
import re
from pathlib import Path

import yaml

SECTIONS = ("Task", "Example Record", "Schema and Splits", "Use with NeMo AutoModel", "Related Resources")
DATASET_KEYS = {"dataset_name", "path_or_dataset", "path_or_dataset_id", "train_data_path", "schema_dataset"}
HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*\Z")
REPO_LINK = re.compile(r"https://github\.com/NVIDIA-NeMo/Automodel/blob/[^/]+/([^\s)#]+)")


def hub_ids(value):
    """Read explicit dataset IDs from parsed YAML, excluding local file paths."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in DATASET_KEYS and isinstance(item, str) and HUB_ID.fullmatch(item):
                if not item.endswith((".json", ".jsonl", ".parquet", ".bin", ".yaml", ".yml")):
                    found.add(item)
            found.update(hub_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(hub_ids(item))
    elif isinstance(value, str) and value.startswith("hf://"):
        candidate = "/".join(value[5:].split("/")[:2])
        if HUB_ID.fullmatch(candidate):
            found.add(candidate)
    return found


def card_errors(text, entry):
    """Validate a card independently of its registration and local source files."""
    errors = []
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not match:
        return ["missing YAML frontmatter"]
    try:
        meta = yaml.safe_load(match[1])
    except yaml.YAMLError as exc:
        return [f"invalid frontmatter: {exc}"]
    if not isinstance(meta, dict):
        return ["frontmatter must be a mapping"]
    dataset_id = entry["id"]
    if meta.get("title") != dataset_id:
        errors.append("title must equal the canonical Hub dataset ID")
    if meta.get("slug") != f"dataset-coverage/{dataset_id}":
        errors.append("slug must be dataset-coverage/<Hub dataset ID>")
    if not isinstance(meta.get("description"), str) or not meta["description"].strip():
        errors.append("description must be nonempty")

    body = text[match.end() :]
    prose = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", body, flags=re.M | re.S)
    headings = re.findall(r"^## (.+)$", prose, re.M)
    if tuple(headings) != SECTIONS:
        errors.append("H2 sections must be: " + " -> ".join(SECTIONS))
        return errors
    areas = dict(zip(SECTIONS, re.split(r"^## .+$", body, flags=re.M)[1:]))
    for heading, content in areas.items():
        if not content.strip():
            errors.append(f"empty section: {heading}")
    for field in ("Input", "Target", "Modality", "License Metadata"):
        if not re.search(r"^\| " + field + r" \| \S", areas["Task"], re.M):
            errors.append(f"Task must describe {field}")
    if entry["task"] not in areas["Task"]:
        errors.append("Task must name the catalog task")

    example = areas["Example Record"]
    if "synthetic example" not in example.lower():
        errors.append("example must be explicitly labeled synthetic")
    blocks = re.findall(r"^```json\n(.*?)\n```", example, re.M | re.S)
    if len(blocks) != 1:
        errors.append("Example Record must contain one JSON block")
    else:
        try:
            record = json.loads(blocks[0])
            if not isinstance(record, dict) or not record:
                errors.append("example must be a nonempty JSON object")
            else:
                fields = re.findall(r"^\| `([^`]+)` \|", areas["Schema and Splits"].split("| Configuration |")[0], re.M)
                if not fields:
                    errors.append("missing field schema table")
                for field in fields:
                    if field not in record:
                        errors.append(f"schema field {field!r} is absent from the example")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid example JSON: {exc.msg}")
    if "| Configuration |" not in areas["Schema and Splits"]:
        errors.append("missing upstream split/source table")

    hub_url = f"https://huggingface.co/datasets/{dataset_id}"
    if f"]({hub_url})" not in text:
        errors.append("missing canonical Hub link")
    revision = entry["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or f"{hub_url}/blob/{revision}/README.md" not in text:
        errors.append("missing revision-pinned upstream card")
    linked_recipes = {p for p in REPO_LINK.findall(areas["Use with NeMo AutoModel"]) if p.endswith((".yaml", ".yml"))}
    if not entry["recipes"] or linked_recipes != set(entry["recipes"]):
        errors.append("recipe links must match the catalog recipes")
    return errors


def validate(root):
    """Return errors across the catalog, cards, navigation, and example YAMLs."""
    errors = []
    directory = root / "docs/dataset-coverage"
    entries = json.loads((directory / "catalog.json").read_text())
    if not isinstance(entries, list) or not entries:
        return ["catalog must be a nonempty list"]
    registered = set()
    known_ids = set()
    navigation = (root / "docs/fern/versions/nightly.yml").read_text()
    index = (directory / "index.mdx").read_text()
    for entry in entries:
        dataset_id = entry["id"]
        if not HUB_ID.fullmatch(dataset_id):
            errors.append(f"invalid Hub dataset ID: {dataset_id}")
            continue
        for name in [dataset_id, *entry.get("aliases", [])]:
            if name in known_ids:
                errors.append(f"duplicate dataset ID or alias: {name}")
            known_ids.add(name)
        card = entry["card"]
        registered.add(card)
        if card not in {f"docs/dataset-coverage/{dataset_id}.mdx", f"docs/dataset-coverage/{dataset_id}.md"}:
            errors.append(f"{card}: path must preserve the canonical Hub ID")
            continue
        path = root / card
        if not path.is_file():
            errors.append(f"missing card: {card}")
            continue
        text = path.read_text()
        errors.extend(f"{card}: {error}" for error in card_errors(text, entry))
        paths = (
            set(REPO_LINK.findall(text)) | set(entry["recipes"]) | {entry["source"]} | set(entry.get("evidence", []))
        )
        for relative in sorted(paths):
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                errors.append(f"{card}: missing or invalid repository path: {relative}")
        nav_path = "../../" + card.removeprefix("docs/")
        if f"path: {nav_path}\n" not in navigation:
            errors.append(f"{card}: missing nightly navigation entry")
        if f"](/dataset-coverage/{dataset_id})" not in index:
            errors.append(f"{card}: missing catalog index link")
    actual = {
        str(p.relative_to(root))
        for p in directory.rglob("*")
        if p.suffix in {".md", ".mdx"} and p != directory / "index.mdx"
    }
    errors.extend(f"unregistered dataset card: {p}" for p in sorted(actual - registered))
    for recipe in sorted((root / "examples").rglob("*")):
        if recipe.suffix not in {".yaml", ".yml"}:
            continue
        try:
            config = yaml.safe_load(recipe.read_text())
        except yaml.YAMLError as exc:
            errors.append(f"{recipe.relative_to(root)}: invalid YAML: {exc}")
            continue
        for dataset_id in sorted(hub_ids(config) - known_ids):
            errors.append(f"{recipe.relative_to(root)}: no card for Hub dataset {dataset_id}")
    return errors


def main():
    """Run offline validation and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        errors = validate(args.repo_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [f"invalid dataset catalog: {exc}"]
    if errors:
        for error in errors:
            print(error)
        return 1
    count = len(json.loads((args.repo_root / "docs/dataset-coverage/catalog.json").read_text()))
    print(f"Validated {count} Hub dataset cards and direct example coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
