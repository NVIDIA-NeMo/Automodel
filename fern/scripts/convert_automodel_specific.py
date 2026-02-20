#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Automodel-specific syntax for Fern MDX compatibility.

Handles: {doc} roles (internal doc links), escaping {variable} in code blocks.
"""

import argparse
import re
from pathlib import Path


def escape_mdx_curly_braces(content: str) -> str:
    """Escape {variable} in code blocks so MDX doesn't parse as JSX."""
    return content.replace("{overrides}", "\\{overrides\\}")


def resolve_doc_path(path: str, file_dir: Path | None) -> str:
    """Resolve doc path to Fern URL. Paths without / are relative to current file's dir."""
    path = path.replace("../", "").replace(".md", "").replace(".mdx", "").strip()
    if "/" not in path and file_dir:
        rel_parts = file_dir.parts
        path = "/".join(rel_parts) + "/" + path
    if not path.startswith("/"):
        path = "/" + path
    return path


def convert_doc_roles(content: str, filepath: Path | None = None) -> str:
    """Convert {doc}`display <path>` and {doc}`path` to internal links."""
    file_dir = None
    if filepath:
        try:
            pages_idx = filepath.parts.index("pages")
            file_dir = Path(*filepath.parts[pages_idx + 1 : filepath.parts.index(filepath.name)])
        except (ValueError, IndexError):
            pass

    def replace_doc_with_path(match: re.Match[str]) -> str:
        display = match.group(1).strip()
        path = match.group(2).strip()
        clean = resolve_doc_path(path, file_dir)
        return f"[{display}]({clean})"

    def replace_doc_path_only(match: re.Match[str]) -> str:
        path = match.group(1).strip()
        clean = resolve_doc_path(path, file_dir)
        display = path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return f"[{display}]({clean})"

    content = re.sub(r"\{doc\}`([^`]+?)\s*<([^>]+)>`", replace_doc_with_path, content)
    content = re.sub(r"\{doc\}`([^`]+)`", replace_doc_path_only, content)
    return content


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    content = escape_mdx_curly_braces(content)
    content = convert_doc_roles(content, filepath)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Automodel-specific syntax for Fern MDX"
    )
    parser.add_argument(
        "pages_dir",
        type=Path,
        help="Path to pages directory (e.g. fern/v0.1.0/pages)",
    )
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if convert_file(mdx_file):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Converted: {mdx_file.relative_to(pages_dir)}")

    print(f"\nConverted {len(changed)} files")


if __name__ == "__main__":
    main()
