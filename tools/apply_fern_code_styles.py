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

"""Apply AutoModel-specific code styling to generated Fern MDX sources."""

import argparse
from pathlib import Path

START_MARKER = "{/* BEGIN GENERATED FERN CODE STYLES */}"
END_MARKER = "{/* END GENERATED FERN CODE STYLES */}"
LIGHT_CODE_BLOCK_SELECTOR = ".light .fern-code-block"
LIGHT_INLINE_CODE_SELECTOR = ".light .fern-prose code:not(.code-block)"
DARK_CODE_BLOCK_SELECTOR = ".dark .fern-code-block"
DARK_INLINE_CODE_SELECTOR = ".dark .fern-prose code:not(.code-block)"
STYLE_SELECTORS = (
    LIGHT_CODE_BLOCK_SELECTOR,
    LIGHT_INLINE_CODE_SELECTOR,
    DARK_CODE_BLOCK_SELECTOR,
    DARK_INLINE_CODE_SELECTOR,
)
GENERATED_STYLE = f"""
{START_MARKER}
<style>{{`
  {LIGHT_CODE_BLOCK_SELECTOR},
  {LIGHT_INLINE_CODE_SELECTOR} {{
    background-color: var(--nv-color-bg-alt, #f7f7f7) !important;
  }}

  {DARK_CODE_BLOCK_SELECTOR},
  {DARK_INLINE_CODE_SELECTOR} {{
    background-color: #1f1f1f !important;
  }}
`}}</style>
{END_MARKER}
"""


def _remove_generated_style(document: str, path: Path) -> str:
    marker_count = document.count(START_MARKER)
    end_marker_count = document.count(END_MARKER)
    if marker_count != end_marker_count or marker_count > 1:
        raise ValueError(f"Expected at most one complete generated style block: {path}")
    if marker_count == 0:
        return document
    if GENERATED_STYLE not in document:
        raise ValueError(f"Generated style block was modified: {path}")
    return document.replace(GENERATED_STYLE, "", 1)


def _has_equivalent_page_style(document: str) -> bool:
    return all(selector in document for selector in STYLE_SELECTORS)


def apply_fern_code_styles(repo_root: Path, clean: bool = False) -> list[Path]:
    """Apply or remove generated code styles in every Fern MDX source.

    Args:
        repo_root: AutoModel repository root containing the ``docs`` directory.
        clean: Remove generated style blocks instead of adding them.

    Returns:
        Paths whose contents changed.
    """
    docs_root = repo_root / "docs"
    if not docs_root.is_dir():
        raise FileNotFoundError(f"Fern docs directory does not exist: {docs_root}")

    paths = sorted(docs_root.rglob("*.mdx"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Refusing to update a symbolic link: {path}")

    changed_paths = []
    for path in paths:
        document = path.read_text(encoding="utf-8")
        updated = _remove_generated_style(document, path)
        if not clean and not _has_equivalent_page_style(updated):
            updated += GENERATED_STYLE
        if updated != document:
            path.write_text(updated, encoding="utf-8")
            changed_paths.append(path)
    return changed_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Apply or clean generated Fern code styles."""
    args = _parse_args()
    changed_paths = apply_fern_code_styles(args.repo_root.resolve(), clean=args.clean)
    action = "Cleaned" if args.clean else "Styled"
    print(f"{action} {len(changed_paths)} MDX files.")


if __name__ == "__main__":
    main()
