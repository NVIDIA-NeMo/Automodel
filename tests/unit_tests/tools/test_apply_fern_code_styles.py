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

from pathlib import Path

import pytest

from tools.apply_fern_code_styles import (
    DARK_CODE_BLOCK_SELECTOR,
    DARK_INLINE_CODE_SELECTOR,
    GENERATED_STYLE,
    LIGHT_CODE_BLOCK_SELECTOR,
    LIGHT_INLINE_CODE_SELECTOR,
    apply_fern_code_styles,
)


def test_apply_styles_is_idempotent_and_clean_is_lossless(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "guide.mdx"
    page.parent.mkdir()
    original = "---\ntitle: Guide\n---\n\n```bash\necho hello\n```\n"
    page.write_text(original, encoding="utf-8")

    assert apply_fern_code_styles(tmp_path) == [page]
    assert page.read_text(encoding="utf-8") == original + GENERATED_STYLE
    assert LIGHT_CODE_BLOCK_SELECTOR in GENERATED_STYLE
    assert LIGHT_INLINE_CODE_SELECTOR in GENERATED_STYLE
    assert DARK_CODE_BLOCK_SELECTOR in GENERATED_STYLE
    assert DARK_INLINE_CODE_SELECTOR in GENERATED_STYLE
    assert apply_fern_code_styles(tmp_path) == []

    assert apply_fern_code_styles(tmp_path, clean=True) == [page]
    assert page.read_text(encoding="utf-8") == original


def test_apply_styles_preserves_an_equivalent_page_style(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "guide.mdx"
    page.parent.mkdir()
    original = """---
title: Guide
---

<style>{`
  .light .fern-code-block,
  .light .fern-prose code:not(.code-block) {
    background-color: #f7f7f7 !important;
  }

  .dark .fern-code-block,
  .dark .fern-prose code:not(.code-block) {
    background-color: #1f1f1f !important;
  }
`}</style>
"""
    page.write_text(original, encoding="utf-8")

    assert apply_fern_code_styles(tmp_path) == []
    assert page.read_text(encoding="utf-8") == original


def test_apply_styles_only_updates_mdx_under_docs(tmp_path: Path) -> None:
    docs_page = tmp_path / "docs" / "nested" / "guide.mdx"
    outside_page = tmp_path / "examples" / "guide.mdx"
    markdown_page = tmp_path / "docs" / "guide.md"
    for path in (docs_page, outside_page, markdown_page):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n", encoding="utf-8")

    assert apply_fern_code_styles(tmp_path) == [docs_page]
    assert docs_page.read_text(encoding="utf-8") == "content\n" + GENERATED_STYLE
    assert outside_page.read_text(encoding="utf-8") == "content\n"
    assert markdown_page.read_text(encoding="utf-8") == "content\n"


def test_apply_styles_rejects_symbolic_links(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    target = tmp_path / "target.mdx"
    target.write_text("content\n", encoding="utf-8")
    (docs_root / "linked.mdx").symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        apply_fern_code_styles(tmp_path)

    assert target.read_text(encoding="utf-8") == "content\n"
