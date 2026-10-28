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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Installs the same compiler used by the docs check once, then exercises its CLI.
pytestmark = pytest.mark.timeout(70)


@pytest.fixture(scope="module")
def mdx_dependencies(tmp_path_factory) -> Path:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("MDX CLI tests require Node.js and npm; the docs CI job provides both")
    prefix = tmp_path_factory.mktemp("mdx-dependencies")
    (prefix / "package.json").write_text('{"private":true,"type":"module"}')
    subprocess.run(
        [
            "npm",
            "--prefix",
            str(prefix),
            "install",
            "--silent",
            "@mdx-js/mdx@3",
            "remark-frontmatter@5",
            "remark-math@6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return prefix


def _validate(root: Path, dependencies: Path) -> subprocess.CompletedProcess[str]:
    tool = Path(__file__).resolve().parents[5] / "tools/docs/checks/mdx.mjs"
    return subprocess.run(
        ["node", str(tool), str(root / "docs")],
        cwd=root,
        env={**os.environ, "MDX_LINT_PREFIX": str(dependencies), "MDX_LINT_REPO_ROOT": str(root)},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("valid", [True, False])
def test_mdx_cli_accepts_valid_pages_and_reports_syntax_errors(tmp_path: Path, mdx_dependencies: Path, valid: bool):
    page = tmp_path / "docs/page.mdx"
    page.parent.mkdir()
    page.write_text("---\ntitle: Example\n---\n# Page\n\n$x^2$\n" if valid else "<Broken>\n")

    result = _validate(tmp_path, mdx_dependencies)

    if valid:
        assert result.returncode == 0, result.stderr
        assert "Validated 1 MDX files" in result.stdout
    else:
        assert result.returncode == 1
        assert "docs/page.mdx:" in result.stderr
        assert "Found 1 MDX syntax error" in result.stderr


def test_mdx_cli_excludes_generated_library_reference(tmp_path: Path, mdx_dependencies: Path):
    generated = tmp_path / "docs/fern/product-docs/reference.mdx"
    generated.parent.mkdir(parents=True)
    generated.write_text("<Broken>\n")
    (tmp_path / "docs/page.mdx").write_text("# Valid page\n")

    result = _validate(tmp_path, mdx_dependencies)

    assert result.returncode == 0, result.stderr
    assert "Validated 1 MDX files" in result.stdout
