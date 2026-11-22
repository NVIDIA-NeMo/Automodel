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


import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def docs_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    source = Path(__file__).resolve().parents[4]
    tool = root / "tools/docs/check_docs.py"
    tool.parent.mkdir(parents=True)
    shutil.copyfile(source / "tools/docs/check_docs.py", tool)
    config = root / "docs/fern/fern.config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"version": "5.999.0"}', encoding="utf-8")
    (root / "docs/index.mdx").write_text(
        "User content\n{/* BEGIN GENERATED LATEST MODEL SUPPORT */}\n"
        "{/* END GENERATED LATEST MODEL SUPPORT */}\nMore user content\n",
        encoding="utf-8",
    )
    generator = root / "tools/docs/build/coverage_tables.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "tools/docs/catalog.py", root / "tools/docs/catalog.py")
    shutil.copyfile(source / "tools/docs/cards.py", root / "tools/docs/cards.py")
    generator.write_text(
        "import os, runpy, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[3]\n"
        "if '--clean' in sys.argv:\n"
        f"    helpers = runpy.run_path({str(source / 'tools/docs/build/coverage_tables.py')!r})\n"
        "    helpers['_strip_generated_tables'](root)\n"
        "else:\n"
        "    card = root / 'docs/index.mdx'\n"
        "    end = '{/* END GENERATED LATEST MODEL SUPPORT */}'\n"
        "    card.write_text(card.read_text().replace(end, 'generated content\\n' + end))\n"
        "    sys.exit(int(os.environ.get('DOCS_GENERATION_EXIT', '0')))\n",
        encoding="utf-8",
    )
    (root / "tools/docs/checks").mkdir()
    (root / "tools/docs/checks/sources.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "assert 'generated content' not in Path('docs/index.mdx').read_text()\n"
        "assert os.environ.get('DOCS_SOURCE_FAILURE') != '1'\n",
        encoding="utf-8",
    )
    bin_dir = root / "bin"
    bin_dir.mkdir()
    npx = bin_dir / "npx"
    npx.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path('validator.json').write_text(json.dumps({\n"
        "    'args': sys.argv[1:], 'no_redirect': os.environ.get('FERN_NO_VERSION_REDIRECTION')}))\n"
        "card = Path('docs/index.mdx')\n"
        "assert 'generated content' in card.read_text(), 'Tables must be generated before validation'\n"
        "end = '{/* END GENERATED LATEST MODEL SUPPORT */}'\n"
        "if os.environ.get('DOCS_CLEANUP_FAILURE') == '1':\n"
        "    card.write_text(card.read_text().replace(end, ''))\n"
        "sys.exit(int(os.environ.get('DOCS_VALIDATION_EXIT', '0')))\n",
        encoding="utf-8",
    )
    npx.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _check_docs(root: Path, *arguments: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "tools/docs/check_docs.py"), *arguments],
        # Invoke outside the checkout to exercise root discovery.
        cwd=root.parent,
        env={**os.environ, "PATH": f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}", **overrides},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("validation_exit", [0, 7])
def test_docs_check_cleans_generated_tables_and_preserves_validator_status(docs_checkout: Path, validation_exit: int):
    assert not (docs_checkout / "tests").exists(), "Validation must work without a pytest tree"
    card = docs_checkout / "docs/index.mdx"
    original = card.read_bytes()

    result = _check_docs(docs_checkout, DOCS_VALIDATION_EXIT=str(validation_exit))

    assert result.returncode == validation_exit, result.stdout + result.stderr
    assert card.read_bytes() == original
    invocation = json.loads((docs_checkout / "validator.json").read_text())
    assert "--package=fern-api@5.999.0" in invocation["args"]
    assert invocation["no_redirect"] == "true"


@pytest.mark.parametrize("validation_exit", [0, 7])
def test_publish_check_retains_tables_only_after_success(docs_checkout: Path, validation_exit: int):
    card = docs_checkout / "docs/index.mdx"
    original = card.read_bytes()

    result = _check_docs(docs_checkout, "--keep-generated", DOCS_VALIDATION_EXIT=str(validation_exit))

    assert result.returncode == validation_exit, result.stdout + result.stderr
    if validation_exit == 0:
        assert "generated content" in card.read_text()
    else:
        assert card.read_bytes() == original


@pytest.mark.parametrize("arguments", [(), ("--keep-generated",)])
def test_generation_failure_stops_validation_and_cleans_partial_tables(docs_checkout: Path, arguments: tuple[str, ...]):
    card = docs_checkout / "docs/index.mdx"
    original = card.read_bytes()

    result = _check_docs(docs_checkout, *arguments, DOCS_GENERATION_EXIT="8")

    assert result.returncode == 8, result.stdout + result.stderr
    assert card.read_bytes() == original
    assert not (docs_checkout / "validator.json").exists()


def test_source_failure_stops_before_generation_or_cleanup(docs_checkout: Path):
    card = docs_checkout / "docs/index.mdx"
    card.write_text(card.read_text() + "Uncommitted user edit\n")
    original = card.read_bytes()

    result = _check_docs(docs_checkout, DOCS_SOURCE_FAILURE="1")

    assert result.returncode == 1, result.stdout + result.stderr
    assert card.read_bytes() == original
    assert not (docs_checkout / "validator.json").exists()


@pytest.mark.parametrize(("validation_exit", "expected_exit"), [(0, 1), (7, 7)])
def test_cleanup_failure_is_reported_without_masking_validation_failure(
    docs_checkout: Path, validation_exit: int, expected_exit: int
):
    result = _check_docs(docs_checkout, DOCS_VALIDATION_EXIT=str(validation_exit), DOCS_CLEANUP_FAILURE="1")

    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert "Expected one ordered marker pair" in result.stderr


@pytest.mark.parametrize("arguments", [(), ("--keep-generated",)])
def test_missing_validator_reports_failure_and_still_cleans_tables(docs_checkout: Path, arguments: tuple[str, ...]):
    card = docs_checkout / "docs/index.mdx"
    original = card.read_bytes()
    (docs_checkout / "bin/npx").unlink()

    result = _check_docs(docs_checkout, *arguments, PATH=str(docs_checkout / "bin"))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Docs check could not run" in result.stderr
    assert card.read_bytes() == original


def test_publish_check_cleans_tables_if_whitespace_check_fails(docs_checkout: Path):
    card = docs_checkout / "docs/index.mdx"
    original = card.read_bytes()
    git = docs_checkout / "bin/git"
    git.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(3)\n", encoding="utf-8")
    git.chmod(0o755)

    result = _check_docs(docs_checkout, "--keep-generated")

    assert result.returncode == 3, result.stdout + result.stderr
    assert card.read_bytes() == original


@pytest.mark.parametrize(
    ("workflow_name", "job"), [("fern-docs-ci.yml", "check"), ("publish-fern-docs.yml", "publish")]
)
def test_docs_ci_uses_the_shared_check(workflow_name: str, job: str):
    root = Path(__file__).resolve().parents[4]
    workflow = yaml.safe_load((root / ".github/workflows" / workflow_name).read_text())
    checks = [step for step in workflow["jobs"][job]["steps"] if step["name"] == "Check documentation"]

    assert len(checks) == 1
    expected = "uv run --script tools/docs/check_docs.py"
    if job == "publish":
        expected += " --keep-generated"
        assert all(
            "sync_model_coverage_tables.py" not in step.get("run", "") for step in workflow["jobs"][job]["steps"]
        )
    assert checks[0]["run"] == expected
    assert "env" not in checks[0], "Validation must run without publish credentials"


@pytest.mark.parametrize(("tool_exit", "make_exit"), [(0, 0), (7, 2)])
@pytest.mark.parametrize("target", ["docs-check", "docs-mdx-check"])
def test_make_docs_check_forwards_to_the_shared_tool(docs_checkout: Path, tool_exit: int, make_exit: int, target: str):
    root = Path(__file__).resolve().parents[4]
    shutil.copyfile(root / "docs/fern/Makefile", docs_checkout / "docs/fern/Makefile")
    uv = docs_checkout / "bin/uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[-1]).resolve().parents[2]\n"
        "(root / 'alias.json').write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['DOCS_TOOL_EXIT']))\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    result = subprocess.run(
        ["make", "--no-print-directory", "-C", str(docs_checkout / "docs/fern"), target],
        cwd=docs_checkout.parent,
        env={
            **os.environ,
            "PATH": f"{docs_checkout / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "DOCS_TOOL_EXIT": str(tool_exit),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == make_exit, result.stdout + result.stderr
    assert json.loads((docs_checkout / "alias.json").read_text()) == [
        "run",
        "--script",
        str(docs_checkout / "tools/docs/check_docs.py"),
    ]
