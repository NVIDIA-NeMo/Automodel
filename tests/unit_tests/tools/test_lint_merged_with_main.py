# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Lint pull requests merged into main through the CLI, against a local origin with pull request refs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "lint_merged_with_main.py"
_CLEAN = "Merged into main: no new ruff findings"
_BASE_SOURCE = """\
def _unwrap(model):
    return model


def load(model):
    return _unwrap(model)


def save(model):
    return model
"""


class TestLintMergedWithMain(unittest.TestCase):
    def setUp(self) -> None:
        ruff = shutil.which("ruff")
        if ruff is None:
            self.fail("ruff must be on PATH")
        self.ruff = ruff
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = self.root / "origin.git"
        self.author = self.root / "author"
        self.ci = self.root / "ci"
        self._git(self.root, "init", "--quiet", "--bare", "--initial-branch=main", str(self.origin))
        self._git(self.root, "clone", "--quiet", str(self.origin), str(self.author))
        self.base = self._commit(
            None, {"pyproject.toml": "[tool.ruff]\nline-length = 120\n", "pkg/checkpointing.py": _BASE_SOURCE}
        )
        self._git(self.author, "push", "--quiet", "origin", f"{self.base}:refs/heads/main")
        self._git(self.root, "clone", "--quiet", str(self.origin), str(self.ci))

    def _git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Lint Test",
                "-c",
                "user.email=lint@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    def _commit(self, parent: str | None, files: dict[str, str]) -> str:
        if parent is not None:
            self._git(self.author, "checkout", "--quiet", "--detach", parent)
        for path, source in files.items():
            target = self.author / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        self._git(self.author, "add", "--all")
        self._git(self.author, "commit", "--quiet", "--message", "change")
        return self._git(self.author, "rev-parse", "HEAD")

    def _open(self, number: int, head: str) -> dict[str, object]:
        self._git(self.author, "push", "--quiet", "--force", "origin", f"{head}:refs/pull/{number}/head")
        return {"number": number, "head_sha": head}

    def _advance_main(self, files: dict[str, str]) -> str:
        main = self._commit(self.base, files)
        self._git(self.author, "push", "--quiet", "origin", f"{main}:refs/heads/main")
        return main

    def _run(self, pull_requests: list[dict[str, object]], base: str = "origin/main") -> tuple[list[dict], str]:
        self._git(self.ci, "fetch", "--quiet", "origin")
        prs = self.root / "prs.json"
        prs.write_text(json.dumps(pull_requests))
        output = self.root / "output"
        command = [sys.executable, str(_SCRIPT), "--base", base, "--prs", str(prs), "--ruff", self.ruff]
        result = subprocess.run(
            [*command, "--output-dir", str(output)], cwd=self.ci, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._git(self.ci, "for-each-ref", "refs/lint-merged-with-main/"), "")
        return json.loads((output / "results.json").read_text()), (output / "summary.md").read_text()

    def test_pull_request_that_lints_against_old_main_fails_once_main_renames_its_helper(self) -> None:
        pull_request = self._open(
            7,
            self._commit(
                self.base, {"pkg/checkpointing.py": _BASE_SOURCE + "\n\ndef keys(model):\n    return _unwrap(model)\n"}
            ),
        )
        old_statuses, _ = self._run([pull_request], base=self.base)
        self._advance_main({"pkg/checkpointing.py": _BASE_SOURCE.replace("_unwrap", "unwrap")})

        statuses, summary = self._run([pull_request])

        self.assertEqual(old_statuses, [{**pull_request, "state": "success", "description": _CLEAN}])
        self.assertEqual(
            statuses, [{**pull_request, "state": "failure", "description": "Merged into main: 1 new ruff finding"}]
        )
        self.assertIn("pkg/checkpointing.py:14:12: F821 Undefined name `_unwrap`", summary)

    def test_findings_main_already_has_do_not_fail_pull_requests(self) -> None:
        broken = "def broken():\n    return missing\n"
        main = self._advance_main({"pkg/broken.py": broken})
        elsewhere = self._open(1, self._commit(self.base, {"pkg/other.py": "VALUE = 1\n"}))
        same_file = self._open(2, self._commit(main, {"pkg/broken.py": broken + "\n\ndef fine():\n    return 1\n"}))

        statuses, summary = self._run([elsewhere, same_file])

        self.assertEqual(
            statuses,
            [
                {**elsewhere, "state": "success", "description": _CLEAN},
                {**same_file, "state": "success", "description": _CLEAN},
            ],
        )
        self.assertIn("Every merge lints cleanly.", summary)

    def test_conflict_with_main_fails_without_linting(self) -> None:
        self._advance_main({"pkg/checkpointing.py": _BASE_SOURCE.replace("    return model\n", "    return [model]\n")})
        pull_request = self._open(
            3,
            self._commit(
                self.base, {"pkg/checkpointing.py": _BASE_SOURCE.replace("    return model\n", "    return (model,)\n")}
            ),
        )

        statuses, summary = self._run([pull_request])

        self.assertEqual(
            statuses,
            [{**pull_request, "state": "failure", "description": "Conflicts with main, so the merge was not linted"}],
        )
        self.assertIn("conflict: pkg/checkpointing.py", summary)

    def test_unformatted_file_fails(self) -> None:
        pull_request = self._open(4, self._commit(self.base, {"pkg/ugly.py": "x=1\n"}))

        statuses, summary = self._run([pull_request])

        self.assertEqual(
            statuses, [{**pull_request, "state": "failure", "description": "Merged into main: 1 file to reformat"}]
        )
        self.assertIn("pkg/ugly.py: needs ruff format", summary)

    def test_unchanged_status_and_moved_head_are_not_posted(self) -> None:
        unchanged = self._open(5, self._commit(self.base, {"pkg/other.py": "VALUE = 1\n"}))
        moved = self._open(6, self._commit(self.base, {"pkg/other.py": "VALUE = 2\n"}))
        self._open(6, self._commit(self.base, {"pkg/other.py": "VALUE = 3\n"}))

        statuses, summary = self._run([{**unchanged, "state": "success", "description": _CLEAN}, moved])

        self.assertEqual(statuses, [])
        self.assertIn("Merged 1 pull request into main", summary)

    def test_symbolic_link_in_pull_request_is_linted_as_its_target_path(self) -> None:
        secret = self.root / "secret.py"
        secret.write_text("print(leaked_secret_name)\n")
        self._git(self.author, "checkout", "--quiet", "--detach", self.base)
        os.symlink(secret, self.author / "pkg" / "leak.py")
        pull_request = self._open(8, self._commit(None, {}))

        statuses, summary = self._run([pull_request])

        self.assertEqual(statuses[0]["state"], "failure")
        self.assertIn("pkg/leak.py:1:1: invalid-syntax", summary)
        self.assertNotIn("leaked_secret_name", summary)


if __name__ == "__main__":
    unittest.main()
