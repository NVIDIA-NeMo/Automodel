#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Doc-coverage check + bot-comment upserter.

Detects architectures registered in ``MODEL_ARCH_MAPPING`` that have no model
card under ``docs/model-coverage/``, then upserts a single bot-marked PR
comment listing them. Runs standalone on a stock ubuntu-latest runner — no
nemo_automodel install required, since the registry is parsed via AST rather
than imported.

Behavior:
  - empty report      -> deletes the existing bot comment, if any.
  - pending entries   -> upserts the bot comment, exits 0.
  - expired entries   -> upserts the bot comment, exits 1 (CI check fails).

Talks to the GitHub REST API directly via ``urllib`` (no ``gh`` CLI on the
runner). ``GH_TOKEN`` (or ``GITHUB_TOKEN``) must be in the env.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, parse, request

_MARKER = "<!-- nemo-doc-coverage-bot -->"
_API_ROOT = "https://api.github.com"

# Fallback if the test module cannot be loaded (path moved, etc.). The test
# is the source of truth — see ``_load_grace_period`` below.
_DOC_GRACE_PERIOD_DAYS_DEFAULT = 7


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_arch_names(registry_path: Path) -> list[str]:
    """Extract ``MODEL_ARCH_MAPPING`` keys from registry.py via AST.

    Avoids importing ``nemo_automodel`` (which would pull in torch and the
    full model stack). Walks the ``OrderedDict([(arch, value), ...])``
    literal — same structure the doc-coverage test relies on.
    """
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets
        if not (len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == "MODEL_ARCH_MAPPING"):
            continue
        if not (isinstance(node.value, ast.Call) and node.value.args):
            continue
        list_arg = node.value.args[0]
        if not isinstance(list_arg, ast.List):
            continue
        names: list[str] = []
        for elt in list_arg.elts:
            if not (isinstance(elt, ast.Tuple) and elt.elts):
                continue
            key = elt.elts[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.append(key.value)
        return names
    return []


def _load_test_module(test_file: Path):
    """Load ``test_doc_coverage.py`` by file path so we can read its
    constants (``_DOC_ARCH_ALIASES``, ``_DOC_GRACE_PERIOD_DAYS``) without
    requiring an importable ``tests`` package. Module-level imports in
    that test file are stdlib-only, so this does not pull in nemo_automodel.
    """
    spec = importlib.util.spec_from_file_location("_doc_coverage_test_const", test_file)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arch_registration_age_days(arch_name: str, repo_root: Path) -> float | None:
    """Days since ``arch_name`` was last added to ``registry.py`` per
    ``git log -S``. Returns None if history is unavailable (shallow clone,
    git not on PATH); the caller treats that as "fresh".
    """
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-1",
                "--format=%ct",
                "-S",
                f'"{arch_name}"',
                "--",
                str(registry_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if not out:
        return None
    try:
        return (time.time() - int(out)) / 86400.0
    except ValueError:
        return None


def _build_report(repo_root: Path) -> tuple[list[dict], int]:
    """Compute the missing-arch report and return ``(report, grace_days)``.
    Each entry has the same shape the legacy JSON file used.
    """
    registry_path = repo_root / "nemo_automodel" / "_transformers" / "registry.py"
    arch_names = _extract_arch_names(registry_path)
    if not arch_names:
        raise RuntimeError(
            f"AST parse of {registry_path} returned no arch names — "
            "the MODEL_ARCH_MAPPING literal format may have changed."
        )

    test_file = repo_root / "tests" / "unit_tests" / "_transformers" / "test_doc_coverage.py"
    test_mod = _load_test_module(test_file)
    aliases: dict[str, str] = dict(getattr(test_mod, "_DOC_ARCH_ALIASES", {})) if test_mod else {}
    grace_days = (
        int(getattr(test_mod, "_DOC_GRACE_PERIOD_DAYS", _DOC_GRACE_PERIOD_DAYS_DEFAULT))
        if test_mod
        else _DOC_GRACE_PERIOD_DAYS_DEFAULT
    )

    docs_dir = repo_root / "docs" / "model-coverage"
    md_contents = [p.read_text(encoding="utf-8") for p in docs_dir.rglob("*.md")]

    report: list[dict] = []
    for arch in arch_names:
        needle = aliases.get(arch, arch)
        if any(needle in c for c in md_contents):
            continue
        age = _arch_registration_age_days(arch, repo_root)
        report.append(
            {
                "arch": arch,
                "needle": needle,
                "expired": (age is not None and age >= grace_days),
                "age_days": (None if age is None else round(age, 2)),
                "remaining_days": (None if age is None else round(max(0.0, grace_days - age), 2)),
            }
        )
    return report, grace_days


def _api(method: str, path: str, body: dict | None = None) -> object:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Neither GH_TOKEN nor GITHUB_TOKEN is set.")
    url = f"{_API_ROOT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = request.Request(url, method=method, data=data)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
        return json.loads(text) if text else None


def _list_comments(repo: str, pr: int) -> list[dict]:
    # Walk the Link header for pagination so we don't miss the bot comment on
    # PRs with long discussion threads.
    comments: list[dict] = []
    path = f"/repos/{repo}/issues/{pr}/comments?per_page=100"
    while path:
        url = f"{_API_ROOT}{path}"
        token = os.environ.get("GH_TOKEN") or os.environ["GITHUB_TOKEN"]
        req = request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with request.urlopen(req) as resp:
            comments.extend(json.loads(resp.read().decode("utf-8")))
            link = resp.headers.get("Link", "")
        path = ""
        for chunk in link.split(","):
            if 'rel="next"' in chunk:
                next_url = chunk.split(";")[0].strip().strip("<>")
                parsed = parse.urlparse(next_url)
                path = parsed.path + ("?" + parsed.query if parsed.query else "")
                break
    return comments


def _find_existing_comment_id(repo: str, pr: int) -> int | None:
    matches = [c["id"] for c in _list_comments(repo, pr) if c.get("body", "").startswith(_MARKER)]
    if not matches:
        return None
    # Reuse the oldest if multiple bot comments exist (pre-marker era).
    return min(matches)


def _get_pr_author(repo: str, pr: int) -> str | None:
    """Return the PR author's login, or None on any API failure. The mention
    is only useful for the initial POST — GitHub doesn't re-notify on PATCH,
    so a flaky lookup here just suppresses the @-ping, never duplicates it.
    """
    try:
        result = _api("GET", f"/repos/{repo}/pulls/{pr}")
    except error.HTTPError:
        return None
    if not isinstance(result, dict):
        return None
    user = result.get("user")
    if not isinstance(user, dict):
        return None
    login = user.get("login")
    return login if isinstance(login, str) else None


def _render_body(report: list[dict], grace_days: int, author: str | None = None) -> str:
    lines = [
        _MARKER,
        "",
        "## Pending model-coverage docs",
        "",
    ]
    if author:
        lines += [f"cc @{author}", ""]
    lines += [
        "The following architectures are registered in `MODEL_ARCH_MAPPING` "
        "but have no model card under `docs/model-coverage/`. This comment "
        "is auto-managed by CI and will be removed once every arch has a "
        "doc page.",
        "",
        "| Architecture | Status |",
        "|---|---|",
    ]
    for entry in report:
        arch = entry["arch"]
        if entry["expired"]:
            status = (
                f"❌ **expired** (registered {entry['age_days']:.1f} days ago) — "
                "the doc-coverage check is now hard-failing. Please add a model card."
            )
        elif entry["age_days"] is None:
            status = "⚠️ pending — registration date unknown (assuming fresh)"
        else:
            status = (
                f"⚠️ pending — {entry['remaining_days']:.1f} day(s) remain in the "
                f"{grace_days}-day grace window before this becomes a hard failure"
            )
        lines.append(f"| `{arch}` | {status} |")

    lines += [
        "",
        "**To resolve**: add a `.md` under "
        "`docs/model-coverage/<modality>/<org>/<model>.md` (preferred for new "
        "architectures), update an existing card to mention the arch name, "
        "or add an entry to `_DOC_ARCH_ALIASES` in "
        "`tests/unit_tests/_transformers/test_doc_coverage.py` with a comment "
        "explaining the mismatch.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help='"owner/name".')
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number. Omit to skip comment posting (e.g., on main-branch runs).",
    )
    args = parser.parse_args()

    report, grace_days = _build_report(_repo_root())
    expired = any(entry["expired"] for entry in report)

    print(json.dumps(report, indent=2))

    if args.pr is None:
        if not report:
            print("All archs documented; nothing to do.")
        return 1 if expired else 0

    try:
        existing_id = _find_existing_comment_id(args.repo, args.pr)
    except error.HTTPError as e:
        # Don't mask a real failure (expired) on a transient API hiccup.
        print(f"Could not list PR comments ({e}); skipping comment update.", file=sys.stderr)
        return 1 if expired else 0

    if not report:
        if existing_id is None:
            print("All archs documented and no prior bot comment; nothing to do.")
        else:
            print(f"All archs documented; deleting stale bot comment {existing_id}.")
            _api("DELETE", f"/repos/{args.repo}/issues/comments/{existing_id}")
        return 0

    author = _get_pr_author(args.repo, args.pr)
    body = _render_body(report, grace_days, author=author)
    if existing_id is None:
        print(f"Posting new bot comment to PR #{args.pr}.")
        _api("POST", f"/repos/{args.repo}/issues/{args.pr}/comments", body={"body": body})
    else:
        print(f"Updating existing bot comment {existing_id} on PR #{args.pr}.")
        _api("PATCH", f"/repos/{args.repo}/issues/comments/{existing_id}", body={"body": body})

    return 1 if expired else 0


if __name__ == "__main__":
    sys.exit(main())
