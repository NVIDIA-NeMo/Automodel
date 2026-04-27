#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Upsert a single PR comment listing arches missing a docs/model-coverage page.

Reads the JSON report produced by
``tests/unit_tests/_transformers/test_doc_coverage.py`` (each entry has
``arch``, ``needle``, ``expired``, ``age_days``, ``remaining_days``) and:

  - empty report  -> deletes the existing bot comment, if any (docs landed).
  - non-empty     -> updates the existing bot comment in place, or posts a
                     new one. The bot comment is identified by a stable HTML
                     marker (``_MARKER`` below) so re-runs of the same PR do
                     not create duplicates.

Talks to the GitHub REST API directly via ``urllib`` so we don't depend on
``gh`` being installed on the CI runner. ``GH_TOKEN`` must be in the env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, parse, request

_MARKER = "<!-- nemo-doc-coverage-bot -->"
_API_ROOT = "https://api.github.com"


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
        # 204 No Content (e.g., DELETE) returns empty body.
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
                # Reduce next_url back to a path so the next iteration
                # re-prefixes _API_ROOT (avoids double-prefixing).
                parsed = parse.urlparse(next_url)
                path = parsed.path + ("?" + parsed.query if parsed.query else "")
                break
    return comments


def _find_existing_comment_id(repo: str, pr: int) -> int | None:
    matches = [c["id"] for c in _list_comments(repo, pr) if c.get("body", "").startswith(_MARKER)]
    if not matches:
        return None
    # If multiple bot comments exist (e.g., from a pre-marker era), reuse the
    # oldest so future runs converge on a single canonical comment.
    return min(matches)


def _render_body(report: list[dict]) -> str:
    lines = [
        _MARKER,
        "",
        "## Pending model-coverage docs",
        "",
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
                "the doc-coverage test now hard-fails. Please add a model card."
            )
        elif entry["age_days"] is None:
            status = "⚠️ pending — registration date unknown (assuming fresh)"
        else:
            status = (
                f"⚠️ pending — {entry['remaining_days']:.1f} day(s) remain in the "
                "grace window before this becomes a hard failure"
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
    parser.add_argument("--report", required=True, help="Path to the JSON report.")
    parser.add_argument("--repo", required=True, help='"owner/name".')
    parser.add_argument("--pr", required=True, type=int, help="PR number.")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"No report at {report_path}; nothing to post.", file=sys.stderr)
        return 0
    report = json.loads(report_path.read_text())

    try:
        existing_id = _find_existing_comment_id(args.repo, args.pr)
    except error.HTTPError as e:
        # Don't fail the build if comment lookup hits a transient API error;
        # the test result and warning annotations still convey the signal.
        print(f"Could not list PR comments ({e}); skipping comment update.", file=sys.stderr)
        return 0

    if not report:
        if existing_id is None:
            print("All archs documented and no prior bot comment; nothing to do.")
        else:
            print(f"All archs documented; deleting stale bot comment {existing_id}.")
            _api("DELETE", f"/repos/{args.repo}/issues/comments/{existing_id}")
        return 0

    body = _render_body(report)
    if existing_id is None:
        print(f"Posting new bot comment to PR #{args.pr}.")
        _api("POST", f"/repos/{args.repo}/issues/{args.pr}/comments", body={"body": body})
    else:
        print(f"Updating existing bot comment {existing_id} on PR #{args.pr}.")
        _api("PATCH", f"/repos/{args.repo}/issues/comments/{existing_id}", body={"body": body})
    return 0


if __name__ == "__main__":
    sys.exit(main())
