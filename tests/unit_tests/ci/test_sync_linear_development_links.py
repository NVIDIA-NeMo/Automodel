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

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[3] / ".github" / "scripts" / "sync_linear_development_links.py"
_SPEC = importlib.util.spec_from_file_location("sync_linear_development_links", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SYNC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SYNC
_SPEC.loader.exec_module(_SYNC)


class _FakeLinearClient:
    def __init__(self, issues_by_url):
        self.issues_by_url = issues_by_url
        self.links = []

    def issues_for_github_url(self, url):
        return self.issues_by_url.get(url, ())

    def link_pull_request(self, *, issue, pull_request_url):
        self.links.append((issue.identifier, pull_request_url))


def test_sync_links_missing_development_attachment():
    pull_request_url = "https://github.com/NVIDIA-NeMo/Automodel/pull/2929"
    github_issue_url = "https://github.com/NVIDIA-NeMo/Automodel/issues/2870"
    linear_issue = _SYNC._LinearIssue(id="linear-584", identifier="AM-584")
    linear = _FakeLinearClient({pull_request_url: (), github_issue_url: (linear_issue,)})

    stats = _SYNC._sync_pull_requests(
        [_SYNC._PullRequest(number=2929, url=pull_request_url, closing_issue_urls=(github_issue_url,))], linear
    )

    assert linear.links == [("AM-584", pull_request_url)]
    assert stats.scanned == 1
    assert stats.linked == 1
    assert stats.planned == 0
    assert stats.existing == 0
    assert stats.unmapped == 0


def test_sync_is_idempotent_and_skips_unsynced_github_issues():
    pull_request_url = "https://github.com/NVIDIA-NeMo/Automodel/pull/2929"
    mapped_issue_url = "https://github.com/NVIDIA-NeMo/Automodel/issues/2870"
    unmapped_issue_url = "https://github.com/NVIDIA-NeMo/Automodel/issues/9999"
    linear_issue = _SYNC._LinearIssue(id="linear-584", identifier="AM-584")
    linear = _FakeLinearClient(
        {pull_request_url: (linear_issue,), mapped_issue_url: (linear_issue,), unmapped_issue_url: ()}
    )

    stats = _SYNC._sync_pull_requests(
        [
            _SYNC._PullRequest(
                number=2929,
                url=pull_request_url,
                closing_issue_urls=(mapped_issue_url, unmapped_issue_url),
            )
        ],
        linear,
    )

    assert linear.links == []
    assert stats.linked == 0
    assert stats.planned == 0
    assert stats.existing == 1
    assert stats.unmapped == 1


def test_parse_pull_request_deduplicates_issue_urls():
    payload = {
        "number": 2929,
        "url": "https://github.com/NVIDIA-NeMo/Automodel/pull/2929",
        "closingIssuesReferences": {
            "nodes": [
                {"url": "https://github.com/NVIDIA-NeMo/Automodel/issues/2870"},
                {"url": "https://github.com/NVIDIA-NeMo/Automodel/issues/2870"},
            ]
        },
    }

    pull_request = _SYNC._GitHubClient._parse_pull_request(payload)

    assert pull_request.closing_issue_urls == ("https://github.com/NVIDIA-NeMo/Automodel/issues/2870",)


def test_dry_run_reports_planned_link_without_mutating_linear():
    pull_request_url = "https://github.com/NVIDIA-NeMo/Automodel/pull/2929"
    github_issue_url = "https://github.com/NVIDIA-NeMo/Automodel/issues/2870"
    linear_issue = _SYNC._LinearIssue(id="linear-584", identifier="AM-584")
    linear = _FakeLinearClient({pull_request_url: (), github_issue_url: (linear_issue,)})

    stats = _SYNC._sync_pull_requests(
        [_SYNC._PullRequest(number=2929, url=pull_request_url, closing_issue_urls=(github_issue_url,))],
        linear,
        dry_run=True,
    )

    assert linear.links == []
    assert stats.linked == 0
    assert stats.planned == 1


@pytest.mark.parametrize("value", ["Automodel", "NVIDIA-NeMo/Automodel/extra", "/Automodel"])
def test_parse_repository_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="OWNER/NAME"):
        _SYNC._parse_repository(value)
