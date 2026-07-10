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

"""Mirror GitHub Development-linked pull requests to synced Linear issues."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PullRequest:
    number: int
    url: str
    closing_issue_urls: tuple[str, ...]


@dataclass(frozen=True)
class _LinearIssue:
    id: str
    identifier: str


@dataclass
class _SyncStats:
    scanned: int = 0
    linked: int = 0
    planned: int = 0
    existing: int = 0
    unmapped: int = 0


def _as_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected {context} to be an object")
    if not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Expected {context} to have string keys")
    return value


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"Expected {context} to be a list")
    return value


def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"Expected {context} to be a string")
    return value


def _as_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"Expected {context} to be an integer")
    return value


def _post_graphql(*, endpoint: str, authorization: str, query: str, variables: dict[str, object]) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": authorization, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            decoded: object = json.loads(response.read())
    except HTTPError as exc:
        response_body = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"GraphQL request failed with HTTP {exc.code}: {response_body}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"GraphQL request failed: {exc}") from exc

    payload = _as_object(decoded, "GraphQL response")
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"GraphQL returned errors: {json.dumps(errors)}")
    return _as_object(payload.get("data"), "GraphQL data")


class _GitHubClient:
    def __init__(self, *, token: str, owner: str, repository: str) -> None:
        self._authorization = f"Bearer {token}"
        self._owner = owner
        self._repository = repository

    def pull_requests(self, number: int | None) -> list[_PullRequest]:
        if number is not None:
            pull_request = self._pull_request(number)
            return [pull_request] if pull_request is not None else []
        return self._open_pull_requests()

    def _pull_request(self, number: int) -> _PullRequest | None:
        query = """
        query PullRequest($owner: String!, $repository: String!, $number: Int!) {
          repository(owner: $owner, name: $repository) {
            pullRequest(number: $number) {
              number
              url
              closingIssuesReferences(first: 10) { nodes { url } }
            }
          }
        }
        """
        data = _post_graphql(
            endpoint=_GITHUB_GRAPHQL_URL,
            authorization=self._authorization,
            query=query,
            variables={"owner": self._owner, "repository": self._repository, "number": number},
        )
        repository = _as_object(data.get("repository"), "GitHub repository")
        payload = repository.get("pullRequest")
        return None if payload is None else self._parse_pull_request(payload)

    def _open_pull_requests(self) -> list[_PullRequest]:
        query = """
        query PullRequests($owner: String!, $repository: String!, $cursor: String) {
          repository(owner: $owner, name: $repository) {
            pullRequests(first: 50, after: $cursor, states: OPEN, orderBy: {field: UPDATED_AT, direction: DESC}) {
              nodes {
                number
                url
                closingIssuesReferences(first: 10) { nodes { url } }
              }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
        """
        cursor: str | None = None
        pull_requests: list[_PullRequest] = []
        while True:
            data = _post_graphql(
                endpoint=_GITHUB_GRAPHQL_URL,
                authorization=self._authorization,
                query=query,
                variables={"owner": self._owner, "repository": self._repository, "cursor": cursor},
            )
            repository = _as_object(data.get("repository"), "GitHub repository")
            connection = _as_object(repository.get("pullRequests"), "GitHub pull request connection")
            pull_requests.extend(
                self._parse_pull_request(node) for node in _as_list(connection.get("nodes"), "PR nodes")
            )

            page_info = _as_object(connection.get("pageInfo"), "GitHub page info")
            if page_info.get("hasNextPage") is not True:
                return pull_requests
            cursor = _as_string(page_info.get("endCursor"), "GitHub end cursor")

    @staticmethod
    def _parse_pull_request(payload: object) -> _PullRequest:
        pull_request = _as_object(payload, "GitHub pull request")
        references = _as_object(pull_request.get("closingIssuesReferences"), "closing issue references")
        issue_urls = {
            _as_string(_as_object(node, "closing issue").get("url"), "closing issue URL")
            for node in _as_list(references.get("nodes"), "closing issue nodes")
        }
        return _PullRequest(
            number=_as_int(pull_request.get("number"), "pull request number"),
            url=_as_string(pull_request.get("url"), "pull request URL"),
            closing_issue_urls=tuple(sorted(issue_urls)),
        )


class _LinearClient:
    def __init__(self, *, token: str, team_key: str) -> None:
        self._authorization = token
        self._team_key = team_key
        self._issue_cache: dict[str, tuple[_LinearIssue, ...]] = {}

    def issues_for_github_url(self, url: str) -> tuple[_LinearIssue, ...]:
        cached = self._issue_cache.get(url)
        if cached is not None:
            return cached

        query = """
        query AttachmentsForURL($url: String!) {
          attachmentsForURL(url: $url, first: 50) {
            nodes { issue { id identifier team { key } } }
          }
        }
        """
        data = _post_graphql(
            endpoint=_LINEAR_GRAPHQL_URL,
            authorization=self._authorization,
            query=query,
            variables={"url": url},
        )
        connection = _as_object(data.get("attachmentsForURL"), "Linear attachment connection")
        issues: dict[str, _LinearIssue] = {}
        for node in _as_list(connection.get("nodes"), "Linear attachment nodes"):
            attachment = _as_object(node, "Linear attachment")
            issue = _as_object(attachment.get("issue"), "Linear issue")
            team = _as_object(issue.get("team"), "Linear team")
            if _as_string(team.get("key"), "Linear team key") != self._team_key:
                continue
            linear_issue = _LinearIssue(
                id=_as_string(issue.get("id"), "Linear issue ID"),
                identifier=_as_string(issue.get("identifier"), "Linear issue identifier"),
            )
            issues[linear_issue.id] = linear_issue

        result = tuple(sorted(issues.values(), key=lambda issue: issue.identifier))
        self._issue_cache[url] = result
        return result

    def link_pull_request(self, *, issue: _LinearIssue, pull_request_url: str) -> None:
        mutation = """
        mutation LinkGitHubPR($issueId: String!, $url: String!) {
          attachmentLinkGitHubPR(issueId: $issueId, url: $url, linkKind: closes) {
            success
            attachment { id }
          }
        }
        """
        data = _post_graphql(
            endpoint=_LINEAR_GRAPHQL_URL,
            authorization=self._authorization,
            query=mutation,
            variables={"issueId": issue.id, "url": pull_request_url},
        )
        payload = _as_object(data.get("attachmentLinkGitHubPR"), "Linear link payload")
        if payload.get("success") is not True:
            raise RuntimeError(f"Linear did not link {pull_request_url} to {issue.identifier}")


def _sync_pull_requests(
    pull_requests: list[_PullRequest], linear: _LinearClient, *, dry_run: bool = False
) -> _SyncStats:
    stats = _SyncStats()
    for pull_request in pull_requests:
        stats.scanned += 1
        if not pull_request.closing_issue_urls:
            continue

        existing_issue_ids = {issue.id for issue in linear.issues_for_github_url(pull_request.url)}
        for github_issue_url in pull_request.closing_issue_urls:
            linear_issues = linear.issues_for_github_url(github_issue_url)
            if not linear_issues:
                stats.unmapped += 1
                _LOGGER.info("No synced Linear issue for %s", github_issue_url)
                continue

            for linear_issue in linear_issues:
                if linear_issue.id in existing_issue_ids:
                    stats.existing += 1
                    continue
                if dry_run:
                    _LOGGER.info("Would link PR #%d to %s", pull_request.number, linear_issue.identifier)
                    stats.planned += 1
                else:
                    linear.link_pull_request(issue=linear_issue, pull_request_url=pull_request.url)
                    _LOGGER.info("Linked PR #%d to %s", pull_request.number, linear_issue.identifier)
                    stats.linked += 1
                existing_issue_ids.add(linear_issue.id)
    return stats


def _parse_repository(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Repository must have OWNER/NAME form, got {value!r}")
    return parts[0], parts[1]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def main() -> None:
    """Synchronize GitHub Development-linked PRs into native Linear attachments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", type=int, help="Only synchronize one pull request")
    parser.add_argument("--dry-run", action="store_true", help="Report missing links without creating them")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    owner, repository = _parse_repository(_required_env("GITHUB_REPOSITORY"))
    github = _GitHubClient(token=_required_env("GITHUB_TOKEN"), owner=owner, repository=repository)
    linear = _LinearClient(token=_required_env("LINEAR_API_KEY"), team_key=os.environ.get("LINEAR_TEAM_KEY", "AM"))
    stats = _sync_pull_requests(github.pull_requests(args.pr_number), linear, dry_run=args.dry_run)
    _LOGGER.info(
        "Scanned %d PRs: linked=%d planned=%d existing=%d unmapped=%d",
        stats.scanned,
        stats.linked,
        stats.planned,
        stats.existing,
        stats.unmapped,
    )


if __name__ == "__main__":
    main()
