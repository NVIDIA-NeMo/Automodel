# `.github/` — workflows and PR mechanics

Adds to the repository root `AGENTS.md`.

**Load the `cicd` skill before investigating a CI failure or changing a
workflow.**

## How CI is triggered

The heavy pipeline runs on `push`, **not** on `pull_request`. `copy-pr-bot`
decides when a PR's commits get pushed to a `pull-request/<N>` branch:

- All commits GPG-signed by a verified NVIDIA contributor → automatic.
- Otherwise an NVIDIAN comments `/ok to test <full-commit-sha>` on the PR.

Pushing another commit does not re-trigger the heavy pipeline. Only the
lightweight `pull_request` workflows (`Validate PR title`, secrets detector)
fire on their own. Do not push empty commits to force a run.

## PR requirements

- Every commit needs DCO sign-off: `git commit -s`. Repair with
  `git commit --amend -s` and force-push.
- The PR title must be Conventional Commits (see the root `AGENTS.md`);
  `.github/workflows/semantic_pull_request.yml` enforces it.
- Fill in `.github/PULL_REQUEST_TEMPLATE.md`: what changed, changelog bullets,
  pre-checks.

## Editing workflows

- Third-party actions are **pinned to a full commit SHA** with the version in a
  trailing comment. Keep that form when bumping.
- `.github/workflows/claude-review.yml` is also development guidance — see the
  root `AGENTS.md`.
  `tests/unit_tests/test_claude_review_policy.py` pins its prompt against
  silent weakening, so a prompt edit must update that test deliberately.
- Reusable `NVIDIA-NeMo/FW-CI-templates` callers are repinned routinely; change
  the ref, not the caller wiring.
