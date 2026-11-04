---
name: cicd
description: CI/CD reference for NeMo AutoModel — upstream GitHub and internal GitLab workflows, pipeline structure, and CI failure investigation.
when_to_use: Investigating a CI failure, understanding the pipeline structure, writing a commit or PR, triggering CI, 'CI is red', 'how do I trigger CI', 'PR workflow', 'where are the logs', 'CI did not run', '/ok to test'.
---

# CI/CD

## Commit and PR Workflow

### Commits

All commits require DCO sign-off:

```bash
git commit -s -m "feat: add new recipe for Qwen2"
```

If sign-off is missing on a recent commit, amend it:

```bash
git commit --amend -s
```

### Pull Requests

Follow the PR template in `.github/PULL_REQUEST_TEMPLATE.md`. Every PR should
include:

1. **What**: concise description of the change.
2. **Changelog**: bullet list of user-visible changes.
3. **Pre-checks**: confirm linting, tests, and sign-off.

Use the Conventional Commit title format in `AGENTS.md`, without bracket prefixes
(e.g., `ci: expand internal CPU checks`). This applies to GitLab MRs too.

The `Validate PR title` check enforces this. Never use bracket-prefixed styles
such as `[model] feat: …` — they fail validation.

See `CONTRIBUTING.md` for DCO requirements.

### Branch Naming

Follow the branch naming convention in `AGENTS.md`.

## How CI Is Triggered

### Internal GitLab fork

These instructions apply to MRs targeting `internal` in
`gitlab-master.nvidia.com/aire/research/retriever/forks/automodel`, including work
on feature branches. Identify the destination from the remote and MR target;
the local remote name is not significant.

- `.gitlab-ci.yml` is the source of truth. MR pipelines start automatically for
  target `internal`; standalone branch pushes do not run this pipeline. GitHub's
  `copy-pr-bot` and `/ok to test` mechanism do not apply.
- Jobs use the CPU runner tag `nemollm-common-aws`. The CPU test job starts in
  parallel with lint via `needs: []`; it does not consume lint artifacts.
- Preserve repository-wide linting, changed-file secret detection, and recipe
  validation. Follow the testing skill when expanding CPU test discovery.
- When GitHub checks or their shared configuration change, review whether the
  corresponding GitLab check needs updating. Reuse shared configuration and
  validators where possible; do not copy GPU or download requirements into the
  offline MR suite.
- Use GitLab MR pipeline/job logs to verify changes. Distinguish runner startup
  failures from script failures before changing tests or dependencies. Retry
  infrastructure failures only; do not mask test failures with automatic retries.
- Report the tested commit, pipeline result, test/skip counts, and runtime when
  changing suite selection. A local pass does not establish runner compatibility.

### Upstream GitHub

CI is triggered on `push` — **not** on `pull_request`. A bot called
`copy-pr-bot` controls when CI runs.

**Mechanism:**
1. When a PR is opened, `copy-pr-bot` watches for a trust signal.
2. Trust is established in one of two ways:
   - All commits on the PR branch are **GPG-signed** by a verified NVIDIA
     contributor → bot triggers automatically.
   - An NVIDIAN posts `/ok to test <commit-sha>` as a PR comment → bot
     triggers manually for that SHA.
3. Once trusted, `copy-pr-bot` copies the PR's code into the remote branch
   `pull-request/<number>` and pushes it, which fires CI.

**Consequences:**
- CI never runs on untrusted pushes — external contributors always need
  `/ok to test`.
- The running workflow branch is `pull-request/<number>`, not the author's
  feature branch.
- Pushing a new commit does **not** automatically re-trigger CI unless the
  commit is signed or `/ok to test <new-sha>` is posted.

## Upstream GitHub Pipeline Structure

```
lint-check
  └── cicd-container-build
        ├── unit-tests-core
        ├── unit-tests-diffusion
        └── functional-tests (L0 always; L1 with needs-more-tests label; L2 on schedule)
```

CI test scripts live in `tests/ci_tests/`. These are executed in the CI
pipeline and should not be run locally unless reproducing a CI failure.

## Upstream GitHub CI Failure Investigation

### Locating the PR from a CI Branch

```bash
# Extract PR number from branch name (e.g. pull-request/1234)
PR_NUMBER=$(git rev-parse --abbrev-ref HEAD | grep -oP '(?<=pull-request/)\d+')

gh pr view "$PR_NUMBER" --repo NVIDIA-NeMo/Automodel
gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Automodel --name-only
gh pr checks "$PR_NUMBER" --repo NVIDIA-NeMo/Automodel
```

### Investigating a Failing Job

1. **Review the changeset**: `gh pr diff "$PR_NUMBER" --repo NVIDIA-NeMo/Automodel`
2. **Identify the failing job** from `gh pr checks` output.
3. **Fetch job logs**:
   ```bash
   gh run list --repo NVIDIA-NeMo/Automodel --branch "pull-request/$PR_NUMBER"
   gh run view <run_id> --repo NVIDIA-NeMo/Automodel --log-failed > run.log
   ```
4. **Scan logs in chunks** — log files can be large, never load them whole:
   ```bash
   wc -l run.log
   tail -200 run.log
   sed -n '1,200p' run.log
   ```
5. **Cross-reference the changeset** against the failing step.

## Upstream GitHub Common Failure Patterns

| Symptom | Likely Cause | Action |
|---|---|---|
| CI never started | Commits not GPG-signed and no `/ok to test` | Post `/ok to test <full-sha>` on the PR |
| Lint job fails | `ruff` violation | Run `ruff check --fix . && ruff format .` locally |
| Unit tests fail | Code regression or missing import | Run failing test locally; check the PR diff |
| Functional test (L0) fails | Integration breakage | Check GPU runner logs |
| DCO sign-off missing | `git commit` run without `-s` | Amend: `git commit --amend -s` |
| Multi-GPU tests fail silently | `CUDA_VISIBLE_DEVICES` not set | Set `CUDA_VISIBLE_DEVICES` explicitly |
| `torchrun` port conflict | Multiple processes sharing a port | Pass `--master_port=<unused_port>` or set `MASTER_PORT` |
