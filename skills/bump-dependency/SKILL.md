---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, flash-attn, torch, NRX, or any package in pyproject.toml), regenerate uv.lock and the PyTorch lock, open a PR, and drive it to green by attaching a watchdog to the "CICD NeMo" workflow and quarantining failing functional tests with `@pytest.mark.pleasefixme` until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml`, `uv.lock`, or `docker/common/uv-pytorch.lock`, and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump torch', 'bump flash-attn', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'bump base image'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in NeMo AutoModel.
Optimised for the case where TE, torch, or another GPU-heavy pin moves
forward — which often surfaces flakes in the L2 + GB200 functional
matrix that have to be quarantined before the PR can land.

The pipeline is always: **edit -> relock (twice) -> push -> watchdog ->
quarantine on red -> re-trigger by pushing -> repeat until green**.

## When to reach for this skill

- Bumping a `[project.optional-dependencies]` pin (TE, flash-attn,
  mamba, grouped_gemm, etc.) in `pyproject.toml`.
- Bumping the base image / `BASE_IMAGE` build arg (touches both
  `pyproject.toml` and `docker/common/uv-pytorch.lock`).
- Any change that touches `uv.lock` and needs the full L0 unit + L2
  e2e matrix (and gb200 e2e mirror) to prove out before merge.

For pure dep additions/removals where the existing lockfile-bot PR is
acceptable, the scheduled `Generate Uv lock` workflow already handles
weekly drift — only reach for this skill when you're driving an
explicit, named bump to green.

## Required context

Read first, then follow the steps below:

- @CONTRIBUTING.md — uv extras, container workflow, DCO sign-off, the
  `update_pyproject_pytorch.sh` workaround for the PyTorch base image
- @skills/build-and-dependency/SKILL.md — `uv sync`/`uv lock`
  mechanics, the two-lockfile model, container choice
- @skills/cicd/SKILL.md — how `copy-pr-bot` mirrors PRs onto
  `pull-request/<N>` and how CI is triggered
- @skills/testing/SKILL.md — tier semantics and the
  `@pytest.mark.pleasefixme` quarantine marker (used by
  `tests/run_test.sh -m "not pleasefixme"`)

## Step 1 — Worktree and edit

```bash
# From the Automodel repo root
git worktree add .claude/worktrees/<slug> -b <branch-name> origin/main
cd .claude/worktrees/<slug>
```

Edit the pin. For a TE bump the canonical knob is in `pyproject.toml`
under `[project.optional-dependencies]` (verify with
`grep -n transformer-engine pyproject.toml`). For an image bump, edit
the `BASE_IMAGE` arg and any container-side pins in `docker/Dockerfile`.

For TE specifically: the GitHub remote uses `release_vX.Y` (underscore),
not `release/vX.Y`. Verify with
`git ls-remote https://github.com/NVIDIA/TransformerEngine.git`.
Use a full SHA for reproducibility unless you explicitly want a moving
tip.

## Step 2 — Regenerate both lockfiles

`uv.lock` is Linux + CUDA only. The repo also tracks
`docker/common/uv-pytorch.lock`, which is what gets installed in the
PyTorch base container — bumps that change torch transitively must
update **both**. The reference for this pattern is
`.github/workflows/uv-lock-generation.yml`.

Build the project image once:

```bash
docker build -f docker/Dockerfile.ci -t automodel-bump .
```

Then relock inside it (so resolution sees the same Linux + CUDA wheel
universe CI sees):

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -v $HOME/.cache/uv:/root/.cache/uv \
  automodel-bump \
  bash -c 'source /opt/venv/env.sh && uv lock'
```

Then regenerate the PyTorch-base lock (mirrors
`uv-lock-generation.yml`):

```bash
mv uv.lock uv_main.lock
bash docker/common/update_pyproject_pytorch.sh "$(pwd)"
docker run --rm -v $(pwd):/workspace -w /workspace automodel-bump \
  bash -c 'source /opt/venv/env.sh && uv lock'
mv uv.lock docker/common/uv-pytorch.lock
mv uv_main.lock uv.lock
# Restore pyproject.toml — update_pyproject_pytorch.sh mutates it.
git checkout -- pyproject.toml
```

Confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock docker/common/uv-pytorch.lock
```

If the diff carries changes you didn't ask for (transitive movements
you can't explain), stop and investigate before pushing.

## Step 3 — Commit and push

```bash
git add pyproject.toml uv.lock docker/common/uv-pytorch.lock
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin <branch-name>
```

Commit/PR title format follows the repo's recent merged history:
`build: bump <package> to <ref>` for the lockfile/dep work, or
`ci: Bump base pytorch image to <ver>` for image bumps (see #2108).
Sign-off (`-s`) is required by the DCO check; signed commits (`-S`)
let `copy-pr-bot` mirror to `pull-request/<N>` without manual
`/ok to test` for every push.

## Step 4 — Open the PR

PR body goes through a tmpfile to preserve formatting (never HEREDOC
into `gh pr create --body`). Wrap it in a `<details>` block:

```bash
cat > /tmp/pr-body.md <<'EOF'
<details><summary>Claude summary</summary>

## What
- Bump <package> to <ref>.
- Regenerate `uv.lock`.
- Regenerate `docker/common/uv-pytorch.lock` for the PyTorch base.

## Lockfile delta
```
Updated <package> <old> -> <new>
```

## Test plan
- [ ] `Nemo_Linting_Test` green
- [ ] `Nemo_CICD_Test` green (full L0 unit + L2 e2e matrix, AWS + GB200)

## Quarantined tests (this bump)
_None yet — will be appended as flakes are identified during CI iteration._

</details>
EOF

gh pr create \
  --repo NVIDIA-NeMo/Automodel \
  --base main \
  --head <branch-name> \
  --title "build: bump <package> to <ref>" \
  --body-file /tmp/pr-body.md \
  --label "Run CICD"
```

There is **no `needs-more-tests` / `full-test-suite` style label** that
gates the matrix in this repo (the tier-expansion language in
`@skills/testing/SKILL.md` is aspirational — see
`.github/workflows/cicd-main.yml`, where the L0 unit + L2 e2e matrix
runs unconditionally on every PR and is mirrored onto GB200 for org
members). The `Run CICD` label is the only opt-in signal worth
applying for a bump.

`gh pr edit` is unreliable. To update a PR's title or body later, use
the REST API directly:

```bash
gh api -X PATCH "repos/NVIDIA-NeMo/Automodel/pulls/<N>" \
  -F "body=@/tmp/pr-body.md"

gh api -X PATCH "repos/NVIDIA-NeMo/Automodel/pulls/<N>" \
  -f "title=build: bump <package> to <ref>"
```

## Step 5 — Trigger CI on the exact SHA

`copy-pr-bot` (see `@skills/cicd/SKILL.md`) mirrors trusted-author PRs
onto `refs/heads/pull-request/<N>`, which is what `CICD NeMo`
actually triggers on (the `push` filter in `cicd-main.yml`). The
mirrored run then waits in the `test` GitHub Environment until the
`Approve Test Queue` cron (`cicd-approve-test-queue.yml`, every 5
min) lets it past the concurrency gate.

For an external contributor, post `/ok to test <full-sha>` on the PR
the first time and after every new push:

```bash
SHA=$(git rev-parse HEAD)
gh pr comment <N> --repo NVIDIA-NeMo/Automodel --body "/ok to test $SHA"
```

For a maintainer with signed commits, the mirror is automatic — to
force a re-run after a quarantine fix, just push:

```bash
git commit --allow-empty -S -s -m "ci: re-trigger"
git push
```

Use the **full** SHA (`git rev-parse HEAD`), never the short form.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job
state changes for the **CICD NeMo** workflow only. Other workflows
(`build-docs`, `Generate Uv lock`, `copyright-check`, `Install test
summary`, `detect-secrets`, `claude-review`) are noise here — the
gate that decides green-or-red for a bump is `CICD NeMo`, and inside
it the rollup is `Nemo_CICD_Test`.

**Always attach a watchdog with the Monitor tool. Never schedule
wakeups or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to
  reason about.
- Natural early termination via `TaskStop` once the run is green.

### Watchdog script

Save to `/tmp/watchdog-<PR>.sh` and `chmod +x`:

```bash
#!/usr/bin/env bash
# Watchdog: monitor "CICD NeMo" runs on pull-request/<PR> and emit
# per-job state changes. Stays alive across re-runs (new commits).
set -u
PR=<PR>
REPO=NVIDIA-NeMo/Automodel
BRANCH="pull-request/$PR"

prev_run_id=""
declare -A prev_state

emit() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

while true; do
  run_json=$(gh run list --repo "$REPO" --workflow "CICD NeMo" \
    --branch "$BRANCH" --limit 1 \
    --json databaseId,status,conclusion,headSha 2>/dev/null || echo "[]")
  run_id=$(echo "$run_json" | jq -r '.[0].databaseId // empty')
  run_status=$(echo "$run_json" | jq -r '.[0].status // empty')
  run_conclusion=$(echo "$run_json" | jq -r '.[0].conclusion // empty')
  run_sha=$(echo "$run_json" | jq -r '.[0].headSha // empty')

  if [[ -z "$run_id" ]]; then
    sleep 30; continue
  fi

  if [[ "$run_id" != "$prev_run_id" ]]; then
    emit "RUN ${run_id} STARTED sha=${run_sha:0:8} status=${run_status}"
    prev_run_id="$run_id"
    unset prev_state
    declare -A prev_state
  fi

  jobs_json=$(gh run view "$run_id" --repo "$REPO" --json jobs 2>/dev/null || echo "{}")
  while IFS=$'\t' read -r name status conclusion; do
    [[ -z "$name" ]] && continue
    cur="${status}/${conclusion}"
    if [[ "${prev_state[$name]:-}" != "$cur" ]]; then
      case "$status" in
        completed)
          emit "JOB ${name} -> ${conclusion}" ;;
        in_progress)
          if [[ -z "${prev_state[$name]:-}" || "${prev_state[$name]}" == "queued/" ]]; then
            emit "JOB ${name} -> in_progress"
          fi ;;
      esac
      prev_state[$name]="$cur"
    fi
  done < <(echo "$jobs_json" | jq -r '.jobs[]? | [.name, .status, (.conclusion // "")] | @tsv')

  if [[ "$run_status" == "completed" ]]; then
    emit "RUN ${run_id} COMPLETED conclusion=${run_conclusion}"
  fi

  sleep 60
done
```

### Arming the watchdog

```text
Monitor(
  description="CICD NeMo run state changes on PR <N>",
  command="bash /tmp/watchdog-<N>.sh",
  persistent=true,
  timeout_ms=3600000
)
```

`persistent: true` keeps it alive across re-runs (you'll push more
commits when quarantining flakes). Stop it with `TaskStop(<task-id>)`
once the run is green.

### Why never a cronjob / scheduled wakeup

- Cronjobs run blind — they fire on a clock, not on an event. You'll
  either over-poll (cache miss every wake-up) or miss long stalls.
- Wakeups can't easily fan out to "tell me whenever a job
  transitions" — they only resume the agent on a fixed interval.
- A persistent Monitor surfaces every job edge in real time and
  exits cleanly when the work is done.

## Step 7 — Quarantine on red, then iterate

When a `JOB <name> -> failure` event fires (excluding `gb200_*`
optional jobs, several of which already have `is-optional: "true"`
set in `cicd-main.yml` and won't fail the rollup):

1. Skim the logs to confirm it's a flake / pre-existing issue, not
   the bump itself:

   ```bash
   RUN_ID=<from "RUN ... STARTED" event>
   gh run view "$RUN_ID" --repo NVIDIA-NeMo/Automodel --log-failed > /tmp/run.log
   wc -l /tmp/run.log
   tail -200 /tmp/run.log
   ```

   If the failure is caused by the bump (real regression, not a
   flake), **stop quarantining** — fix the underlying issue or
   revert the bump. Quarantining a real regression hides the very
   signal the bump PR exists to surface.

2. Mark the offending pytest function with
   `@pytest.mark.pleasefixme`. This is the canonical Automodel
   quarantine mechanism: `tests/run_test.sh` passes
   `-m "not pleasefixme"` on every CI run, so any test wearing that
   marker is silently skipped without altering the launcher script
   or the CI matrix:

   ```python
   import pytest

   @pytest.mark.pleasefixme  # quarantined: <PR-link> (<reason>)
   def test_something_flaky():
       ...
   ```

   Map a CI job name (e.g. `gb200_L2_HF_Transformer_VLM` or
   `L2_HF_Transformer_VLM`) to its test folder:

   - prefix `gb200_` → same test folder, GCP runner mirror
   - the rest is `L2_<Name>` → look up `test-folder` for that
     `test-name` in the `cicd-e2e-tests` matrix in
     `.github/workflows/cicd-main.yml` (e.g.
     `L2_HF_Transformer_VLM` → `tests/functional_tests/hf_transformer_vlm/`)

   `tests/functional_tests/hf_transformer_vlm/test_hf_transformer_vlm.py`
   has working examples of `@pytest.mark.pleasefixme` to crib from.

3. Append the test to the PR description's **Quarantined tests**
   section, with a one-line reason and a follow-up tracking link if
   you have one. This is the durable record of what this bump
   deferred.

4. Commit, push to retrigger:

   ```bash
   git commit -S -s -m "ci: quarantine flaky <test> for <package> bump"
   git push
   ```

   The push to your branch forwards through `copy-pr-bot` to
   `pull-request/<N>` and `CICD NeMo` re-fires automatically. If
   you're an external contributor, also post:

   ```bash
   SHA=$(git rev-parse HEAD)
   gh pr comment <N> --repo NVIDIA-NeMo/Automodel --body "/ok to test $SHA"
   ```

5. Update the PR body via `gh api PATCH` so the quarantine list
   stays current.

The watchdog is persistent — it will pick up the new run
automatically and emit `RUN <id> STARTED` for the new attempt.

## Step 8 — Stop when green

`RUN <id> COMPLETED conclusion=success` is the exit condition. Then:

```bash
# Sanity check — the protected gates are Nemo_CICD_Test and
# Nemo_Linting_Test (see branch protection).
gh pr checks <N> --repo NVIDIA-NeMo/Automodel | awk '{print $2}' | sort | uniq -c

# Tear down
TaskStop(<watchdog-task-id>)

# Tick the boxes in the PR body
gh api -X PATCH "repos/NVIDIA-NeMo/Automodel/pulls/<N>" -F "body=@/tmp/pr-body.md"
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `uv lock` resolves a different torch in CI than locally | Local resolved against PyPI wheels; CI installs into the PyTorch base container | Always relock inside the project Docker image (Step 2) |
| `docker/common/uv-pytorch.lock` not regenerated → CI installs stale torch | Forgot the second `uv lock` after `update_pyproject_pytorch.sh` | Repeat Step 2's second relock; commit both lockfiles |
| `pyproject.toml` shows uncommitted noise after Step 2 | `update_pyproject_pytorch.sh` mutates `pyproject.toml` in place | `git checkout -- pyproject.toml` after the second relock |
| CI never starts on a new push | Author isn't trusted → `copy-pr-bot` didn't mirror to `pull-request/<N>` | Sign commits with `-S`, or post `/ok to test $(git rev-parse HEAD)` |
| Quarantine commit doesn't fire a new `CICD NeMo` run | Empty/no-op commit was filtered out, or the existing run is still in progress (concurrency cancels older runs) | Push a non-empty commit; check `gh run list --workflow "CICD NeMo" --branch pull-request/<N>` |
| Run sits in `waiting` for ~5 min after every push | `cicd-wait-in-queue` requires the `test` environment, which the `Approve Test Queue` cron approves on a 5-min cadence | Expected — don't try to bypass the queue |
| Job name doesn't match a folder under `tests/functional_tests/` | `gb200_` prefix is the hardware indicator, not part of the folder | Strip `gb200_`, then look up the matrix entry's `test-folder` in `cicd-main.yml` |
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` before locking |
| `gb200_*` job fails but the rollup is still green | Several gb200 entries are `is-optional: "true"` in `cicd-main.yml` | Don't quarantine an optional gb200 failure unless it persists across reruns |

## Anti-patterns

- **Cron / scheduled wakeups for this loop.** Always Monitor.
- **Polling all workflows.** Filter to `CICD NeMo` — `build-docs`,
  `copyright-check`, `Install test summary`, `Generate Uv lock`,
  `claude-review`, and `detect-secrets` are all noise for a bump.
- **Quarantining a real regression** to "make CI green." That
  defeats the purpose of the bump PR. Only mark
  `@pytest.mark.pleasefixme` if the failure reproduces on `main` or
  is clearly unrelated infrastructure.
- **`gh pr edit`** for title/body. Use `gh api PATCH`.
- **HEREDOC in `gh pr create --body`.** Always go through a tmpfile
  + `--body-file`.
- **Bundling unrelated changes** (feature work, refactors) into a
  bump PR. Bumps should stay surgical so CI failures attribute
  cleanly.
- **Skipping the second relock** for `docker/common/uv-pytorch.lock`.
  The two lockfiles are not interchangeable — CI builds the
  container from the PyTorch lock, not `uv.lock`.
