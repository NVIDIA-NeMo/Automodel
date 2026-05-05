---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, flash-attn, torch, NRX, or any package in pyproject.toml), regenerate `uv.lock` and `docker/common/uv-pytorch.lock`, open a PR, and drive it to green by attaching a watchdog to the `CICD NeMo` workflow and quarantining failing functional tests with `@pytest.mark.pleasefixme` until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml`, `uv.lock`, or `docker/common/uv-pytorch.lock`, and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump torch', 'bump flash-attn', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'bump base image'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in NeMo AutoModel.
Optimised for the case where TE, torch, or another GPU-heavy pin moves
forward — which often surfaces flakes in the L0 unit + L2 e2e matrix
(plus its GB200 mirror) that have to be quarantined before the PR can
land.

The pipeline is always: **edit → relock (twice) → push → watchdog →
quarantine on red → re-trigger by pushing → repeat until
`Nemo_CICD_Test` is green**.

## When to reach for this skill

- Bumping a `[project.optional-dependencies]` pin (TE, flash-attn,
  mamba, grouped_gemm, etc.) in `pyproject.toml`.
- Bumping the base image / `BASE_IMAGE` build arg (touches both
  `pyproject.toml` and `docker/common/uv-pytorch.lock`).
- Any change that touches `uv.lock` and needs the full L0 unit + L2
  e2e matrix (and its gb200 mirror) to prove out before merge.

For routine drift, the scheduled `Generate Uv lock` workflow already
handles weekly bumps — only reach for this skill when you're driving
an explicit, named bump to green.

## Required context

Read first, then follow the steps below:

- @CONTRIBUTING.md — uv extras, container workflow, DCO sign-off, the
  `update_pyproject_pytorch.sh` workaround for the PyTorch base image.
- @skills/build-and-dependency/SKILL.md — `uv sync`/`uv lock` mechanics,
  the two-lockfile model, container choice.
- @skills/cicd/SKILL.md — `copy-pr-bot` trust, `pull-request/<N>` mirror,
  `Approve Test Queue` 5-min stall, log/artifact retrieval.
- @skills/testing/SKILL.md — tier semantics. Note: the
  `@pytest.mark.pleasefixme` quarantine marker is not documented there
  yet, so it lives here for now (Step 7).

## Step 1 — Worktree and edit

Create a worktree off `main` per @CONTRIBUTING.md.

Edit the pin. For a TE bump the canonical knob is in `pyproject.toml`
under `[project.optional-dependencies]` (verify with
`grep -n transformer-engine pyproject.toml`). For an image bump, edit
the `BASE_IMAGE` arg and any container-side pins in `docker/Dockerfile`.

For TE specifically: the GitHub remote uses `release_vX.Y` (underscore),
not `release/vX.Y`. Verify with
`git ls-remote https://github.com/NVIDIA/TransformerEngine.git`. Pin
to a full SHA for reproducibility unless you explicitly want a moving
tip.

## Step 2 — Regenerate both lockfiles

This repo tracks **two** lockfiles: `uv.lock` (the project lock) and
`docker/common/uv-pytorch.lock` (the lock that gets installed into the
PyTorch base container). Bumps that change torch transitively must
update **both**. The reference for the dance is
`.github/workflows/uv-lock-generation.yml`.

After acquiring the project image per @skills/build-and-dependency/SKILL.md,
run `uv lock` inside it for `uv.lock`. Then regenerate the PyTorch-base
lock:

```bash
mv uv.lock uv_main.lock
bash docker/common/update_pyproject_pytorch.sh "$(pwd)"
docker run --rm -v $(pwd):/workspace -w /workspace automodel-bump \
  bash -c 'source /opt/venv/env.sh && uv lock'
mv uv.lock docker/common/uv-pytorch.lock
mv uv_main.lock uv.lock
git checkout -- pyproject.toml      # update_pyproject_pytorch.sh mutates it
```

Confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock docker/common/uv-pytorch.lock
```

If the diff carries changes you didn't ask for (transitive movements
you can't explain), stop and investigate before pushing.

## Step 3 — Commit and push

Sign-off + signed-commit + commit-title format per @CONTRIBUTING.md and
@skills/cicd/SKILL.md. For a bump:

```bash
git add pyproject.toml uv.lock docker/common/uv-pytorch.lock
git commit -S -s -m "build: bump <package> to <ref>"
git push -u origin <branch-name>
```

Recent merged history uses `build: bump <package> to <ref>` for
lockfile/dep work and `ci: Bump base pytorch image to <ver>` for image
bumps (see #2108).

## Step 4 — Open the PR

Title and labels per @CONTRIBUTING.md. The PR body template — durable
record of the bump:

```markdown
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
```

There is **no `needs-more-tests` / `full-test-suite` matrix-expand
label** in this repo: the L0 unit + L2 e2e matrix runs unconditionally
on every PR (mirrored onto GB200 for org members). Apply `Run CICD` and
move on.

To update the PR title or body later, use `gh api -X PATCH
"repos/NVIDIA-NeMo/Automodel/pulls/<N>" -F "body=@/tmp/pr-body.md"` —
never `gh pr edit`.

## Step 5 — Trigger CI on the exact SHA

Trigger mechanics + `pull-request/<N>` mirror branch + the 5-min
`Approve Test Queue` stall live in @skills/cicd/SKILL.md "How CI Is
Triggered". For this loop the rule is simple:

- Maintainers with signed commits: pushing the new SHA is enough; the
  mirror is automatic. To force a re-run on a no-op,
  `git commit --allow-empty -S -s -m "ci: re-trigger"`.
- External contributors: post `/ok to test $(git rev-parse HEAD)` for
  every new SHA.

Use the **full** SHA — the short form silently fails to match.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **`CICD NeMo`** workflow only. Other workflows
(`build-docs`, `Generate Uv lock`, `copyright-check`, `Install test
summary`, `detect-secrets`, `claude-review`) are noise here — the gate
that decides green-or-red for a bump is the `Nemo_CICD_Test` rollup
inside `CICD NeMo`.

**Always attach a watchdog with the Monitor tool. Never schedule wakeups
or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to reason
  about.
- Natural early termination via `TaskStop` once the run is green.

### Watchdog script

Save to `/tmp/watchdog-<PR>.sh` and chmod +x:

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
- Wakeups can't easily fan out to "tell me whenever a job transitions"
  — they only resume the agent on a fixed interval.
- A persistent Monitor surfaces every job edge in real time and exits
  cleanly when the work is done.

## Step 7 — Quarantine on red, then iterate

When a `JOB <name> -> failure` event fires (excluding `gb200_*` jobs
that already have `is-optional: "true"` in `cicd-main.yml` — those don't
fail the rollup):

1. **Triage the failure — is it the bump or a flake?** Pull logs per
   @skills/cicd/SKILL.md "CI Failure Investigation". Only quarantine if
   the failure reproduces on `main` or is clearly unrelated
   infrastructure. If it's caused by the bump itself, **stop
   quarantining** — fix or revert. Quarantining a real regression hides
   the very signal the bump PR exists to surface.

2. **Quarantine via `@pytest.mark.pleasefixme`.** This is the canonical
   Automodel quarantine mechanism: `tests/run_test.sh` passes
   `-m "not pleasefixme"` on every CI run, so any test wearing that
   marker is silently skipped without altering the launcher script or
   the CI matrix:

   ```python
   import pytest

   @pytest.mark.pleasefixme  # quarantined: <PR-link> (<reason>)
   def test_something_flaky():
       ...
   ```

   Map a CI job name (e.g. `gb200_L2_HF_Transformer_VLM` or
   `L2_HF_Transformer_VLM`) to its test folder by looking up the
   `cicd-e2e-tests` matrix in `.github/workflows/cicd-main.yml`:

   - prefix `gb200_` → same test folder, GCP runner mirror
   - the rest is `L2_<Name>` → `test-folder` for that `test-name` in
     the matrix (e.g. `L2_HF_Transformer_VLM` →
     `tests/functional_tests/hf_transformer_vlm/`)

   See `tests/functional_tests/hf_transformer_vlm/test_hf_transformer_vlm.py`
   for working `@pytest.mark.pleasefixme` examples to crib from.

3. **Append to the PR body's Quarantined tests section** with a
   one-line reason and a follow-up tracking link if you have one.

4. **Commit and push to retrigger:**

   ```bash
   git commit -S -s -m "ci: quarantine flaky <test> for <package> bump"
   git push
   # External contributors also need:
   #   gh pr comment <N> --repo NVIDIA-NeMo/Automodel \
   #     --body "/ok to test $(git rev-parse HEAD)"
   ```

5. **Update the PR body** via `gh api PATCH` so the quarantine list
   stays current.

The watchdog is persistent — it picks up the new run automatically and
emits `RUN <id> STARTED` for the new attempt. Loop back to step 1.

## Step 8 — Stop when green

`RUN <id> COMPLETED conclusion=success` is the exit condition. Then:

```bash
gh pr checks <N> --repo NVIDIA-NeMo/Automodel | awk '{print $2}' | sort | uniq -c
TaskStop(<watchdog-task-id>)
gh api -X PATCH "repos/NVIDIA-NeMo/Automodel/pulls/<N>" -F "body=@/tmp/pr-body.md"
```

The protected gates are `Nemo_CICD_Test` and `Nemo_Linting_Test` — both
must be green for branch protection to allow merge.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `docker/common/uv-pytorch.lock` not regenerated → CI installs stale torch | Forgot the second `uv lock` after `update_pyproject_pytorch.sh` | Repeat Step 2's second relock; commit both lockfiles |
| `pyproject.toml` shows uncommitted noise after Step 2 | `update_pyproject_pytorch.sh` mutates `pyproject.toml` in place | `git checkout -- pyproject.toml` after the second relock |
| Run sits in `waiting` for ~5 min after every push | `cicd-wait-in-queue` requires the `test` environment, which the `Approve Test Queue` cron approves on a 5-min cadence | Expected — don't try to bypass the queue |
| Job name doesn't match a folder under `tests/functional_tests/` | `gb200_` prefix is the hardware indicator, not part of the folder | Strip `gb200_`, then look up `test-folder` in `cicd-main.yml` |
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` before locking |
| `gb200_*` job fails but the rollup is still green | Several gb200 entries are `is-optional: "true"` in `cicd-main.yml` | Don't quarantine an optional gb200 failure unless it persists across reruns |
| Quarantine commit doesn't fire a new `CICD NeMo` run | Empty/no-op commit was filtered out, or the existing run is still in progress (concurrency cancels older runs) | Push a non-empty commit; check `gh run list --workflow "CICD NeMo" --branch pull-request/<N>` |

## Anti-patterns

- **Cron / scheduled wakeups for this loop.** Always Monitor.
- **Polling all workflows.** Filter to `CICD NeMo` — `build-docs`,
  `copyright-check`, `Install test summary`, `Generate Uv lock`,
  `claude-review`, and `detect-secrets` are all noise for a bump.
- **Quarantining a real regression** to "make CI green." That defeats
  the purpose of the bump PR. Only mark `@pytest.mark.pleasefixme` if
  the failure reproduces on `main` or is clearly unrelated
  infrastructure.
- **`gh pr edit`** for title/body. Use `gh api PATCH`.
- **HEREDOC in `gh pr create --body`.** Always go through a tmpfile +
  `--body-file`.
- **Bundling unrelated changes** (feature work, refactors) into a
  bump PR. Bumps should stay surgical so CI failures attribute cleanly.
- **Skipping the second relock** for `docker/common/uv-pytorch.lock`.
  The two lockfiles are not interchangeable — CI builds the container
  from the PyTorch lock, not `uv.lock`.
