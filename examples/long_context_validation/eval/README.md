# Shared SWE-bench eval harness (long-context CP validation)

Model-agnostic scripts to evaluate a **consolidated HF checkpoint** on
[SWE-bench](https://www.swebench.com/) (Verified or Lite) with the
[OpenHands](https://github.com/All-Hands-AI/OpenHands) 3-tool agent surface
(`execute_bash` / `str_replace_editor` / `finish`). Shared by the per-model pages
that link here — e.g. [`../gemma4_31B/`](../gemma4_31B) (Phase 3) and
[`../qwen3_32b/`](../qwen3_32b) — so there is **one** eval harness, parameterized by
`MODEL` / `NAME` / `PARSER` / topology, not a copy per model.

## Scripts

| Script | Role | Compute |
|---|---|---|
| `setup_eval_tooling.sh` | small isolated Python 3.10 venv + `mini-swe-agent` (the oh3 agent) + `swebench` — an isolated sandbox for the eval tooling (one-time) | login |
| `prewarm_images.sub` | pre-import the per-instance enroot images **once** into a shared lustre cache (avoids Docker Hub throttling from concurrent pulls) | CPU |
| `openhands3_run.sub` | serve vLLM + run the oh3 agent over a slice → `preds.json` | 1 node / 8 GPU (both models) |
| `grade_enroot.sub` | apply patches + run FAIL_TO_PASS/PASS_TO_PASS in enroot → resolve rate | CPU |
| `probe_indist.sub` | optional one-call smoke test: does the ckpt emit structured tool calls yet? | 2 GPU |

Helpers (imported, not run directly): `oh3_run.py` (the agent), `enroot_env.py`
(Docker-less enroot backend), `grade_enroot.py` (local grader), `prewarm_images.py`,
`probe_indist.py`.

## How it works

```
   the model              the "hands"            the workbench             the judge
  ┌──────────┐          ┌────────────┐         ┌────────────┐          ┌──────────┐
  │  vLLM    │◄────────►│ agent loop │────────►│  enroot    │          │ enroot + │
  │ (serves  │  chat +  │ (oh3_run)  │  runs   │ per-task   │  patch   │ swebench │
  │  ckpt)   │  tools   │            │ commands│  container │ ───────► │  tests   │
  └──────────┘          └────────────┘         └────────────┘          └──────────┘
     on GPUs               Python              one Linux box              on CPU
```

Each SWE-bench task is a real GitHub bug: a repo snapshot at the buggy commit plus a
hidden test that passes only once the bug is fixed. The four phases:

- **Serve** — vLLM loads the checkpoint and exposes an OpenAI-style endpoint the agent
  calls once per turn.
- **Agent** — `oh3_run.py` drives the 3-tool loop: ask the model for a tool call, run it,
  feed the result back, repeat until it finishes or hits the turn cap.
- **Sandbox** — each task runs in its own **enroot** container built from the task's
  official SWE-bench image (the buggy repo, right commit, right deps). The agent's
  `execute_bash` / file edits run *inside* it; the candidate patch is a `git diff`.
- **Grade** — the patch is applied and the task's hidden tests (FAIL_TO_PASS /
  PASS_TO_PASS) run in the same enroot image; both pass ⇒ **resolved**.

## Run order

```bash
# 0. one-time tooling
bash setup_eval_tooling.sh

# 1. pre-warm the shared image cache for the subset (REQUIRED before full runs)
SUBSET=verified sbatch prewarm_images.sub

# 2. serve + run the agent -> preds.json.  MODEL is REQUIRED; pick the tool-call parser
#    to match the checkpoint (see "Tool-call parser" below).
MODEL=<consolidated ckpt> NAME=<label> RUN_TAG=<run> SLICE=0:500 SUBSET=verified \
  <PARSER=hermes | NOPARSER=1> MAX_TOKENS=16384 TP=2 DP=4 WORKERS=16 \
  sbatch --gpus-per-node=8 openhands3_run.sub

# 3. grade (SUBSET must match the eval subset; confirm enroot-errs=0 before trusting a 0.0 resolve)
PREDS=<runs>/<run>/preds.json SUBSET=verified RUN_TAG=grade_<run> sbatch grade_enroot.sub
```

Runs are **resumable**: `preds.json` is written incrementally, so re-submitting the
same `RUN_TAG` after a 4h-wall cutoff skips completed instances and continues.

## Key env knobs

- `MODEL` (**required**): consolidated HF checkpoint dir or HF id.
- `NAME`: served-model-name label (any string; must match what the agent requests).
- `SUBSET` (`verified` | `lite`), `SPLIT` (`test`), `SLICE` (`0:500`).
- `TP` × `DP`: tensor- × data-parallel (8-GPU node → `TP=2 DP=4` = 4 replicas).
- `WORKERS`: concurrent agent workers. High-attempt models run many heavy repo
  builds/tests concurrently — use `16` to avoid host OOM; lighter models tolerate `32`.
- `MAX_TOKENS`: per-turn generation cap.
- `RUN_TAG`: names the output dir under `<cache>/eval/runs/<RUN_TAG>`.

### Tool-call parser (`PARSER` / `NOPARSER`)

A tool call is only usable if something parses the model's raw text into a structured
`{name, arguments}` call. Match the parser to the checkpoint's **measured** output:

- **`PARSER=<vllm parser>`** (default `gemma4`): use vLLM's server-side parser, e.g.
  `PARSER=hermes` for a Qwen3-it checkpoint that emits `<tool_call>{...}</tool_call>`.
- **`NOPARSER=1`**: disable the parser (sets `tool_choice=none`) so raw `call:name{...}`
  reaches `content`, where `oh3_run.py`'s `json.loads`/hermes fallbacks recover the call.
  Use this when a built-in parser drops or mangles the args — e.g. **gemma4 base / base-SFT**
  emit JSON-prior args the `gemma4` parser corrupts, and **Qwen3** uses `NOPARSER=1` because
  the strict `hermes` parser drops calls with control chars in code args (its lenient
  `parse_hermes_tool_calls` fallback recovers them). See the per-model pages.

Serve base vs SFT (or any two checkpoints you compare) **identically** so the delta
reflects the model, not the serving config.

## Grading

Grading runs the official `swebench` spec **locally inside the same enroot images**
(no Docker, no cloud upload). The harness prints an **`enroot-errs`** count: **it must
be 0**, otherwise some per-instance containers failed to build and a `0.0` resolve rate
is a false zero, not a real result.

**Gold-patch sanity (manual one-off).** To trust the grader, we ran it once on the
dataset's *known-correct* solutions: set each candidate `model_patch` to the instance's
reference `patch` (the gold solution) for 5 instances and grade that. Being the known
good fix, it resolves **5/5 (100%)**, which validates the **grading harness** end-to-end
— images build, the patch applies in `/testbed`, `eval_script` runs the FAIL_TO_PASS /
PASS_TO_PASS tests, and scoring is correct — so a `0.0` on real preds is a genuine model
result, not a broken grader. This was a manual one-off (there is no `--gold` flag) that
deliberately bypasses the model, so it checks **only grading**, not the agent/serving path.

## Per-model results

Numbers, checkpoints, and model-specific serving notes live on the per-model pages:
- Gemma4-31B: [`../gemma4_31B/README.md`](../gemma4_31B/README.md) (Phase 3).
- Qwen3-32B: [`../qwen3_32b/`](../qwen3_32b) (eval added in a follow-up).

## Gotchas

- **enroot data path must be node-local** (`/tmp`), not lustre — lustre can't represent
  overlay whiteouts. The scripts set `ENROOT_*_PATH` under `/tmp`.
- **Pre-warm images first.** Importing images at agent-worker concurrency triggers Docker
  Hub burst-throttling; `prewarm_images.sub` imports each once at low concurrency + retries.
- **Don't strand GPUs.** No `--exclusive` (or explicit `--mem`/`--cpus-per-task`) with a
  GPU subset — the idle-GPU monitor auto-cancels such jobs; request `--gpus-per-node=N`.
  Serve + agent live in one job so no GPU sits idle and no dangling endpoint is left.
