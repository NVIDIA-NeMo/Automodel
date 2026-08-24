# Shared SWE-bench eval harness (long-context CP validation)

Model-agnostic scripts to evaluate a **consolidated HF checkpoint** on
[SWE-bench](https://www.swebench.com/) (Verified or Lite) with the
[OpenHands](https://github.com/All-Hands-AI/OpenHands) 3-tool agent surface
(`execute_bash` / `str_replace_editor` / `finish`). Shared by the per-model pages
that link here — e.g. [`../gemma4_31B/`](../gemma4_31B) (Phase 3) and
[`../qwen3_32b/`](../qwen3_32b) — so there is **one** eval harness, parameterized by
`MODEL` / `NAME` / `PARSER` / topology, not a copy per model.

How it works: **vLLM** serves the checkpoint; the **oh3 agent** drives the 3-tool
loop against each SWE-bench instance's repo inside a **Docker-less enroot** container
(the cluster has enroot + pyxis, no Docker); the resulting patches are graded
**locally in enroot** with the official `swebench` spec.

## Scripts

| Script | Role | Compute |
|---|---|---|
| `setup_eval_tooling.sh` | py3.10 venv + `mini-swe-agent` + `swebench` (one-time) | login |
| `prewarm_images.sub` | pre-import the per-instance enroot images **once** into a shared lustre cache (avoids Docker Hub throttling from concurrent pulls) | CPU |
| `openhands3_run.sub` | serve vLLM + run the oh3 agent over a slice → `preds.json` | 1 node / 8 GPU |
| `grade_enroot.sub` | apply patches + run FAIL_TO_PASS/PASS_TO_PASS in enroot → resolve rate | CPU |
| `probe_indist.sub` | optional one-call smoke test: does the ckpt emit structured tool calls yet? | 2 GPU |

Helpers (imported, not run directly): `oh3_run.py` (the agent), `enroot_env.py`
(Docker-less enroot backend), `grade_enroot.py` (local grader), `prewarm_images.py`,
`probe_indist.py`.

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
  Use this for checkpoints whose args don't match any built-in parser — e.g. the
  **gemma4 base / base-SFT** emit JSON-prior args that vLLM's `gemma4` parser mangles.

Serve base vs SFT (or any two checkpoints you compare) **identically** so the delta
reflects the model, not the serving config.

## Grading

Grading runs the official `swebench` spec **locally inside the same enroot images**
(no Docker, no cloud upload — the SWE-bench cloud grader returned "failed" for every
submission on our account, including a gold-patch control). The harness prints an
**`enroot-errs`** count: **it must be 0**, otherwise some per-instance containers failed
to build and a `0.0` resolve rate is a false zero, not a real result. Sanity: feeding
the gold patches resolves 5/5 (100%), so the grader is trusted.

## Cluster constraints (learned)

- **enroot data path must be node-local** (`/tmp`), not lustre — lustre can't represent
  overlay whiteouts. The scripts set `ENROOT_*_PATH` under `/tmp`.
- **Pre-warm images first.** Importing images at agent-worker concurrency triggers Docker
  Hub burst-throttling; `prewarm_images.sub` imports each once at low concurrency + retries.
- **Don't strand GPUs.** No `--exclusive` (or explicit `--mem`/`--cpus-per-task`) with a
  GPU subset — the idle-GPU monitor auto-cancels such jobs; request `--gpus-per-node=N`.
  Serve + agent live in one job so no GPU sits idle and no dangling endpoint is left.

## Per-model results

Numbers, checkpoints, and model-specific serving notes live on the per-model pages:
- Gemma4-31B: [`../gemma4_31B/README.md`](../gemma4_31B/README.md) (Phase 3).
- Qwen3-32B: [`../qwen3_32b/`](../qwen3_32b) (eval added in a follow-up).
