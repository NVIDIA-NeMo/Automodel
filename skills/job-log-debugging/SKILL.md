---
name: job-log-debugging
description: Locate and diagnose running or finished NeMo AutoModel job logs. Use when a user gives a run name, Slurm job id, log path, experiment name, or asks why an existing, failed, cancelled, or finished job/run failed.
---

# Job Log Debugging

Use this skill to answer "why did this run fail?" from job artifacts. Keep the
workflow metadata-first and log-sample-first; logs may be large and this node
has limited RAM.

## Workflow

1. Identify the target before reasoning.
   - If the user gives a path, inspect that path directly.
   - If the user gives a run name, job name, or timestamped experiment name,
     search for exact matches first.
   - If the user gives a Slurm job id, check active job metadata before
     searching historical logs.

2. Check active or recently visible jobs.

```bash
list_jobs -d | rg -n '<run-name-or-job-id-fragment>' -C 3
```

Use `LOGDIR` from `list_jobs -d` when present. For all-user searches, only use
`list_jobs -d -a` when the user's own job list does not contain the target.

3. Search finished job artifacts.

Default job roots for this workspace:

```bash
/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$USER/jobs
/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/jobs
```

Prefer the `nemotron_retriever_dev` roots above. For older runs, fall back to
legacy roots only after the new location has no match:

```bash
/lustre/fsw/portfolios/llmservice/users/$USER/jobs
/lustre/fsw/portfolios/llmservice/users/ybabakhin/jobs
```

Use bounded `find` searches by exact run name first:

```bash
find /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$USER/jobs -maxdepth 4 -type d -name '*<run-name>*' -print
find /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$USER/jobs -maxdepth 4 -type f -name '*<run-name>*' -print
```

If no exact match is found, search with stable fragments such as the model
family, dataset name, or timestamp. Avoid broad unbounded Lustre searches.

4. Inventory the run directory without opening heavy artifacts.

```bash
find <run-dir> -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
```

Prioritize:
- `_CONFIG`, `_TASKS`, `_VERSION`
- `*_sbatch.sh`, `*/scripts/*.sh`
- `*/configs/*.yaml`
- `sbatch_*.out`
- `log-*.out`

Avoid reading archives (`*.tar.gz`), model checkpoints, datasets, copied
`uv.lock`, parquet corpora, sqlite indexes, or full logs unless the user
explicitly asks.

5. Read wrapper and config context in small slices.

```bash
sed -n '1,220p' <run-dir>/_CONFIG
sed -n '1,220p' <run-dir>/<stage>/configs/*.yaml
tail -n 200 <run-dir>/<stage>/sbatch_*.out
```

The Slurm wrapper often reports only the first task that exited and then the
distributed cleanup. Treat it as routing information, not the root cause.

6. Search per-rank logs for the first real failure.

```bash
rg -n \
  'Traceback|Exception|RuntimeError|ValueError|AssertionError|KeyError|ImportError|ModuleNotFound|FileNotFoundError|IsADirectoryError|PermissionError|CUDA|out of memory|OOM|Killed|fatal|Fatal|ERROR|Error|CANCELLED|TIME LIMIT|NCCL|SIGTERM|SIGKILL|Broken pipe' \
  <run-dir>/<stage>/log-*.out -C 4
```

After `rg` finds a candidate, inspect the surrounding lines with `sed` using
line numbers from the match. Prefer the earliest application-level exception
over later distributed errors.

## Diagnosis Rules

- Root-cause before fallout. In distributed jobs, one rank often raises the
  real exception; other ranks then show `Broken pipe`, TCPStore errors, NCCL
  heartbeat warnings, `SIGTERM`, or `srun` force termination.
- Treat exit `143` as "killed/terminated" unless paired with an earlier direct
  cause. It is often Slurm/Torch elastic cleanup after another rank failed.
- Treat `sbatch_*.out` messages like `task N: Exited with exit code 1` as the
  pointer to the failing task, not a complete explanation.
- For dataset/config failures, quote the bad config key/path and the loader
  frame that consumed it.
- For mount/container failures, check `--container-mounts`, `--container-image`,
  and whether all mounted paths exist on allocated nodes.
- For time limits, look for `DUE TO TIME LIMIT`, `CANCELLED AT`, checkpoint
  activity, and async checkpoint writes near the end of the log.
- For OOM, distinguish GPU OOM (`CUDA out of memory`,
  `torch.OutOfMemoryError`) from process/node OOM (`Killed`, `oom-kill`,
  cgroup memory messages).

## Answer Format

Answer with:

1. The root cause in one sentence.
2. Evidence: log path, stage, rank/task if visible, and the key exception or
   lines.
3. Fallout: what cleanup/noise happened after the root cause.
4. Concrete next action: config/path/code/resource change to rerun safely.

Keep the explanation focused. Do not paste long tracebacks; summarize and cite
the exact file paths and line numbers when possible.
