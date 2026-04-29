---
name: automodel researcher
---

# Automodel Researcher
Autonomous ML researcher that iteratively tunes a fine-tuning config to minimize validation loss.

## Paths
- Notebook: `/path/to/your/notebook.ipynb` (or `/path/to/your/launcher.sh` if the workflow starts from a script)
- Config: `/path/to/your/config.yaml`
- Experiments Output: `$HOME/experiments`
- AutoModel Documentation: `$HOME/Automodel/AGENTS.md`

Replace `Notebook` and `Config` with the paths for the task before starting. The agent reads the notebook or launcher script to understand how training is launched, then tunes copied config files under the experiments directory.

## Budget
- **Number of experiments:** `50` — stop the loop and produce the final deliverable once this many experiments have been recorded in `results.tsv` (regardless of status: `keep`, `discard`, or `crash` all count toward the budget).
- **Per-experiment duration:** `30 minutes` — every run is hard-killed at this cap.

## Setup
0. Run `nvidia-smi` to understand the current hardware setup.
1. Read the notebook and config for full context.
2. Create the experiments dir: `mkdir -p $HOME/experiments`
3. Initialize `results.tsv` with just the header row in the `experiments` directory. The baseline will be recorded after the first run.
4. Initialize `journal.md` in the `experiments` directory. This is the human-readable log of everything you do (see **Journal** section below).
5. Read the notebook carefully to understand the training pipeline. Pay particular attention to how training is launched (e.g., `automodel`, `torchrun`, etc.) and verify that all required dependencies are installed and available before proceeding.
6. Keep the YAML RNG configuration fixed for every run so results stay comparable across experiments:
   ```yaml
   rng:
     _target_: nemo_automodel.components.training.rng.StatefulRNG
     seed: 7777
     ranked: true
   ```
7. Kick off experimentation.

## Permissions
**You MAY:**
- Edit the config YAML (learning rate, batch size, warmup, scheduler, weight decay, precision, gradient accumulation — everything is fair game).
- Kill a running experiment early if metrics are clearly diverging. Log it with status discard and prefix the comments field with [ai_killed] followed by your reasoning.
- Install or upgrade packages.

**You MAY NOT:**
- Edit the notebook itself.
- Overwrite a previous experiment's launcher script or log output.
- Create any files outside the `$HOME/experiments` directory.

## Experimentation
- For each experiment, copy the launcher script into `<experiments_dir>` as `<script_name>_{EXP_NUM}.sh`.
- The config file should also be copied as `<script_name>_{EXP_NUM}.yaml` in `<experiments_dir>`. In addition, override the `checkpoint_dir` path to `<script_name>_{EXP_NUM}_checkpoint`.
- Logs, configs, checkpoints, and the launcher script for each run go into the `<experiments_dir>` folder.
- Each experiment runs for the duration specified in the **Budget** section, followed by a hard kill.

**The goal:** Achieve the lowest validation loss possible within the experiment budget.
**First Run — Baseline:** Make any necessary changes to the launcher script to ensure everything executes successfully. Modify the config file only if required to get a clean run.
**Second Run — GPU Saturation:** Fully utilize and saturate the GPU by tuning the batch size. Once you've found the optimal batch size, keep it constant for subsequent runs.
**Third Run onwards — Optimization:** With the batch size locked in, modify only the config file to improve validation loss.

## Logging results
Append to `results.tsv` (tab-separated, never comma-separated), Status is `keep`, `discard`, or `crash`. Example:
```
timestamp	run_name	config_change	val_loss	run_time_min	avg_gpu_utilization_pct	status	comments
20260426_214915	baseline_cli_crash	source/installed CLI mismatch	0.0000	0.2	0	crash	TypeError on check_model_inputs before training started; installed automodel wrapper loaded instead of source recipe; fixed by pinning to source CLI
20260426_222005	baseline	public SmolLM2 fallback; local/global batch 8; lr 1e-4	0.1263	30	60	keep	clean baseline; best@step1319; util 60 = bandwidth-bound, room to grow batch; peak_mem 38141/80000 MiB; later val drifted 0.16-0.25; exit 124 expected timeout
20260427_015921	shuffle_train	batch 9 warmup100; train dataloader shuffle true	0.1063	30	84	keep	NEW BEST; biggest single-knob gain so far (-4.6% vs warmup100); shuffle also pushed util to 84 near target; best@step1019 lr 1e-4; peak_mem 42475 MiB
20260427_005639	lora_rank16_alpha64	batch 9 warmup100; LoRA dim 16 alpha 64	0.1244	7.2	65	discard	[ai_killed] killed at step ~1500 after best@step219 lr 1e-4; doubling adapter capacity hurt — val drifted to 0.18-0.20 with no recovery; peak_mem 41249 MiB
20260428_063913	seed4242_best_recipe	seed4242 lr1.05e-4 betas [0.875,0.95]; LoRA dim 8 alpha 32	0.0892	30	86	keep	NEW BEST (+0.0006 over seed7777); best@step6729 lr 1.05e-4; CAUTION: step-6729 dip reproducing across seeds since shuffle_seed3333 — likely a fixed val-batch artifact, not a true minimum; seed spread 0.0892-0.0916 ≈ hyperparam gains, suggests we're at noise floor; peak_mem 42475 MiB; last_val 0.0948
```

## Journal (human-readable log)
`results.tsv` captures the *what*; the journal captures the ***why***. Maintain `$HOME/experiments/journal.md` as a plain-English chronological log of every meaningful step you take. The reasoning matters more than the numbers — a future reader (or future you) should be able to read the journal and reconstruct how you got from the baseline to the current best.

**What to log in `journal.md`:**
- Files created, modified, or deleted (with paths).
- Packages installed or upgraded (with versions).
- Commands run that change system state (kills, env changes, mounts, etc.).
- For each experiment: a short paragraph stating your *hypothesis* before the run ("I think raising the learning rate will help because the loss curve is still descending steeply at the end of the baseline"), and a short paragraph *after* the run with what actually happened and what you learned.
- Dead ends, surprises, and decisions to abandon a direction — these are especially valuable.
- Any documentation or repository sections you consulted and what you took away from them.

**Format:** Plain Markdown. Each entry starts with a timestamp header and or run name. Keep it natural and narrative — full sentences, not bullet fragments. Example:

```markdown
## 2026-03-27 00:00 UTC — Setup
Cloned repo, created `$HOME/experiments`, initialized `results.tsv` and this journal.
Ran `nvidia-smi`: 1× A100 80GB available, driver 550.x, CUDA 12.4.
...

## 2026-03-27 00:15 UTC — baseline
Launching the config as-is to establish a reference point. No changes to the launcher
or YAML beyond pointing `checkpoint_dir` at `dapt_1_checkpoint`. Hypothesis: this run
will likely be GPU-underutilized given the default `local_batch_size=1`, but I want a
clean baseline before tuning anything.
...
```

## The Experiment Loop
0. Ensure the GPU is free. If a Python process is occupying the GPU, kill it and verify that it has been terminated before proceeding.
1. Read the current config and `results.tsv` — note the best val_loss so far.
2. Modify the config with the change most likely to reduce val_loss.
3. Run the experiment by invoking `<script_name>_{EXP_NUM}.sh` (see above). Redirect all output to `<experiments_dir>/${RUN_NAME}.log` — do NOT let output flood your context.
4. Read the log file at `<experiments_dir>/${RUN_NAME}.log` and formulate your analysis.
5. Append the result from step 4 to `results.tsv`.
6. Repeat from step 0.

**Crashes**: If a run crashes (OOM, a bug, etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from a computer, and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

**GPU Saturation**: Monitor GPU utilization throughout each run. The target is an average utilization of at least 85%. Log the observed average in `results.tsv` and note it, then move on.

**Reference and Documentation**: If you get stuck, exhaust your ideas, or notice validation loss has plateaued with no further progress, consult the documentation and explore the repository for parameters or features you might have overlooked. Refer to the NeMo AutoModel documentation at `$HOME/Automodel/skills/README.md`.

## Final Deliverable: Best-Run Notebook
Once the experiment budget defined in **Budget** is exhausted (or the human interrupts the loop), produce a clean, runnable notebook in `$HOME/experiments` that captures the best-performing configuration discovered during experimentation. Start from the original notebook and apply **surgical edits only** — change exclusively the parameters and settings that reflect what you learned about the hardware and the winning run. Every modification should map directly to a finding in `results.tsv` or `journal.md`; if a change cannot be justified by the story those files tell, leave the original untouched.
