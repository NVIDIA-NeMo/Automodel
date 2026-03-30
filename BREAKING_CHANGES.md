# Breaking Changes

## CLI Signature Change

**Before:**

```
automodel <command> <domain> -c <config.yaml>
```

**After:**

```
automodel <config.yaml> [--nproc-per-node N] [--overrides ...]
```

A short alias `am` is also available:

```
am <config.yaml> [--nproc-per-node N] [--overrides ...]
```

The positional `<command>` and `<domain>` arguments have been removed. The recipe
class is now specified inside the YAML config via the `recipe._target_` key.

## YAML Config: New Required `recipe` Section

All YAML configs now require a top-level `recipe:` key:

```yaml
recipe:
  _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction
```

Configs without this key will produce an error with guidance on which target to add.

### Available Recipe Targets

| Use Case | `_target_` |
|---|---|
| LLM fine-tuning / pre-training | `nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction` |
| VLM fine-tuning | `nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM` |
| Knowledge distillation | `nemo_automodel.recipes.llm.kd.KnowledgeDistillationRecipeForNextTokenPrediction` |
| Benchmarking | `nemo_automodel.recipes.llm.benchmark.BenchmarkingRecipeForNextTokenPrediction` |
| Sequence classification | `nemo_automodel.recipes.llm.train_seq_cls.TrainFinetuneRecipeForSequenceClassification` |
| Biencoder training | `nemo_automodel.recipes.biencoder.train_biencoder.TrainBiencoderRecipe` |

## Launcher Configuration Moved to YAML

Multi-node launch settings (SLURM, Kubernetes, NeMo-Run) are now configured
entirely within the YAML config file rather than through CLI arguments.

| Launcher | YAML section |
|---|---|
| SLURM | `slurm:` |
| Kubernetes | `k8s:` |
| NeMo-Run | `nemo_run:` |

If none of these sections are present the job runs locally (interactive mode).

## SLURM: Script-Based Submission

The built-in SLURM template and all related YAML fields (`nodes`,
`ntasks_per_node`, `container_image`, `partition`, `account`, `time`,
`extra_mounts`, `hf_home`, `hf_token`, `wandb_key`, `gpus_per_node`,
`master_port`, `env_vars`, `job_name`) have been removed.

SLURM now requires a `script` field pointing to your sbatch script:

```yaml
slurm:
  script: my_cluster.sub
```

All cluster-specific configuration (SBATCH directives, container runtime,
mounts, secrets, NCCL tuning) lives in the sbatch script. Copy the reference
template to get started:

```bash
cp slurm.sub my_cluster.sub
```

The CLI generates a `torchrun` command and exports it as `$AUTOMODEL_COMMAND`
for your script to use. The command uses SLURM environment variables
(`$SLURM_NNODES`, `$SLURM_GPUS_PER_NODE`) so it stays in sync with your
`#SBATCH` directives.

The legacy `custom_script` field is still accepted as an alias for `script`.

Exported environment variables:

| Variable | Description |
|---|---|
| `AUTOMODEL_COMMAND` | Full torchrun invocation |
| `AUTOMODEL_CONFIG` | Absolute path to `job_config.yaml` |
| `AUTOMODEL_JOB_DIR` | Job artifacts directory |
| `AUTOMODEL_REPO_ROOT` | Path to AutoModel source |

`AUTOMODEL_NNODES` and `AUTOMODEL_NPROC_PER_NODE` have been removed — use
`$SLURM_NNODES` and `$SLURM_GPUS_PER_NODE` directly in your script.

## Lightweight CLI-Only Install

A new `automodel[cli]` install extra is available for login nodes or environments
where you only need to submit jobs (SLURM, k8s, NeMo-Run) without running
training locally:

```
pip install nemo-automodel[cli]
```

This installs only `pyyaml` -- no PyTorch, no CUDA dependencies. It is enough
to submit jobs via SLURM or Kubernetes. If you also need NeMo-Run, install it
separately (`pip install nemo-run`). If you try to run a local/interactive job
with the CLI-only install, you will get a clear error message with instructions
to install the full package.

## CLI Module Lives Inside the Package

The CLI entry-point lives at `nemo_automodel/cli/app.py` and is registered as
the `automodel` / `am` console entry-points. A thin convenience wrapper
(`app.py`) at the repository root is available for running from a source
checkout but is **not** installed as part of the package.

## Example Wrapper Scripts Deprecated

The Python wrapper scripts in `examples/` (for example, `examples/llm_finetune/finetune.py`)
are deprecated. They now print a deprecation warning and delegate to the recipe
directly. Use `automodel <config.yaml>` instead.
