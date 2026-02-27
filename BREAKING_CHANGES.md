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

## Lightweight CLI-Only Install

A new `automodel[cli]` install extra is available for login nodes or environments
where you only need to submit jobs (SLURM, k8s, NeMo-Run) without running
training locally:

```
pip install nemo_automodel[cli]
```

This installs only `pyyaml` -- no PyTorch, no CUDA dependencies. It is enough
to submit jobs via SLURM or Kubernetes. If you also need NeMo-Run, install it
separately (`pip install nemo-run`). If you try to run a local/interactive job
with the CLI-only install, you will get a clear error message with instructions
to install the full package.

## CLI Module Moved to Repository Root

The CLI entry-point has moved from `nemo_automodel/_cli/app.py` to
`cli/app.py` at the repository root. This separates the CLI from the
core `nemo_automodel` library, making it possible to install and evolve them
independently.

A backward-compatibility shim remains at the old location and will emit a
`DeprecationWarning` on import. Update any direct imports:

```python
# Before
from nemo_automodel._cli.app import main

# After
from cli.app import main
```

## Example Wrapper Scripts Deprecated

The Python wrapper scripts in `examples/` (e.g., `examples/llm_finetune/finetune.py`)
are deprecated. They now print a deprecation warning and delegate to the recipe
directly. Use `automodel <config.yaml>` instead.
