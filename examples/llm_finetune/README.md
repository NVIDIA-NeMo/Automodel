# LLM Fine-Tuning Examples

This directory contains NeMo AutoModel LLM fine-tuning recipes organized by model family. Each subdirectory provides YAML configs for a specific family, such as Llama, Mistral, Qwen, Gemma, Nemotron, and others. The main AutoModel README identifies `examples/llm_finetune/` as the location for LLM fine-tune configs and shows these recipes being launched through the `automodel` CLI.

## Running a Recipe

Set up the environment with `uv`, then launch a recipe with `automodel`:

```bash
uv venv
uv sync --frozen
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

To run on multiple GPUs on a single node:

```bash
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 8
```

These commands follow the repository's documented setup and launch pattern.

## Important Note on `finetune.py`

A legacy `finetune.py` entry point exists in this directory, but it is deprecated. The script emits a deprecation warning and explicitly instructs users to launch recipes with:

```bash
automodel <config.yaml> [--nproc-per-node N]
```

So new documentation in this directory should prefer `automodel` over `python finetune.py`. This is also consistent with the main README's documented usage. The inspected script loads a config, constructs `TrainFinetuneRecipeForNextTokenPrediction`, then runs `setup()` followed by `run_train_validation_loop()`, which confirms that these examples are training-entry recipes rather than deployment scripts.

## Multi-Node Launches

For SLURM-based multi-node runs, copy the reference `slurm.sub` script, adapt it for your cluster, and submit it with `sbatch`:

```bash
cp slurm.sub my_cluster.sub
sbatch my_cluster.sub
```

Cluster-specific settings such as nodes, GPUs, partition, container, and mounts should be defined in the sbatch script. NeMo-Run sections are also supported through the cluster guide.

## After Fine-Tuning

These recipes are focused on training. After fine-tuning completes, the resulting checkpoints can be used in downstream evaluation, inference, or deployment workflows. The main README also highlights checkpointing and interoperability with Hugging Face and other NeMo ecosystem components. 

## Deployment Guidance

This examples directory does not currently document a single canonical deployment command for all fine-tuned LLM recipes. Based on the materials reviewed here, the safest documented guidance is:

1. **Use the generated checkpoints in your follow-up evaluation or inference workflow.**
2. **Use AutoModel's documented container workflow** when you want a reproducible GPU-backed environment. The contributing guide documents both the AutoModel container path and a custom Docker build path.
3. **Refer to the broader NeMo and AutoModel documentation for production deployment architecture**, rather than assuming a serving/export API directly from these training examples. The repository positions AutoModel as part of the broader NeMo ecosystem for scalable training and deployment-oriented environments.

## Development Notes

If you update documentation here, the contributing guide points contributors to the documentation development guide and requires signed-off commits:

```bash
git commit -s -m "docs: add llm finetune README"
```

Unsigned commits are not accepted. 