# CI Tests

Configuration, scripts, and utilities for AutoModel's CI recipe validation pipeline.

## Directory Structure

```
ci_tests/
  configs/{test_folder}/
    nightly_recipes.yml         # Recipes included in nightly scope
    convergence_recipes.yml     # Recipes included in convergence scope (2x time)
    override_recipes.yml        # Exemptions, known issues
  scripts/
    finetune_launcher.sh        # Finetune + checkpoint robustness test runner
    vllm_launcher.sh            # vLLM deployment test runner
  golden_values/{test_folder}/
    {model}/{config}_{gpu}.jsonl  # Reference loss curves
  utils/
    generate_ci_tests.py        # Generates CI pipeline YAML from recipe configs
```

## Pipeline Generation

`generate_ci_tests.py` reads recipe lists from `configs/{test_folder}/` for the given scope, reads each recipe's `ci:` section from the YAML under `examples/`, and outputs a CI pipeline YAML with one job per recipe.

**Scopes:**
- **nightly** -- Recipes listed in `nightly_recipes.yml`
- **convergence** -- Recipes in `convergence_recipes.yml`, time automatically doubled
- **release** -- All recipe YAMLs found under `examples/{test_folder}/`

**Stage assignment** is based on recipe type and configuration:

| Stage | Criteria |
|-------|----------|
| `sft` / `peft` | No `checkpoint_robustness` |
| `sft_ckpt_robustness` / `peft_ckpt_robustness` | Has `checkpoint_robustness` |
| `sft_vllm_deploy` / `peft_vllm_deploy` | Has `vllm_deploy: true` |
| `benchmark` | Filename contains `benchmark` |

SFT vs PEFT is determined by whether `peft` appears in the recipe filename.

## Recipe CI Configuration

Each recipe YAML under `examples/` has an optional `ci:` section:

```yaml
ci:
  recipe_owner: username          # Required. Maintainer's handle
  time: "00:25:00"                # Required. SLURM wall time (HH:MM:SS)
  nodes: 2                        # Optional. SLURM node count (default: 1)
  node_multiplier: true           # Optional. Dynamic node scaling
  local_batch_size: 2             # Optional. Override batch size for CI
  nproc_per_node: 1               # Optional. GPUs per node, overrides cluster default (CI var: CONFIG_NPROC_PER_NODE)
  vllm_deploy: true               # Optional. Enable vLLM deployment test
  checkpoint_robustness:          # Optional. Enable robustness testing
    hf_kl_threshold: 1e-3
    tokenizer_name: org/model
    no_check_resume: true         # Skip phase 6 (training resumption)
    # See checkpoint robustness section for all options
```

## Checkpoint Robustness

When `checkpoint_robustness` is present, the robustness test runs after the finetune under the same SLURM allocation. It trains for 5 steps, saves a checkpoint, then validates through:

1. **Reference logits** -- Capture logits before teardown
2. **AutoModel reload** -- Reload from consolidated checkpoint, verify KL = 0
3. **HF reload** -- Load into vanilla `transformers`/`peft`, verify KL below `hf_kl_threshold`
4. **Cross-TP** (optional) -- Reload with different `tp_size`
5. **Training resumption** (on by default) -- Baseline + resumed run, verify loss continuity

Phase 5 is the most expensive (two additional training passes). Use `no_check_resume: true` to skip it.

`ci.time` must cover both finetune and robustness. Estimated overhead:
- ~30% with `no_check_resume: true`
- ~50-60% with resumption check (default)

## How To

### Add a New Recipe to Nightly

1. Create recipe YAML under `examples/{test_folder}/{model_family}/`
2. Add `ci:` section with `recipe_owner` and `time`
3. Add the path to `configs/{test_folder}/nightly_recipes.yml`

### Enable Checkpoint Robustness

1. Add `checkpoint_robustness:` under `ci:` with at least `hf_kl_threshold` and `tokenizer_name`
2. Increase `ci.time` per the guidelines below
3. For large models, consider `no_check_resume: true`

### Enable vLLM Deploy

1. Add `vllm_deploy: true` under `ci:`
2. Robustness must also be enabled (vLLM test loads from the robustness checkpoint)

### Add a New Test Folder

1. Create `examples/{new_folder}/` with recipe YAMLs
2. Create `configs/{new_folder}/` with `nightly_recipes.yml`, `convergence_recipes.yml`, `override_recipes.yml`
3. Create `golden_values/{new_folder}/`
4. Add a CI job template for the new folder in the CI template file
5. Verify with `generate_ci_tests.py --test-folder {new_folder} --scope nightly`

### Exempt a Recipe

Edit `configs/{test_folder}/override_recipes.yml`:

```yaml
exempt_models:
  - model_family           # Skips all recipes under this folder

exempt_configs:
  config_stem:
    reason: "Description, PIC: @owner, issue#"

known_issue:
  - config_stem            # allow_failure instead of blocking
```

## Time Allocation Guidelines

`ci.time` covers the entire SLURM job: finetune, robustness (if enabled), model downloads, setup, and teardown.

| Model Size | Finetune Only | Robustness (`no_check_resume`) | Robustness (full) |
|------------|---------------|--------------------------------|-------------------|
| < 2B | 10 min | 15 min | 15 min |
| 2-5B | 12 min | 15 min | 20 min |
| 5-10B | 18 min | 25 min | 25-30 min |
| 10-20B | 22 min | 30 min | 35 min |
| 20-50B | 35 min | 45 min | 45 min |
| 50B+ | 50 min | 60 min | 60 min |

MoE models, multi-node jobs, and convergence scope (auto 2x) may need additional time. vLLM deploy runs as a separate job and does not consume finetune time.
