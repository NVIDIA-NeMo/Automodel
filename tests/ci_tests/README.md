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

Each recipe YAML under `examples/` has a `ci:` section. It is required for
newly added CI recipes, which must declare `recipe_owner`, `time`, and `nodes`
(enforced by `validate_new_recipe_ci.py`); pre-existing recipes are grandfathered:

```yaml
ci:
  recipe_owner: username          # Required. Maintainer's handle
  time: "00:25:00"                # Required. SLURM wall time (HH:MM:SS)
  nodes: 2                        # Required for new recipes. SLURM node count (omitted -> defaults to 1)
  node_multiplier: true           # Optional. Dynamic node scaling
  max_steps: 50                   # Optional. Override max training steps for CI
  local_batch_size: 2             # Optional. Override batch size for CI
  nproc_per_node: 1               # Optional. GPUs per node, overrides cluster default (CI var: CONFIG_NPROC_PER_NODE)
  vllm_deploy: true               # Optional. Enable vLLM deployment test
  checkpoint_robustness:          # Optional. Enable robustness testing
    hf_kl_threshold: 1e-3
    tokenizer_name: org/model
    check_source_load_parity: true  # Optional. Compare raw HF source load vs constructed trainer before training
    hf_device_map_auto: true      # Optional. Use for large HF reference loads that do not fit on one GPU
    no_check_resume: true         # Skip phase 6 (training resumption)
    # See checkpoint robustness section for all options
```

## Checkpoint Robustness

When `checkpoint_robustness` is present, the robustness test runs after the finetune under the same SLURM allocation. It trains for 5 steps, saves a checkpoint, then validates through:

0. **Source-load parity** (optional) -- With `check_source_load_parity: true`, capture logits from the raw HF source load, release the HF model, construct a parity-only trainer model, compare the constructed pre-training model against those HF logits, then release it so training starts from a fresh trainer
1. **Reference logits** -- Capture logits before teardown
2. **AutoModel reload** -- Reload from consolidated checkpoint, verify KL = 0
3. **HF reload** -- Load into vanilla `transformers`/`peft`, verify KL below `hf_kl_threshold`
4. **Cross-TP** (optional) -- Reload with different `tp_size`
5. **Training resumption** (on by default) -- Baseline + resumed run, verify loss continuity

LLM recipes use the causal-LM harness, while `examples/vlm_finetune/` recipes use the VLM finetune recipe and
`AutoModelForImageTextToText`. VLM parity currently exercises the language path with text-only `input_ids`; real-image
multimodal parity is a separate follow-up.

Phase 5 is the most expensive (two additional training passes). Use `no_check_resume: true` to skip it.

Use source-load parity for recipes where the initial HF checkpoint load is itself part of the contract, especially
remote-code, force-HF, custom model, or tied/untied `lm_head` paths. The raw HF reference model is loaded only long
enough to capture logits and is released before the trainer model is constructed.

For large reference models, set `hf_device_map_auto: true` so HF can use `device_map="auto"` instead of placing the
whole reference load on one rank's GPU. This is intentionally opt-in rather than the default: small models should keep
the simpler single-device HF load for deterministic behavior, while large models (for example 9B+ or configs that
already require multi-GPU HF reloads) should enable it to avoid rank-0 OOM. Tune `source_load_kl_threshold` and
`source_load_mean_kl_threshold` only when backend or dtype differences are expected. The first threshold bounds the
worst token, while the stricter mean threshold prevents broad drift; Phase 0 also reports p95 KL for diagnosis.
`source_load_cosine_threshold` remains an independent full-logit check.

`ci.time` must cover both finetune and robustness. Estimated overhead:
- ~30% with `no_check_resume: true`
- ~50-60% with resumption check (default)

## How To

### Add a New Recipe to Nightly

1. Create recipe YAML under `examples/{test_folder}/{model_family}/`
2. Add `ci:` section with `recipe_owner`, `time`, and `nodes` (use `nodes: 1` for single-node)
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
