# CI Tests

Configuration, scripts, and utilities for AutoModel's CI recipe validation pipeline.

## Directory Structure

```
ci_tests/
  configs/{test_folder}/
    nightly_recipes.yml         # Recipes included in nightly scope
    release_recipes.yml         # Explicit release list for non-auto-discovered folders
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
- **release** -- All recipe YAMLs in auto-discovered folders, or recipes listed
  in `release_recipes.yml` for explicitly managed folders such as `llm_pretrain`

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
  env_vars:                       # Optional. Environment variables forwarded to the job
    REQUIRE_FINITE_METRICS: "true" # Fail when no step metrics are logged or loss/grad_norm is non-finite
  vllm_deploy: true               # Optional. Enable vLLM deployment test
  vllm_deploy_time: "00:30:00"    # Optional. Override the vLLM deploy SLURM wall time (defaults to 00:10:00)
  checkpoint_robustness:          # Optional. Enable robustness testing
    tokenizer_name: org/model
    parity_sequence_length: 2048  # Optional. Full-logit parity prompt length (default: 2048; 1K-4K recommended)
    # cross_framework_gate_sequence_length: 128  # Exceptional: gate Phases 0/3 on this prefix of the same forward
    # capture_router_diagnostics: true  # GLM-only Phase 0 router evidence; disabled by default
    # shape_diagnostic: {enabled: true, sweep_lengths: [128, 256, 512, 1024]}  # Informational HF shape probe
    parity_tolerance_profile: standard  # Optional: strict, standard (default), or relaxed
    hf_device_map_auto: true      # Optional. Use for large HF reference loads that do not fit on one GPU
    # skip_resume: true           # Exceptional: skip native-checkpoint resume (Phase 4)
    # See checkpoint robustness section for all options
```

## Checkpoint Robustness

When `checkpoint_robustness` is present, the robustness test runs after the finetune under the same SLURM allocation.
LLM and VLM tests run each lifecycle phase in a fresh process by default so the test models a real restart and does not
depend on Python object teardown. `process_isolation: false` retains the old single-process path as a compatibility
fallback.

The public phase model is deliberately numbered 0 through 5. The two isolated jobs that produce and consume the
source reference are implementation details of Phase 0, not separate phases.

| Phase | Default | Operation | Blocking oracle |
|-------|---------|-----------|-----------------|
| 0. Source parity | Yes | Compare the original vanilla-HF source checkpoint with a freshly constructed AutoModel before training. | Full-logit parity plus tied-input/output-embedding alias checks. |
| 1. Train, save, reference | Yes | Train for the configured short trajectory, save the checkpoint, and capture finite reference logits from the trained model. | Training and checkpoint publication complete; reference logits are finite. When Phase 4 is enabled, boundary state and continuation artifacts are also captured. |
| 2. AutoModel model reload | Yes | Reload the saved model payload through AutoModel. Dense models use the exported HF-format consolidated weights. PEFT models restore the AutoModel checkpoint payload. | Full-logit parity. PEFT additionally requires exact trainable-adapter fingerprints. |
| 3. Vanilla-HF model reload | Yes | Load the same exported dense weights with `transformers`, or the exported adapter with `peft`, and run a forward pass. | Full-logit parity. PEFT additionally requires exact saved-adapter tensor fingerprints. |
| 4. Native training resume | Yes | Restore the native distributed checkpoint at the Phase 1 boundary, including model, optimizer, scheduler, RNG, and data state, then replay identical batches. | Exact restored state and pre-update fingerprints, followed by the configured shared-trajectory loss envelope. This phase does not use KL. |
| 5. Cross-TP reload | No | Reload the exported dense weights with `cross_tp_size`. | Full-logit parity against the Phase 1 reference. |

Phases 0–4 are the core lifecycle and are enabled by default. Phase 5 is an optional topology-portability test enabled
by setting `cross_tp_size`. A phase should be skipped only for a documented incompatibility; see the skip controls below.

LLM recipes use the causal-LM harness, while `examples/vlm_finetune/` recipes use the VLM finetune recipe and
`AutoModelForImageTextToText`. VLM parity currently exercises the language path with text-only `input_ids`; real-image
multimodal parity is a separate follow-up.

The resume oracle is deliberately distinct from independent-run reproducibility. It checks scheduler position,
complete optimizer parameter-group settings, LR and weight decay, RNG state, and per-rank batch identity. On the first
shared step, after both branches have entered the same FSDP unshard/reshard lifecycle but before the optimizer update,
it requires exact rank-local model parameters and optimizer tensors. Persistent buffers and gradients at that point,
plus post-update model/optimizer fingerprints, are recorded diagnostically so a numerical divergence can be localized.
Exact loss deltas and diagnostic comparisons are written for successful and failed runs.

After the native checkpoint has been restored, Phase 4 disables further checkpoint writes: the continuation is an
oracle for restored state and training trajectory, and its final checkpoint is not consumed by any later phase.

Select the Phase 4 shared scale-aware loss envelope with `resume_tolerance_profile`. Each stage allows
`atol + rtol * max(abs(uninterrupted_loss), abs(resumed_loss))`: `strict` uses `1e-6 + 0%` for both stages;
`standard` (default) uses `1e-5 + 0.2%` for the first step and `5e-3 + 0.2%` later; `relaxed` uses
`1e-4 + 0.75%` first and `1e-2 + 0.75%` later. Prefer profiles over model-specific calibration, and use `relaxed`
only for demonstrated distributed or low-precision drift after the exact state gates pass. `resume_first_loss_threshold`
and `resume_loss_threshold` remain authoritative absolute-only overrides for exceptional cases. Use
`skip_resume: true` only for an explicitly documented restore blocker.

CI also reuses the normal finetune that already precedes checkpoint robustness as a separate, non-blocking training-
reproducibility metric; it does not launch another baseline. Normal finetune and checkpoint Phase 1 record per-rank
batch digests, loss, and LR. They are compared only when component fingerprints match for model initialization and
seed, dataset/dataloader ordering, batch sizes and topology, optimizer, LR scheduler, loss, and backend configuration.
Otherwise the log reports `not_comparable` and names the mismatched components. Loss differences use the separately
calibrated `training_reproducibility_loss_threshold` (default `5e-2`). Exceeding that envelope remains non-blocking but
emits a prominent `ALERT` and saves a machine-readable `report.json` in the reproducibility artifact directory. This is
an opportunistic diagnostic rather than required coverage: phase-specific overrides may make the two existing runs
incomparable, while the shared-trajectory resume check remains the blocking reproducibility oracle.

Phase 0 makes the initial HF checkpoint load part of the default contract. The raw HF reference model is loaded only
long enough to capture logits and is released before the trainer model is constructed. This catches remote-code,
force-HF, custom-model, and tied/untied `lm_head` regressions before training can obscure them.

The AutoModel side always keeps the recipe's configured attention backend. The independent vanilla-HF reference uses
that backend when the pinned Transformers model declares support for it; otherwise it uses `eager` and logs an
attention-compatibility fallback. This preserves a working HF reference instead of making a recipe backend that HF
cannot execute look like a checkpoint failure.

For large reference models, set `hf_device_map_auto: true` so HF can use `device_map="auto"` instead of placing the
whole reference load on one rank's GPU. This remains opt-in: small models keep the simpler single-device HF load,
while large models (for example 9B+ or configs that already require multi-GPU HF reloads) should enable it to avoid
rank-0 OOM.

The other ranks wait up to 1,800 seconds for the rank-0-only vanilla-HF reload by default. Set
`hf_reload_timeout_seconds` only when a documented large or CPU-offloaded reference can legitimately take longer;
this changes the synchronization timeout, not any numerical gate.

### Full-Logit Metrics and Profiles

Phases 0, 2, 3, and 5 compare every vocabulary logit for every prompt token. A version-controlled snapshot of the
long-form finetuning guide is tokenized with each model's tokenizer, then truncated to `parity_sequence_length` tokens
(default 2048; 1K-4K is the recommended range). The snapshot contains more than 6,000 words and is protected by a
checked SHA-256 digest, so a 4K test uses stable, unique document content rather than a repeated short prompt. The
harness fails with an actionable error instead of repeating content if a requested length exceeds the tokenized
document. Pipeline-parallel runs resize their stage activation buffers to the configured parity length; reduce the
length only when a model has a documented memory limit.

For a diagnosed cross-framework instability, `cross_framework_gate_sequence_length` can gate Phases 0 and 3 on a
prefix of the same long forward. It does not launch a shorter forward: the harness still runs and reports all
`parity_sequence_length` tokens, while mean KL, p95 KL, and cosine use the configured prefix for pass/fail. Phase 2,
Phase 5, repeatability records, and native resume keep their full-sequence behavior. Use this only when evidence has
localized the full-sequence tail to a framework-sensitive numerical path, such as near-tied routed-MoE decisions;
do not use it as a general substitute for fixing a model or selecting the appropriate tolerance profile.

`capture_router_diagnostics: true` additionally records the full-sequence vanilla-HF and AutoModel router logits,
correction biases, and actual expert selections during Phase 0 for GLM 4.7 Flash. Its concise CI summary focuses on
layers 1-5: route-flip context, large-margin flips outside the measured score-noise band, and whether replacement
experts systematically follow correction bias. Flip counts never gate because a stable MoE can flip many near ties.
The full report also groups final-token KL by flipped-layer count, separating the zero-flipped-layer empirical floor
from the routed tail. Raw captures and JSON evidence live under `.checkpoint_robustness/router_diagnostics/` and are
embedded in the schema-v3 Phase 0 parity record. The option is model-specific and off by default because it retains
router tensors until the forward finishes; unsupported model families fail explicitly when it is enabled.

`shape_diagnostic.enabled: true` measures the vanilla-HF reference against itself when only forward sequence shape
changes. If `cross_framework_gate_sequence_length` is shorter than `parity_sequence_length`, the harness adds one
standing forward and reports `KL(HF_full[:gate_length] || HF_gate_length)` beside the cross-framework gate. Optional
`sweep_lengths` add onboarding or kernel-upgrade calibration points, all using prefixes of the same SHA-checked fixed
document. With router capture enabled, each point also reports self-flip counts and the final-token mean KL for tokens
that flip in at least 11 layers; that amplification cost, not raw flip frequency, distinguishes sensitive GLM behavior
from stable routed MoEs such as Qwen. Bitwise-zero points are labeled `same_kernel_regime_not_probing`, not evidence of
general shape stability. Shape diagnostics are informational, print one concise CI line, and store full evidence under
`.checkpoint_robustness/shape_diagnostics/` and in the schema-v3 Phase 0 parity record. Leave `sweep_lengths` empty for
the cheap standing check; calibration sweeps add one vanilla-HF forward per unique length.

Every comparison reports mean, p95, and max per-token `KL(reference || candidate)`; mean, p95, and max per-token
Jensen-Shannon divergence (natural log, bounded by `ln(2)`); whole-tensor cosine similarity; and mean/max absolute
logit difference. The full record is printed as `CHECKPOINT_PARITY_METRICS <json>` and saved under
`<checkpoint_dir>/.checkpoint_robustness/parity_metrics/`. Named profiles gate mean KL, p95 KL, and cosine similarity.
JSD, max KL, and absolute logit differences remain diagnostics, allowing their usefulness to be evaluated without
changing pass/fail policy. Record schema version 3 keeps `metrics` as the full-sequence record and adds `gate_metrics`,
`gate_sequence_length`, and explicit full-sequence failure diagnostics. Schema version 2 introduced `mean_jsd`,
`p95_jsd`, and `max_jsd`; those metric fields retain their meaning.

Each vanilla-HF reference is forwarded twice through the same loaded model. The resulting `hf_source_self_repeat`
or `hf_export_self_repeat` record is informational and distinguishes cross-framework drift from an unstable reference.
Phase 1 likewise always emits an informational `automodel_reference_self_repeat` record. Phase 2 emits
`automodel_reload_self_repeat` for the `relaxed` profile, an informational reload gate, or a reload
comparison that exceeds its active thresholds. This keeps the dense passing path to one additional AutoModel forward
while capturing both sides of the repeatability diagnosis for sensitive or failing configurations.

| Self-repeat record | What it measures |
|--------------------|------------------|
| `hf_source_self_repeat` | Repeatability of the original loaded HF checkpoint. |
| `automodel_reference_self_repeat` | Repeatability of the trained Phase 1 AutoModel reference. |
| `automodel_reload_self_repeat` | Repeatability of the independently reloaded Phase 2 AutoModel. |
| `hf_export_self_repeat` | Repeatability of the exported checkpoint reloaded in vanilla HF. |

All self-repeat records have `enforced: false`: they cannot fail the job, select a profile, or change an active
threshold. Use their logged JSON metrics for offline diagnosis and profile calibration. If a primary reload comparison
is large while both relevant self-repeat comparisons are small, investigate checkpoint/load correctness. If a
self-repeat comparison is already large, the model or reference execution is itself numerically variable and the
primary comparison includes that variability.

| Profile | Same implementation mean / p95 / cosine | Cross-framework mean / p95 / cosine | Cross-topology mean / p95 / cosine |
|---------|-----------------------------------------|---------------------------------------|-------------------------------------|
| `strict` | `1e-7` / `1e-6` / `0.999999` | `1e-4` / `1e-3` / `0.9999` | `1e-6` / `1e-5` / `0.99999` |
| `standard` (default) | `3e-3` / `1.2e-2` / `0.999` | `6e-3` / `3e-2` / `0.998` | `6e-3` / `3e-2` / `0.998` |
| `relaxed` | `2e-2` / `5e-2` / `0.995` | `2.5e-2` / `1e-1` / `0.99` | `2e-2` / `5e-2` / `0.995` |
Use `strict` for deterministic same-kernel paths and `standard` for dense models and numerically stable MoE paths.
Reserve `relaxed` for demonstrated discontinuous distributed behavior, normally expert-parallel MoE routing, and
document the evidence in the recipe. Model size, TP/PP, or MoE status alone does not justify it. A dense model that
exceeds `standard` should be investigated. `parity_tolerance_profile` is the fallback for every comparison. If only
one verified comparison needs another shared profile after exact checkpoint-state and within-process repeatability
checks pass, select it with `parity_tolerance_profile_overrides`; every unspecified comparison retains the fallback.
Use `parity_threshold_overrides` only when that comparison also exceeds the closest shared profile, and override only
the necessary gate. Every unspecified metric remains inherited from the active comparison profile.
The comparison class is selected by the harness. For every profile, a cross-topology comparison is never stricter
than the same-implementation comparison because changing topology adds a numerical variation source.

```yaml
parity_tolerance_profile: standard
parity_tolerance_profile_overrides:
  hf_reload: relaxed
```

The global `standard` line above is optional because it is the default. Supported comparison names are `source_load`,
`automodel_reload`, `hf_reload`, and `cross_tp`. Phase 4 uses the separate `resume_tolerance_profile` because it gates
the restored training loss trajectory rather than full-logit metrics.

For a measured exception that exceeds even its selected comparison profile:

```yaml
parity_tolerance_profile: relaxed
parity_threshold_overrides:
  automodel_reload:
    mean_kl: 0.04
    cosine_similarity: 0.99
```

Supported metric names are `mean_kl`, `p95_kl`, and `cosine_similarity`. Numeric overrides are exceptional calibration
escape hatches, not additional profiles. JSD and max KL remain diagnostic and cannot be overridden.

Legacy positive `check_*` controls, generic numeric cosine fields, and max-KL threshold fields are no longer accepted.
All live recipes use default-on phases, semantic `skip_*` controls, and named profiles. The optional structured
profile and numeric override mappings remain available for measured one-model exceptions.

Retrieval checkpoint robustness uses the same phase contract for Phases 1–4. Because a biencoder produces embeddings
rather than language-model logits, its Phase 2 AutoModel reload gates the selected profile's same-implementation
cosine threshold, and its Phase 3 vanilla-HF reload gates the cross-framework cosine threshold. KL gates do not apply
to embedding outputs. Retrieval profile and numeric override mappings therefore support only `automodel_reload` and
`hf_reload`; retrieval currently has no Phase 0 source-load or Phase 5 cross-TP comparison.

### Phase Controls

| Field | Effect |
|-------|--------|
| `skip_source_load_parity: true` | Skip all of Phase 0. |
| `skip_source_load_logit_parity: true` | Keep the Phase 0 HF load/forward smoke and report full metrics, but make source-vs-AutoModel logit parity informational. |
| `skip_automodel_reload_logit_parity: true` | Keep the Phase 2 reload, forward smoke, and PEFT fingerprints, but make its logit metrics informational. |
| `skip_hf_reload: true` | Skip all of Phase 3, including its load and forward smoke. |
| `skip_hf_reload_logit_parity: true` | Keep the Phase 3 load, forward smoke, and PEFT fingerprints, but make its logit metrics informational. |
| `skip_resume: true` | Skip Phase 4. |
| `cross_tp_size: N` | Enable Phase 5 with tensor-parallel size `N` for dense models. |
| `process_isolation: false` | Use the legacy single-process lifecycle as a compatibility fallback. |

Removed fields map to the current contract as follows: omit `check_source_load_parity`, `check_hf_reload`, and
`check_resume` to keep their phases enabled; use `skip_source_load_parity`, `skip_hf_reload`, or `skip_resume` to
disable one. Use `skip_automodel_reload_logit_parity` and `skip_hf_reload_logit_parity` instead of their shorter legacy
aliases. Replace generic or phase-specific legacy KL/cosine thresholds with `parity_tolerance_profile` and, when only
one comparison needs a different shared profile, `parity_tolerance_profile_overrides`. Use
`parity_threshold_overrides` only for a measured exception that does not fit a shared profile.

`ci.time` must cover both finetune and robustness. Resume adds one short restored continuation; it no longer launches a
separate fresh baseline.

## How To

### Add a New Recipe to Nightly

1. Create recipe YAML under `examples/{test_folder}/{model_family}/`
2. Add `ci:` section with `recipe_owner`, `time`, and `nodes` (use `nodes: 1` for single-node)
3. Add the path to `configs/{test_folder}/nightly_recipes.yml`

### Enable Checkpoint Robustness

1. Add `checkpoint_robustness:` under `ci:` and set `tokenizer_name` when the model does not use the default Llama tokenizer IDs
2. Increase `ci.time` per the guidelines below
3. For large vanilla-HF loads, consider `hf_device_map_auto: true`; add a `skip_*` field only with a documented blocker

### Enable vLLM Deploy

1. Add `vllm_deploy: true` under `ci:`
2. Robustness must also be enabled (vLLM test loads from the robustness checkpoint)
3. For large models that need more than 10 minutes to load, set `vllm_deploy_time`

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

| Model Size | Finetune Only | Robustness (`skip_resume`) | Robustness (full) |
|------------|---------------|--------------------------------|-------------------|
| < 2B | 10 min | 15 min | 15 min |
| 2-5B | 12 min | 15 min | 20 min |
| 5-10B | 18 min | 25 min | 25-30 min |
| 10-20B | 22 min | 30 min | 35 min |
| 20-50B | 35 min | 45 min | 45 min |
| 50B+ | 50 min | 60 min | 60 min |

MoE models, multi-node jobs, and convergence scope (auto 2x) may need additional time. vLLM deploy runs as a separate job and does not consume finetune time.
