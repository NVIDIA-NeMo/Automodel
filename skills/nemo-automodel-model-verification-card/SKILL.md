---
name: nemo-automodel-model-verification-card
description: Create or update agent-readable NeMo AutoModel model verification cards beside model examples. Use when defining a model's verification inventory, recording verified training or deployment evidence, adding a model card YAML, comparing a bounded run with a reference framework, or validating card status and metric completeness.
---

# NeMo AutoModel Model Verification Cards

Create one compact YAML card that shows what has been verified for a model, what has not, and how each verified result can be reproduced. Keep the card beside the model's existing example recipes and name it `<model>_verification_card.yaml`.

Read [references/card-schema.md](references/card-schema.md) before creating or changing a card.

## Workflow

1. Locate the model's current example directory and inspect its runnable training, fine-tuning, and benchmark recipes.
2. Read the reference framework's model card and copy only the workload contract that is meaningful in AutoModel. Do not copy framework-specific conversion or checkpoint items.
3. Define every supported card item in `verification_index`. For each hardware target, put every item in exactly one of `verified` or `not_verified`.
4. Add the matching item leaves under `items`, using the same hardware and status.
5. Add or reference a runnable public recipe for each verified training item. Keep private executor configuration, mounts, tokens, cluster paths, job IDs, and internal issue URLs out of the repository.
6. Validate the card, lint the example YAMLs, and run focused tests before review.

## Canonical inventory

Use these item names when the capability applies:

- `pretrain`: bounded real-data pretraining with finite loss and five recorded metrics.
- `sft`: bounded full supervised fine-tuning.
- `sft_long_context`: CoderForge SFT at sequence length 131072. This is a separate convergence cohort from ordinary SFT.
- `peft`: bounded parameter-efficient fine-tuning.
- `checkpoint_resume`: resume the pretrain recipe from a middle checkpoint into a fresh output root and compare the resumed trajectory with the uninterrupted run.
- `vllm_checkpoint_compatibility`: save a consolidated AutoModel checkpoint and prove that vLLM can load and generate from it.
- `pretrain_performance`: tuned performance using the exact public benchmark recipe. Keep this separate from real-data functional pretraining.

Do not add `checkpoint_save_reload` or `checkpoint_hf_compatibility`. The pretrain/resume item covers native checkpoint continuity, and vLLM load is the downstream compatibility gate.

## Verification gates

Mark an item `verified` only after the represented workload completed and its expected result was checked. Submission or configuration parsing is not verification.

For every verified training item, record these five metrics from complete optimizer-step rows:

- `initial_loss`
- `final_loss`
- `last_10_steps_step_time_ms_avg`
- `last_10_steps_model_tflops_per_gpu_avg`
- `last_10_steps_tokens_per_second_per_gpu_avg`

For `checkpoint_resume`, resume directly from the middle pretrain checkpoint into a new output root. Compare every shared post-checkpoint loss using `abs(resumed - reference) <= 1e-6 + 0.01 * abs(reference)`, and require learning rate and processed-token counts to match exactly.

Treat timing, TFLOPS, and tokens/s from a functional convergence run as sanity observations. They do not verify `pretrain_performance`; that item requires a separate tuned run of the public benchmark recipe.

For `sft_long_context`, use `togethercomputer/CoderForge-Preview` and sequence length 131072. Start from the maintained CoderForge example under `examples/long_context_validation/` and adapt it to the model without changing this contract.

For `vllm_checkpoint_compatibility`, load the consolidated checkpoint through vLLM and run deterministic generation. Successful AutoModel reload alone does not satisfy this item.

## Validate

Run:

```bash
uv run python skills/nemo-automodel-model-verification-card/scripts/validate_card.py \
  examples/<domain>/<model>/<model>_verification_card.yaml
```

Then run the example YAML linter and the focused unit tests for the card and YAML discovery logic. Never change a status merely to make validation pass.

## Completion checklist

- Card name ends in `_verification_card.yaml` and sits beside current model examples.
- Every indexed item appears exactly once under `verified` or `not_verified` for each hardware target.
- Item leaves use only `verified` or `not_verified` and match the index.
- Every verified item records the immutable AutoModel commit, date, precision, public command or recipe, and concrete expected result.
- Every verified training leaf includes all five metrics.
- Resume evidence includes all shared steps, the declared tolerance, and exact LR/token checks.
- Long-context SFT uses CoderForge and sequence length 131072.
- Performance is a separate item and uses a public benchmark recipe.
- The card contains no private runtime information or credentials.
- The bundled validator and focused tests pass.
