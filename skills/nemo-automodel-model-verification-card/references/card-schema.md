# Card schema

Cards are YAML mappings with this shape:

```yaml
title: model_slug
summary: >
  Scope and interpretation of the recorded evidence.
model:
  hf_id: organization/model
  hf_revision: immutable_revision
  architecture: ArchitectureName
verification_environment:
  automodel_commit: immutable_commit
verification_index:
  training:
    H100:
      verified: [pretrain, checkpoint_resume]
      not_verified: [sft, sft_long_context, peft]
  deployment:
    H100:
      verified: []
      not_verified: [vllm_checkpoint_compatibility]
  performance:
    H100:
      verified: []
      not_verified: [pretrain_performance]
items:
  pretrain:
    H100:
      status: verified
      precision: bf16
      automodel_commit: immutable_commit
      recipe: examples/path/to/recipe.yaml
      command: >
        public reproducible command
      last_verified: "YYYY-MM-DD"
      metrics:
        initial_loss: 0.0
        final_loss: 0.0
        last_10_steps_step_time_ms_avg: 0.0
        last_10_steps_model_tflops_per_gpu_avg: 0.0
        last_10_steps_tokens_per_second_per_gpu_avg: 0.0
      expected_result: >
        Concrete checked outcome.
```

## Status rules

`verified` and `not_verified` are the only statuses. The index contains both lists for every category/hardware pair, even when one is empty. Each item/hardware pair appears in exactly one list and has one matching leaf under `items`.

Do not encode required/optional policy. A card reports current evidence, not prioritization.

## Verified leaves

A verified leaf contains:

- `precision`
- `automodel_commit`
- `recipe` or `command`
- `last_verified`
- `expected_result`

Verified leaves in `training` or `performance` also contain the five standard `metrics` fields. `checkpoint_resume` additionally contains:

```yaml
resume_comparison:
  shared_steps: 50
  passed_steps: 50
  loss_tolerance: 1.0e-6 + 1% of abs(reference)
  learning_rate_exact: true
  consumed_tokens_exact: true
```

## Unverified leaves

An unverified leaf needs only `status: not_verified`, but should state a reproducible `verification_contract` or `next_step` when that prevents ambiguity. Do not add placeholder metrics.

## Paths and evidence

Repository paths are relative to the repository root. Commands may use portable container paths such as `/work/data` when the recipe documents the required mounts. Never record absolute host experiment paths, private cluster identifiers, Slurm IDs, credentials, or internal tracker URLs in a public card.
