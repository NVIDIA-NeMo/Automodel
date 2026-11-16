# Legacy LLM benchmarks and canonical performance testing

The standalone synthetic recipes are retained while they migrate to the
[`ci.perf_testing` system in PR #4637](https://github.com/NVIDIA-NeMo/Automodel/pull/4637).
Make runtime tuning changes in the canonical training recipe. Performance settings
allow scheduling, walltime, and optional booleans `fake_balanced_gate` and `validation`; they do
not permit another copy of the model, data, batches, parallelism or optimizer.
Balanced-gate results must be labeled: they remove routing imbalance and are not
measurements of learned routing.

To benchmark a mapped canonical recipe directly, add `--benchmark`:

```bash
automodel examples/llm_pretrain/qwen/qwen3_moe_30b_te_deepep.yaml \
  --benchmark --nproc-per-node 8
```

The flag applies `ci.perf_testing.fake_balanced_gate` when specified, disables
checkpoint saving, and defaults to 50 steps with no epoch limit. Validation is
disabled unless `ci.perf_testing.validation: true` retains the recipe's cadence.
Explicit CLI settings take precedence; for example, add
`--step_scheduler.max_steps 100` for a longer run. Model, dataset, batch,
parallelism and optimizer settings continue to come from the canonical YAML.
The `schedule` and `time` metadata do not submit or resize a local job.

The same flag works with external `torchrun`, including Slurm jobs:

```bash
torchrun --standalone --nproc-per-node=8 -m nemo_automodel.cli.app \
  examples/llm_pretrain/qwen/qwen3_moe_30b_te_deepep.yaml --benchmark
```

Performance CI skips validation by default. Set `ci.perf_testing.validation: true`
to retain the canonical recipe's periodic and checkpoint-triggered validation
cadence. Checkpoint saving remains disabled for performance runs.

## Fixed-length synthetic pretraining

The 56 full-training benchmarks below have `llm_pretrain` counterparts using the
existing `MockIterableDatasetConfig`. Random token IDs span the model vocabulary,
excluding padding IDs. Fixed sequence lengths match the legacy benchmarks. Labels
shift within each sequence, with the final label ignored. There is no THD packing;
the model builds canonical positions. The default collator joins the generator's
single-row tensors using `step_scheduler.local_batch_size`.

These synthetic workloads measure runtime performance; they do not establish
learning quality or convergence on real text.

The canonical configs contain the benchmark's model initialization, precision,
kernels, loss, optimizer, parallelism and global/local batches. Their normal route
uses real gates; only `ci.perf_testing.fake_balanced_gate` enables balanced gates.
Kimi and Step-5 keep synthetic-only `benchmark_static_routing` disabled in both
normal and performance runs because it cannot represent real routing. This is a
recorded difference from the two Kimi and one Step-5 legacy synthetic runs.

The legacy `qwen/qwen3_moe_30b_te_fp8.yaml` and `moonlight/moonlight_16b_te_fp8.yaml`
put activation checkpointing and resharding in a top-level `parallelizer` section,
which no recipe reads. The Qwen3 FP8 counterpart sets
`distributed.activation_checkpointing: true`, which the legacy run requested but
never applied; that legacy run ran out of memory without it. The Moonlight FP8
counterpart omits the section because its values equal the defaults.

Run a config directly with the resources in its `ci` section, for example:

```bash
torchrun --standalone --nproc-per-node=8 -m nemo_automodel.recipes.llm.train_ft \
  --config examples/llm_pretrain/qwen/qwen3_moe_30b_te_deepep.yaml
```

Use the same config for performance testing. Mock data requires no dataset download
or tokenization. Model initialization may still require Hugging Face access.
`max_steps` bounds the run. Use the existing local-shard Megatron or NanoGPT
examples with prepared real data for production pretraining.

GB200 configs use four GPUs per node and need an ARM64 image on a GB200 cluster;
H100 configs use eight. Deprecated variants remain disabled. Existing known-issue
and unavailable-kernel exclusions still apply. New configuration coverage is not
GPU validation; publish the exact source revision and terminal job results.

| Retained benchmark | Canonical pretraining config | Sequence length |
|---|---|---:|
| [deepseek/deepseek_v3_sdpa_compile_gb200.yaml](deepseek/deepseek_v3_sdpa_compile_gb200.yaml) | [deepseekv3_pretrain_sdpa_compile_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_sdpa_compile_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_sdpa_compile_mxfp8_gb200.yaml](deepseek/deepseek_v3_sdpa_compile_mxfp8_gb200.yaml) | [deepseekv3_pretrain_sdpa_compile_mxfp8_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_sdpa_compile_mxfp8_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_sdpa_compile_te_gb200.yaml](deepseek/deepseek_v3_sdpa_compile_te_gb200.yaml) | [deepseekv3_pretrain_sdpa_compile_te_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_sdpa_compile_te_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_sdpa_compile_torchmm_gb200.yaml](deepseek/deepseek_v3_sdpa_compile_torchmm_gb200.yaml) | [deepseekv3_pretrain_sdpa_compile_torchmm_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_sdpa_compile_torchmm_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_te_deepep.yaml](deepseek/deepseek_v3_te_deepep.yaml) | [deepseek/deepseek_v3_te_deepep.yaml](../llm_pretrain/deepseek/deepseek_v3_te_deepep.yaml) | 4096 |
| [deepseek/deepseek_v3_te_deepep_1024.yaml](deepseek/deepseek_v3_te_deepep_1024.yaml) | [deepseek/deepseek_v3_te_deepep_1024.yaml](../llm_pretrain/deepseek/deepseek_v3_te_deepep_1024.yaml) | 4096 |
| [deepseek/deepseek_v3_te_deepep_gb200.yaml](deepseek/deepseek_v3_te_deepep_gb200.yaml) | [deepseekv3_pretrain_te_deepep_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_te_deepep_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_te_mxfp8_gb200.yaml](deepseek/deepseek_v3_te_mxfp8_gb200.yaml) | [deepseekv3_pretrain_te_mxfp8_gb200.yaml](../llm_pretrain/deepseekv3_pretrain_te_mxfp8_gb200.yaml) | 4096 |
| [deepseek/deepseek_v3_torch.yaml](deepseek/deepseek_v3_torch.yaml) | [deepseek/deepseek_v3_torch.yaml](../llm_pretrain/deepseek/deepseek_v3_torch.yaml) | 4096 |
| [deepseek/deepseek_v4_flash_h100_optimized.yaml](deepseek/deepseek_v4_flash_h100_optimized.yaml) | [deepseek/deepseek_v4_flash_h100_optimized.yaml](../llm_pretrain/deepseek/deepseek_v4_flash_h100_optimized.yaml) | 4096 |
| [deepseek/deepseek_v4_flash_mok_32gpu.yaml](deepseek/deepseek_v4_flash_mok_32gpu.yaml) | [deepseek/deepseek_v4_flash_mok_32gpu.yaml](../llm_pretrain/deepseek/deepseek_v4_flash_mok_32gpu.yaml) | 1024 |
| [glm/glm_4.5_air_te_deepep.yaml](glm/glm_4.5_air_te_deepep.yaml) | [glm/glm_4.5_air_te_deepep.yaml](../llm_pretrain/glm/glm_4.5_air_te_deepep.yaml) | 4096 |
| [glm/glm_4.7_flash_mok_4gpu.yaml](glm/glm_4.7_flash_mok_4gpu.yaml) | [glm/glm_4.7_flash_mok_4gpu.yaml](../llm_pretrain/glm/glm_4.7_flash_mok_4gpu.yaml) | 4096 |
| [glm/glm_4.7_flash_te_deepep.yaml](glm/glm_4.7_flash_te_deepep.yaml) | [glm/glm_4.7_flash_te_deepep.yaml](../llm_pretrain/glm/glm_4.7_flash_te_deepep.yaml) | 4096 |
| [glm/glm_4.7_te_deepep.yaml](glm/glm_4.7_te_deepep.yaml) | [glm/glm_4.7_te_deepep.yaml](../llm_pretrain/glm/glm_4.7_te_deepep.yaml) | 4096 |
| [glm/glm_5_te_deepep.yaml](glm/glm_5_te_deepep.yaml) | [glm/glm_5_te_deepep.yaml](../llm_pretrain/glm/glm_5_te_deepep.yaml) | 4096 |
| [gpt_oss/gptoss_120b_te_deepep.yaml](gpt_oss/gptoss_120b_te_deepep.yaml) | [gpt_oss/gptoss_120b_te_deepep.yaml](../llm_pretrain/gpt_oss/gptoss_120b_te_deepep.yaml) | 4096 |
| [gpt_oss/gptoss_120b_te_deepep_gb200.yaml](gpt_oss/gptoss_120b_te_deepep_gb200.yaml) | [gpt_oss/gptoss_120b_te_deepep_gb200.yaml](../llm_pretrain/gpt_oss/gptoss_120b_te_deepep_gb200.yaml) | 4096 |
| [gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml](gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml) | [gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml](../llm_pretrain/gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml) | 4096 |
| [gpt_oss/gptoss_20b_te_deepep.yaml](gpt_oss/gptoss_20b_te_deepep.yaml) | [gpt_oss/gptoss_20b_te_deepep.yaml](../llm_pretrain/gpt_oss/gptoss_20b_te_deepep.yaml) | 4096 |
| [gpt_oss/gptoss_20b_torch.yaml](gpt_oss/gptoss_20b_torch.yaml) | [gpt_oss/gptoss_20b_torch.yaml](../llm_pretrain/gpt_oss/gptoss_20b_torch.yaml) | 4096 |
| [kimi/kimi_k2_te_deepep.yaml](kimi/kimi_k2_te_deepep.yaml) | [kimi/kimi_k2_te_deepep.yaml](../llm_pretrain/kimi/kimi_k2_te_deepep.yaml) | 4096 |
| [kimi/kimi_k3_gb200.yaml](kimi/kimi_k3_gb200.yaml) | [kimi/kimi_k3_gb200.yaml](../llm_pretrain/kimi/kimi_k3_gb200.yaml) | 1024 |
| [kimi/kimi_k3_gb200_2k.yaml](kimi/kimi_k3_gb200_2k.yaml) | [kimi/kimi_k3_gb200_2k.yaml](../llm_pretrain/kimi/kimi_k3_gb200_2k.yaml) | 2048 |
| [llama3_1/llama3_1_8b_quack_rope.yaml](llama3_1/llama3_1_8b_quack_rope.yaml) | [llama3_1/llama3_1_8b_quack_rope.yaml](../llm_pretrain/llama3_1/llama3_1_8b_quack_rope.yaml) | 4096 |
| [llama3_1/llama3_1_8b_quack_rope_baseline.yaml](llama3_1/llama3_1_8b_quack_rope_baseline.yaml) | [llama3_1/llama3_1_8b_quack_rope_baseline.yaml](../llm_pretrain/llama3_1/llama3_1_8b_quack_rope_baseline.yaml) | 4096 |
| [llama3_3/custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml](llama3_3/custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml) | [llama3_3/custom_llama3_1_70b_pretrain_8nodes.yaml](../llm_pretrain/llama3_3/custom_llama3_1_70b_pretrain_8nodes.yaml) | 8192 |
| [minimax/minimax_m2.5_te_deepep.yaml](minimax/minimax_m2.5_te_deepep.yaml) | [minimax/minimax_m2.5_te_deepep.yaml](../llm_pretrain/minimax/minimax_m2.5_te_deepep.yaml) | 4096 |
| [mistral/mistral_large4_gb200.yaml](mistral/mistral_large4_gb200.yaml) | [mistral/mistral_large4_gb200.yaml](../llm_pretrain/mistral/mistral_large4_gb200.yaml) | 4096 |
| [mistral/mistral_small_4_te_deepep.yaml](mistral/mistral_small_4_te_deepep.yaml) | [mistral/mistral_small_4_te_deepep.yaml](../llm_pretrain/mistral/mistral_small_4_te_deepep.yaml) | 4096 |
| [moonlight/moonlight_16b_te_deepep.yaml](moonlight/moonlight_16b_te_deepep.yaml) | [moonlight/moonlight_16b_te_deepep.yaml](../llm_pretrain/moonlight/moonlight_16b_te_deepep.yaml) | 2048 |
| [moonlight/moonlight_16b_te_fp8.yaml](moonlight/moonlight_16b_te_fp8.yaml) | [moonlight/moonlight_16b_te_fp8.yaml](../llm_pretrain/moonlight/moonlight_16b_te_fp8.yaml) | 2048 |
| [moonlight/moonlight_16b_torch.yaml](moonlight/moonlight_16b_torch.yaml) | [moonlight/moonlight_16b_torch.yaml](../llm_pretrain/moonlight/moonlight_16b_torch.yaml) | 2048 |
| [nemotron/nemotron_nano_v3_te_deepep.yaml](nemotron/nemotron_nano_v3_te_deepep.yaml) | [nemotron/nemotron_nano_v3_te_deepep.yaml](../llm_pretrain/nemotron/nemotron_nano_v3_te_deepep.yaml) | 4096 |
| [nemotron/nemotron_super_v3_te_deepep.yaml](nemotron/nemotron_super_v3_te_deepep.yaml) | [nemotron/nemotron_super_v3_te_deepep.yaml](../llm_pretrain/nemotron/nemotron_super_v3_te_deepep.yaml) | 4096 |
| [nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml](nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml) | [nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml](../llm_pretrain/nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml) | 8192 |
| [nemotron/nemotron_ultra_v3_te_deepep.yaml](nemotron/nemotron_ultra_v3_te_deepep.yaml) | [nemotron/nemotron_ultra_v3_te_deepep.yaml](../llm_pretrain/nemotron/nemotron_ultra_v3_te_deepep.yaml) | 4096 |
| [qwen/qwen2_5_7b_quack.yaml](qwen/qwen2_5_7b_quack.yaml) | [qwen/qwen2_5_7b_quack.yaml](../llm_pretrain/qwen/qwen2_5_7b_quack.yaml) | 4096 |
| [qwen/qwen2_5_7b_quack_baseline.yaml](qwen/qwen2_5_7b_quack_baseline.yaml) | [qwen/qwen2_5_7b_quack_baseline.yaml](../llm_pretrain/qwen/qwen2_5_7b_quack_baseline.yaml) | 4096 |
| [qwen/qwen3.5_moe_te_deepep.yaml](qwen/qwen3.5_moe_te_deepep.yaml) | [qwen/qwen3.5_moe_te_deepep.yaml](../llm_pretrain/qwen/qwen3.5_moe_te_deepep.yaml) | 4096 |
| [qwen/qwen3_8_flash_next_fa4_ep64.yaml](qwen/qwen3_8_flash_next_fa4_ep64.yaml) | [qwen/qwen3_8_flash_next_fa4_ep64.yaml](../llm_pretrain/qwen/qwen3_8_flash_next_fa4_ep64.yaml) | 8192 |
| [qwen/qwen3_moe_235b_te_deepep.yaml](qwen/qwen3_moe_235b_te_deepep.yaml) | [qwen/qwen3_moe_235b_te_deepep.yaml](../llm_pretrain/qwen/qwen3_moe_235b_te_deepep.yaml) | 4096 |
| [qwen/qwen3_moe_235b_te_deepep_gb200.yaml](qwen/qwen3_moe_235b_te_deepep_gb200.yaml) | [qwen/qwen3_moe_235b_te_deepep_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_235b_te_deepep_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_235b_te_mxfp8_gb200.yaml](qwen/qwen3_moe_235b_te_mxfp8_gb200.yaml) | [qwen/qwen3_moe_235b_te_mxfp8_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_235b_te_mxfp8_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_30b_compile_attn_gb200.yaml](qwen/qwen3_moe_30b_compile_attn_gb200.yaml) | [qwen/qwen3_moe_30b_compile_attn_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_30b_compile_attn_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_30b_quack_deepep.yaml](qwen/qwen3_moe_30b_quack_deepep.yaml) | [qwen/qwen3_moe_30b_quack_deepep.yaml](../llm_pretrain/qwen/qwen3_moe_30b_quack_deepep.yaml) | 4096 |
| [qwen/qwen3_moe_30b_te_bf16_gb200.yaml](qwen/qwen3_moe_30b_te_bf16_gb200.yaml) | [qwen/qwen3_moe_30b_te_bf16_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_30b_te_bf16_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_30b_te_deepep.yaml](qwen/qwen3_moe_30b_te_deepep.yaml) | [qwen/qwen3_moe_30b_te_deepep.yaml](../llm_pretrain/qwen/qwen3_moe_30b_te_deepep.yaml) | 4096 |
| [qwen/qwen3_moe_30b_te_deepep_gb200.yaml](qwen/qwen3_moe_30b_te_deepep_gb200.yaml) | [qwen/qwen3_moe_30b_te_deepep_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_30b_te_deepep_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_30b_te_fp8.yaml](qwen/qwen3_moe_30b_te_fp8.yaml) | [qwen/qwen3_moe_30b_te_fp8.yaml](../llm_pretrain/qwen/qwen3_moe_30b_te_fp8.yaml) | 4096 |
| [qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml](qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml) | [qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml) | 4096 |
| [qwen/qwen3_moe_30b_torch.yaml](qwen/qwen3_moe_30b_torch.yaml) | [qwen/qwen3_moe_30b_torch.yaml](../llm_pretrain/qwen/qwen3_moe_30b_torch.yaml) | 4096 |
| [qwen/qwen3_moe_30b_torchmm_gb200.yaml](qwen/qwen3_moe_30b_torchmm_gb200.yaml) | [qwen/qwen3_moe_30b_torchmm_gb200.yaml](../llm_pretrain/qwen/qwen3_moe_30b_torchmm_gb200.yaml) | 4096 |
| [qwen/qwen3_next_te_deepep.yaml](qwen/qwen3_next_te_deepep.yaml) | [qwen/qwen3_next_te_deepep.yaml](../llm_pretrain/qwen/qwen3_next_te_deepep.yaml) | 4096 |
| [step/step_3.5_flash_te_deepep.yaml](step/step_3.5_flash_te_deepep.yaml) | [step/step_3.5_flash_te_deepep.yaml](../llm_pretrain/step/step_3.5_flash_te_deepep.yaml) | 4096 |
| [step/step_5_preview_full92_mock_h100.yaml](step/step_5_preview_full92_mock_h100.yaml) | [step/step_5_preview_full92_h100.yaml](../llm_pretrain/step/step_5_preview_full92_h100.yaml) | 8192 |

## Real-data finetuning

The 32 corresponding finetuning recipes use real supervised data and retain
pretrained initialization and learning hyperparameters, including the
optimizer algorithm and weight decay. They now match the benchmark's loss
implementation/precision, LoRA rank and options, optimizer execution settings,
garbage-collection interval, parallelism and batch sizes. The former Qwen32B mock
example and the migrated HellaSwag recipes now use real SQuAD. Existing
`hellaswag` filenames remain stable for CI references; newly added SQuAD recipes
use `squad` filenames. Sample limits use
`limit_dataset_samples`; the deprecated `num_samples_limit` spelling is retired
from these configs. These shorter or packed supervised inputs still define a
different workload from full-length pretraining, so report both token throughput
and actual sequence lengths instead of assuming their TPS must match. The migrated
loss upcast, optimizer-state and gradient-reduction precision come from throughput
benchmarks; their effect on finetuning convergence has not been validated.

The packed Qwen30B MXFP8 recipe keeps RoPE fusion disabled because its THD path has
a documented NaN-gradient issue. Kimi and Step-5 static routing stay disabled. Neither
exception is hidden in performance overrides. No current legacy benchmark enables
a positive `packed_sequence_size`; fixed-length synthetic data alone does not
request THD packing.

GLM-4.7 Flash finetuning explicitly sets `gate_precision: float32`. The benchmark
and pretraining configs omit this setting, and `Glm4MoeLiteForCausalLM` resolves it
to `torch.float32`. All three use FP32 router projections.

Other recorded differences:

- DeepSeek V4 finetuning omits `group_compressor_fsdp_units`, which requires unpadded
  inputs with implicit positions; its supervised batches can be padded. It keeps the other
  `model.config` tuning of the pretraining config.
- `qwen/qwen3_moe_30b_hellaswag.yaml` is the PyTorch-backend counterpart of
  `qwen/qwen3_moe_30b_torch.yaml`. Use `qwen/qwen3_moe_30b_te_deepep.yaml` for the TE
  and DeepEP backends.

| Retained benchmark | Canonical finetuning config |
|---|---|
| [deepseek/deepseek_v4_flash_h100_optimized.yaml](deepseek/deepseek_v4_flash_h100_optimized.yaml) | [deepseek_v4/deepseek_v4_flash_hellaswag.yaml](../llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml) |
| [glm/glm_4.7_flash_te_deepep.yaml](glm/glm_4.7_flash_te_deepep.yaml) | [glm/glm_4.7_flash_te_deepep.yaml](../llm_finetune/glm/glm_4.7_flash_te_deepep.yaml) |
| [glm/glm_4.7_te_deepep.yaml](glm/glm_4.7_te_deepep.yaml) | [glm/glm_4.7_te_deepep.yaml](../llm_finetune/glm/glm_4.7_te_deepep.yaml) |
| [glm/glm_5_te_deepep.yaml](glm/glm_5_te_deepep.yaml) | [glm/glm_5_hellaswag_pp.yaml](../llm_finetune/glm/glm_5_hellaswag_pp.yaml) |
| [gpt_oss/gptoss_120b_te_deepep.yaml](gpt_oss/gptoss_120b_te_deepep.yaml) | [gpt_oss/gpt_oss_120b.yaml](../llm_finetune/gpt_oss/gpt_oss_120b.yaml) |
| [gpt_oss/gptoss_20b_te_deepep.yaml](gpt_oss/gptoss_20b_te_deepep.yaml) | [gpt_oss/gpt_oss_20b.yaml](../llm_finetune/gpt_oss/gpt_oss_20b.yaml) |
| [llama3_3/llama_3_3_70b_instruct_squad_peft_pp.yaml](llama3_3/llama_3_3_70b_instruct_squad_peft_pp.yaml) | [llama3_3/llama_3_3_70b_instruct_squad_peft.yaml](../llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft.yaml) |
| [minimax/minimax_m2.5_te_deepep.yaml](minimax/minimax_m2.5_te_deepep.yaml) | [minimax_m2/minimax_m2.5_hellaswag_pp.yaml](../llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml) |
| [moonlight/moonlight_16b_te_deepep.yaml](moonlight/moonlight_16b_te_deepep.yaml) | [moonlight/moonlight_16b_te.yaml](../llm_finetune/moonlight/moonlight_16b_te.yaml) |
| [nemotron/nemotron_nano_v3_te_deepep.yaml](nemotron/nemotron_nano_v3_te_deepep.yaml) | [nemotron/nemotron_nano_v3_hellaswag.yaml](../llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml) |
| [nemotron/nemotron_super_v3_te_deepep.yaml](nemotron/nemotron_super_v3_te_deepep.yaml) | [nemotron/nemotron_super_v3_hellaswag.yaml](../llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml) |
| [nemotron/nemotron_super_v3_lora.yaml](nemotron/nemotron_super_v3_lora.yaml) | [nemotron/nemotron_super_v3_hellaswag_peft.yaml](../llm_finetune/nemotron/nemotron_super_v3_hellaswag_peft.yaml) |
| [nemotron/nemotron_ultra_v3_te_deepep.yaml](nemotron/nemotron_ultra_v3_te_deepep.yaml) | [nemotron/nemotron_ultra_v3_squad.yaml](../llm_finetune/nemotron/nemotron_ultra_v3_squad.yaml) |
| [qwen/custom_qwen2_5_32b_peft_benchmark.yaml](qwen/custom_qwen2_5_32b_peft_benchmark.yaml) | [qwen/qwen2_5_32b_peft_benchmark.yaml](../llm_finetune/qwen/qwen2_5_32b_peft_benchmark.yaml) |
| [qwen/qwen2_5_7b_peft_benchmark.yaml](qwen/qwen2_5_7b_peft_benchmark.yaml) | [qwen/qwen2_5_7b_squad_peft.yaml](../llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml) |
| [qwen/qwen3_moe_30b_torch.yaml](qwen/qwen3_moe_30b_torch.yaml) | [qwen/qwen3_moe_30b_hellaswag.yaml](../llm_finetune/qwen/qwen3_moe_30b_hellaswag.yaml) |
| [qwen/qwen3_moe_30b_te_deepep.yaml](qwen/qwen3_moe_30b_te_deepep.yaml) | [qwen/qwen3_moe_30b_te_deepep.yaml](../llm_finetune/qwen/qwen3_moe_30b_te_deepep.yaml) |
| [qwen/qwen3_next_te_deepep.yaml](qwen/qwen3_next_te_deepep.yaml) | [qwen/qwen3_next_te_deepep.yaml](../llm_finetune/qwen/qwen3_next_te_deepep.yaml) |
| [step/step_3.5_flash_te_deepep.yaml](step/step_3.5_flash_te_deepep.yaml) | [stepfun/step_3.5_flash_hellaswag_pp.yaml](../llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml) |
| [step/step_5_preview_full92_mock_h100.yaml](step/step_5_preview_full92_mock_h100.yaml) | [stepfun/step_5_preview_tulu3_sft.yaml](../llm_finetune/stepfun/step_5_preview_tulu3_sft.yaml) |
| [deepseek/deepseek_v4_flash_mok_32gpu.yaml](deepseek/deepseek_v4_flash_mok_32gpu.yaml) | [deepseek_v4/deepseek_v4_flash_squad_mok_gb200.yaml](../llm_finetune/deepseek_v4/deepseek_v4_flash_squad_mok_gb200.yaml) |
| [glm/glm_4.7_flash_mok_4gpu.yaml](glm/glm_4.7_flash_mok_4gpu.yaml) | [glm/glm_4.7_flash_squad_mok_gb200.yaml](../llm_finetune/glm/glm_4.7_flash_squad_mok_gb200.yaml) |
| [gpt_oss/gptoss_120b_te_deepep_gb200.yaml](gpt_oss/gptoss_120b_te_deepep_gb200.yaml) | [gpt_oss/gpt_oss_120b_te_deepep_gb200.yaml](../llm_finetune/gpt_oss/gpt_oss_120b_te_deepep_gb200.yaml) |
| [gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml](gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml) | [gpt_oss/gpt_oss_120b_te_mxfp8_gb200.yaml](../llm_finetune/gpt_oss/gpt_oss_120b_te_mxfp8_gb200.yaml) |
| [kimi/kimi_k3_gb200.yaml](kimi/kimi_k3_gb200.yaml) | [kimi/k3_hellaswag.yaml](../llm_finetune/kimi/k3_hellaswag.yaml) |
| [kimi/kimi_k3_gb200_2k.yaml](kimi/kimi_k3_gb200_2k.yaml) | [kimi/k3_squad_te_torchmm_gb200.yaml](../llm_finetune/kimi/k3_squad_te_torchmm_gb200.yaml) |
| [nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml](nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml) | [nemotron/nemotron_super_v3_squad_te_hybridep_gb200.yaml](../llm_finetune/nemotron/nemotron_super_v3_squad_te_hybridep_gb200.yaml) |
| [qwen/qwen3_moe_30b_compile_attn_gb200.yaml](qwen/qwen3_moe_30b_compile_attn_gb200.yaml) | [qwen/qwen3_moe_30b_squad_compile_attn_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_squad_compile_attn_gb200.yaml) |
| [qwen/qwen3_moe_30b_te_bf16_gb200.yaml](qwen/qwen3_moe_30b_te_bf16_gb200.yaml) | [qwen/qwen3_moe_30b_squad_te_bf16_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_squad_te_bf16_gb200.yaml) |
| [qwen/qwen3_moe_30b_te_deepep_gb200.yaml](qwen/qwen3_moe_30b_te_deepep_gb200.yaml) | [qwen/qwen3_moe_30b_squad_te_deepep_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_squad_te_deepep_gb200.yaml) |
| [qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml](qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml) | [qwen/qwen3_moe_30b_te_hybridep_mxfp8_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_te_hybridep_mxfp8_gb200.yaml) |
| [qwen/qwen3_moe_30b_torchmm_gb200.yaml](qwen/qwen3_moe_30b_torchmm_gb200.yaml) | [qwen/qwen3_moe_30b_squad_torchmm_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_squad_torchmm_gb200.yaml) |

## Interpreting throughput

Fewer measured steps do not by themselves explain a large steady-state TPS loss.
Compare after warmup with the same sequence length, global/local batches,
precision, kernels, topology, routing mode and software image. Short runs can
still be distorted by compilation, startup, final partial batches or exhaustion.
The performance generator requests 50 steps and derives the required epoch count,
so a short supervised dataset cannot silently truncate that measurement window.
Normal training and the legacy benchmark also have different timing boundaries;
report wall time and tokens per step alongside TPS when diagnosing a remaining gap.

## Packed finetuning comparison

The compatible migrated finetuning configs pack real supervised examples to the
sequence length of their corresponding pretraining config (or the retained
synthetic reference for LoRA). SQuAD padding is disabled before packing.
TE attention uses THD packing; supported SDPA/HF paths use NEAT packing. Packing
preserves document boundaries and prompt/answer labels. Full local microbatches
are retained with `dataloader.drop_last: true`; global and local batch sizes and
parallelism stay in the canonical configuration. THD configs use unfused RoPE to
avoid the fused packed-input gradient issue documented by existing packed recipes.

The following recipes remain unpacked pending support or data-size validation:

- Qwen3-Next: recurrent state is not reset at packed document boundaries.
- GPT-OSS 20B with flex attention: the current flex path does not consume the
  packed document mask.
- DeepSeek V4 with cudnn/torch attention: the selected packing paths require
  capability and document-mask validation before enabling them.
- GLM5: packing remains disabled pending validation of SQuAD at its large
  distributed batch size. Kimi K3 uses 1,024-token packing. The previous
  HellaSwag sample counts no longer describe these recipes.

These exceptions must be labeled in performance reports. Do not interpret them
as measured packed-finetuning results.

SQuAD changes the number of packs and optimizer steps per epoch. Previous
HellaSwag pack counts, partial-step percentages and throughput measurements are
historical evidence, not measurements of the current configs. Revalidate the
configured global batch on the actual tokenizer and dataset before comparing TPS.

Performance runs skip validation by default. Ordinary training retains the existing
validation sampler and loop; this migration does not add support for uneven-rank
validation. Keep tokens per step in performance reports so partial accumulation
windows remain visible.
