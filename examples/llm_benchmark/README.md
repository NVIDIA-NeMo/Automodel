# Legacy LLM benchmarks and the performance-testing migration

The standalone synthetic benchmark recipes in this directory are being migrated to the
[`ci.perf_testing` system in PR #4637](https://github.com/NVIDIA-NeMo/Automodel/pull/4637).
They are retained for now for historical comparisons and for variants that do not yet have a canonical training counterpart.

For the migrated pairs below, the canonical `llm_finetune` recipe owns the runtime tuning:
parallelism, backend/kernel selection, activation checkpointing, compile options, and global/local batch sizes.
Pretrained weight-loading adapters remain enabled; random-initialization benchmark load flags are not runtime tuning.
Run that training config to measure performance on its real dataset. `ci.perf_testing` selects the schedule and walltime;
it does not duplicate the workload or permit generic configuration overrides. Its sole optional model override is the
boolean `fake_balanced_gate`; omit it to measure the recipe's real routing. Label any explicit balanced-routing result.

These legacy files still use their original synthetic inputs and benchmark measurement loops. Their tokens/sec values
are a separate workload reference, not a throughput target for real-data training. During migration, make changes to the
canonical training settings first. Do not restore a second performance-only copy of those settings under `ci.perf_testing`.

## Canonical counterparts

| Retained legacy benchmark | Canonical training recipe |
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

## GB200 counterparts

GB200 recipes use four GPUs per node and require an ARM64 runtime image on a GB200 cluster.
They are scheduled separately from EOS/H100; an EOS result does not validate these configurations.

| Retained legacy benchmark | Canonical GB200 training recipe | Nodes |
|---|---|---:|
| [deepseek/deepseek_v4_flash_mok_32gpu.yaml](deepseek/deepseek_v4_flash_mok_32gpu.yaml) | [deepseek_v4/deepseek_v4_flash_hellaswag_mok_gb200.yaml](../llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag_mok_gb200.yaml) | 8 |
| [glm/glm_4.7_flash_mok_4gpu.yaml](glm/glm_4.7_flash_mok_4gpu.yaml) | [glm/glm_4.7_flash_hellaswag_mok_gb200.yaml](../llm_finetune/glm/glm_4.7_flash_hellaswag_mok_gb200.yaml) | 1 |
| [gpt_oss/gptoss_120b_te_deepep_gb200.yaml](gpt_oss/gptoss_120b_te_deepep_gb200.yaml) | [gpt_oss/gpt_oss_120b_te_deepep_gb200.yaml](../llm_finetune/gpt_oss/gpt_oss_120b_te_deepep_gb200.yaml) | 8 |
| [gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml](gpt_oss/gptoss_120b_te_mxfp8_gb200.yaml) | [gpt_oss/gpt_oss_120b_te_mxfp8_gb200.yaml](../llm_finetune/gpt_oss/gpt_oss_120b_te_mxfp8_gb200.yaml) | 8 |
| [kimi/kimi_k3_gb200.yaml](kimi/kimi_k3_gb200.yaml) | [kimi/k3_hellaswag.yaml](../llm_finetune/kimi/k3_hellaswag.yaml) | 64 |
| [kimi/kimi_k3_gb200_2k.yaml](kimi/kimi_k3_gb200_2k.yaml) | [kimi/k3_hellaswag_te_torchmm_gb200.yaml](../llm_finetune/kimi/k3_hellaswag_te_torchmm_gb200.yaml) | 64 |
| [nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml](nemotron/nemotron_super_v3_te_hybridep_8k_ep16_gb200.yaml) | [nemotron/nemotron_super_v3_hellaswag_te_hybridep_gb200.yaml](../llm_finetune/nemotron/nemotron_super_v3_hellaswag_te_hybridep_gb200.yaml) | 4 |
| [qwen/qwen3_moe_30b_compile_attn_gb200.yaml](qwen/qwen3_moe_30b_compile_attn_gb200.yaml) | [qwen/qwen3_moe_30b_hellaswag_compile_attn_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_hellaswag_compile_attn_gb200.yaml) | 2 |
| [qwen/qwen3_moe_30b_te_bf16_gb200.yaml](qwen/qwen3_moe_30b_te_bf16_gb200.yaml) | [qwen/qwen3_moe_30b_hellaswag_te_bf16_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_hellaswag_te_bf16_gb200.yaml) | 2 |
| [qwen/qwen3_moe_30b_te_deepep_gb200.yaml](qwen/qwen3_moe_30b_te_deepep_gb200.yaml) | [qwen/qwen3_moe_30b_hellaswag_te_deepep_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_hellaswag_te_deepep_gb200.yaml) | 2 |
| [qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml](qwen/qwen3_moe_30b_te_mxfp8_gb200.yaml) | [qwen/qwen3_moe_30b_te_hybridep_mxfp8_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_te_hybridep_mxfp8_gb200.yaml) | 2 |
| [qwen/qwen3_moe_30b_torchmm_gb200.yaml](qwen/qwen3_moe_30b_torchmm_gb200.yaml) | [qwen/qwen3_moe_30b_hellaswag_torchmm_gb200.yaml](../llm_finetune/qwen/qwen3_moe_30b_hellaswag_torchmm_gb200.yaml) | 2 |

Both Kimi variants retain real routing; benchmark-only static routing is disabled because it requires fake gates.
The existing packed Qwen MXFP8 training config retains unfused RoPE because fused RoPE is documented to produce NaN gradients with THD inputs.
The two MoK variants require a Mixture-of-Kittens image build (`MOK_ARCH`); the standard image generator excludes them until that dependency is available.
The Nemotron Super GB200 variant retains the existing AM-156 known-issue exclusion; scheduling metadata does not clear that issue.

Nine GB200 benchmark variants still have no exact-model canonical training counterpart: the six DeepSeek-V3 variants,
the two Qwen3-235B variants, and Mistral Large 4. They remain standalone benchmarks; this migration does not substitute
a different model (such as DeepSeek-V3.2 or Qwen3-30B) and claim equivalence.

The two Kimi variants retain their respective kernel and pipeline tuning on the same real HellaSwag dataset.
The benchmark names `2k` and `8k` describe their historical synthetic input lengths; the canonical training data and packing determine actual sequence lengths.

GPU validation results must distinguish scheduled, excluded, queued, failed, and successful recipes.
A legacy synthetic TPS value is not a pass threshold for these real-data recipes.

## Retained synthetic counterpart

`llm_finetune/qwen/qwen2_5_32b_peft_benchmark.yaml` already uses `MockIterableDataset` despite its directory.
Its data is unchanged in this migration. Treat its throughput as a retained synthetic measurement, not as real-data
finetuning performance. The other canonical counterparts in the table retain their existing real datasets.

## Sequence packing

Fixed-length mock inputs are not packed real-data sequences. A positive
`packed_sequence.packed_sequence_size` in a benchmark must be carried into its canonical training recipe with a
compatible dataset and collator. None of the retained benchmark configs currently enables a positive packing size.
