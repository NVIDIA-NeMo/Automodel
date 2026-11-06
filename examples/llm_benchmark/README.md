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

## Sequence packing

Fixed-length mock inputs are not packed real-data sequences. A positive
`packed_sequence.packed_sequence_size` in a benchmark must be carried into its canonical training recipe with a
compatible dataset and collator. None of the retained benchmark configs currently enables a positive packing size.
