# LLM Finetune Nightly Recipes

Defaults: Time = 00:10:00, Nodes = 1, vLLM deploy time = 00:30:00 (separate job)

For release testing, all recipes under `llm_finetune/` are added automatically.
Deprecated model families are excluded via [override_recipes.yml](override_recipes.yml).
The nightly scope uses only the recipes listed in [nightly_recipes.yml](nightly_recipes.yml).

## SFT

| Recipe | Time | Nodes | Ckpt Robustness | vLLM Deploy | vLLM Smoke |
|---|:---:|:---:|:---:|:---:|:---:|
| devstral2_small_2512_squad | 00:15:00 | 1 | - | - | - |
| gpt_oss_20b | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| llama3_1_8b_hellaswag_pp | 00:10:00 | 1 | - | - | - |
| llama3_2_1b_hellaswag | 00:15:00 | 1 | ✅ | - | - |
| llama3_2_1b_squad | 00:10:00 | 1 | - | - | - |
| llama3_3_nemotron_super_49B_squad | 00:45:00 | 2 | ✅ | ✅ | - |
| ministral3_3b_squad | 00:15:00 | 1 | ✅ | - | - |
| moonlight_16b_te | 00:10:00 | 1 | - | - | - |
| nemotron_flash_1b_squad | 00:15:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_8b_v1_squad | 00:20:00 | 1 | ✅ | - | - |
| nemotron_nano_9b_squad | 00:25:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_v3_hellaswag | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| nemotron_super_v3_hellaswag | 00:15:00 | 4 | ✅ | ✅ | ✅ |
| qwen3_moe_30b_hellaswag | 00:20:00 | 1 | ✅ | - | - |
| qwen3_moe_30b_te_deepep | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| step_3.5_flash_hellaswag_pp | 00:30:00 | 16 | - | - | - |

## PEFT

| Recipe | Time | Nodes | Ckpt Robustness | vLLM Deploy | vLLM Smoke |
|---|:---:|:---:|:---:|:---:|:---:|
| gpt_oss_20b_peft | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| gpt_oss_20b_single_gpu_peft | 00:10:00 | 1 | - | - | - |
| llama3_2_1b_hellaswag_peft | 00:15:00 | 1 | ✅ | - | - |
| llama3_3_nemotron_super_49B_squad_peft | 00:45:00 | 1 | ✅ | ✅ | - |
| ministral3_3b_squad_peft | 00:15:00 | 1 | ✅ | - | - |
| nemotron_flash_1b_squad_peft | 00:15:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_8b_v1_squad_peft | 00:15:00 | 1 | ✅ | - | - |
| nemotron_nano_9b_squad_peft | 00:25:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_v3_hellaswag_peft | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| nemotron_super_v3_hellaswag_peft | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| qwen3_moe_30b_lora | 00:15:00 | 1 | ✅ | - | - |
