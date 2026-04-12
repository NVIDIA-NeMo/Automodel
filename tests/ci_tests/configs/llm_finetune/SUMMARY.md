# LLM Finetune Nightly Recipes

Defaults: Time = 00:10:00, Nodes = 1, vLLM deploy time = 00:30:00 (separate job)

For release testing, all recipes under `llm_finetune/` are added automatically.
The nightly scope uses only the recipes listed in [nightly_recipes.yml](nightly_recipes.yml).

## SFT

| Recipe | Time | Nodes | Ckpt Robustness | vLLM Deploy | vLLM Smoke |
|---|:---:|:---:|:---:|:---:|:---:|
| baichuan_2_7b_squad | 00:45:00 | 1 | ✅ | ✅ | - |
| cohere_command_r_7b_squad | 00:10:00 | 1 | - | - | - |
| devstral2_small_2512_squad | 00:15:00 | 1 | - | - | - |
| falcon3_7b_instruct_squad | 00:10:00 | 1 | - | - | - |
| gemma_2_9b_it_squad | 00:10:00 | 1 | - | - | - |
| gemma_3_270m_squad | 00:20:00 | 1 | ✅ | ✅ | - |
| glm_4_9b_chat_hf_squad | 00:10:00 | 1 | - | - | - |
| gpt_oss_20b | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| gpt_oss_20b_single_gpu | 00:10:00 | 1 | - | - | - |
| granite_3_3_2b_instruct_squad | 00:10:00 | 1 | - | - | - |
| llama3_1_8b_hellaswag_pp | 00:10:00 | 1 | - | - | - |
| llama3_2_1b_hellaswag | 00:15:00 | 1 | ✅ | - | - |
| llama3_2_1b_squad | 00:10:00 | 1 | - | - | - |
| llama3_3_nemotron_super_49B_squad | 00:45:00 | 2 | ✅ | ✅ | - |
| ministral3_3b_squad | 00:15:00 | 1 | ✅ | - | - |
| mistral_nemo_2407_squad | 00:10:00 | 1 | - | - | - |
| moonlight_16b_te | 00:10:00 | 1 | - | - | - |
| nemotron_flash_1b_squad | 00:15:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_8b_v1_squad | 00:20:00 | 1 | ✅ | - | - |
| nemotron_nano_9b_squad | 00:25:00 | 1 | ✅ | ✅ | - |
| nemotron_nano_v3_hellaswag | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| nemotron_super_v3_hellaswag | 00:15:00 | 4 | ✅ | ✅ | ✅ |
| olmo_2_0425_1b_instruct_squad | 00:10:00 | 1 | - | - | - |
| phi_3_mini_it_squad | 00:10:00 | 1 | - | - | - |
| phi_4_squad | 00:35:00 | 1 | ✅ | ✅ | - |
| qwen2_5_7b_squad | 00:45:00 | 1 | ✅ | ✅ | - |
| qwen3_moe_30b_hellaswag | 00:15:00 | 1 | ✅ | - | - |
| qwen3_moe_30b_te_deepep | 00:15:00 | 1 | ✅ | ✅ | ✅ |
| seed_coder_8b_instruct_squad | 00:10:00 | 1 | - | - | - |
| starcoder_2_7b_squad | 00:15:00 | 1 | - | - | - |
| step_3.5_flash_hellaswag_pp | 00:30:00 | 16 | - | - | - |

## PEFT

| Recipe | Time | Nodes | Ckpt Robustness | vLLM Deploy | vLLM Smoke |
|---|:---:|:---:|:---:|:---:|:---:|
| baichuan_2_7b_squad_peft | 00:45:00 | 1 | ✅ | ✅ | - |
| falcon3_7b_instruct_squad_peft | 00:10:00 | 1 | - | - | - |
| gemma_2_9b_it_squad_peft | 00:15:00 | 1 | - | - | - |
| gemma_3_270m_squad_peft | 00:20:00 | 1 | ✅ | ✅ | - |
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
| phi_2_squad_peft | 00:10:00 | 1 | - | - | - |
| phi_2_squad_tp2_peft | 00:10:00 | 1 | - | - | - |
| phi_4_squad_peft | 00:35:00 | 1 | ✅ | ✅ | - |
| phi_4_squad_tp2_peft | 00:10:00 | 1 | - | - | - |
| qwen2_5_7b_peft_benchmark | 00:10:00 | 1 | - | - | - |
| qwen2_5_7b_squad_peft | 00:30:00 | 1 | ✅ | ✅ | - |
| qwen3_moe_30b_lora | 00:15:00 | 1 | ✅ | - | - |
| seed_coder_8b_instruct_squad_peft | 00:10:00 | 1 | - | - | - |
