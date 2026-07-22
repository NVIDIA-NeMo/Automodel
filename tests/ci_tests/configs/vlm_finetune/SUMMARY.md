# VLM Finetune Nightly Recipes

Defaults: Time = 00:10:00, Nodes = 1

For release testing, all recipes under `vlm_finetune/` are added automatically.
Deprecated model families are excluded via [override_recipes.yml](override_recipes.yml).
The nightly scope uses only the recipes listed in [nightly_recipes.yml](nightly_recipes.yml).

## SFT

| Recipe | Time | Nodes |
|---|:---:|:---:|
| gemma3n_vl_4b_medpix | 00:30:00 | 1 |
| gemma4_4b | 00:30:00 | 1 |
| gemma4_26b_a4b_moe | 00:30:00 | 1 |
| internvl_3_5_4b | 00:10:00 | 1 |
| ministral3_3b_medpix | 00:10:00 | 1 |
| mistral4_medpix | 00:30:00 | 4 |
| nemotron_parse_v1_1 | 00:30:00 | 1 |
| qwen3_5_4b | 00:30:00 | 1 |
| qwen3_5_35b | 00:30:00 | 1 |
| qwen3_vl_4b_instruct_rdr | 00:10:00 | 1 |

## PEFT

| Recipe | Time | Nodes |
|---|:---:|:---:|
| gemma3n_vl_4b_medpix_peft | 00:10:00 | 1 |
