# Large Language Models (LLMs)

## Introduction
Large Language Models (LLMs) power a variety of tasks such as dialogue systems, text classification, summarization, and more.
NeMo AutoModel provides a simple interface for loading and fine-tuning LLMs hosted on the Hugging Face Hub.

## Run LLMs with NeMo AutoModel
To run LLMs with NeMo AutoModel, make sure you're using NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you intend to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest version of NeMo AutoModel by using:

```bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/AutoModel.git
```

For other installation options (e.g., uv), please see our [Installation Guide](../guides/installation.md).

## Supported Models
NeMo AutoModel supports the [AutoModelForCausalLM](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm) in the [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) category. During preprocessing, it uses `transformers.AutoTokenizer`, which is sufficient for most LLM cases. If your model requires custom text handling, such as for reasoning tasks, you can override the default tokenizer during the data preparation stage.

The table below lists the main architectures we test against (FSDP2 combined with SFT/PEFT) and includes a representative checkpoint for each.


| Architecture                          | Models                                | Example HF Models                                                                 |
|---------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------|
| `AquilaForCausalLM`                   | Aquila, Aquila2                       | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.                                      |
| `BaiChuanForCausalLM`                 | Baichuan2, Baichuan                   | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. — example recipes: [baichuan_2_7b_squad.yaml](../../examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml), [baichuan_2_7b_squad_peft.yaml](../../examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml) |
| `BambaForCausalLM`                    | Bamba                                 | `ibm-ai-platform/Bamba-9B`                                                        |
| `ChatGLMModel` / `ChatGLMForConditionalGeneration` | ChatGLM                      | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`,  etc.                                   |
| `CohereForCausalLM` / `Cohere2ForCausalLM` | Command‑R                        | `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024`, etc. — example recipes: [cohere_command_r_7b_squad.yaml](../../examples/llm_finetune/cohere/cohere_command_r_7b_squad.yaml), [cohere_command_r_7b_squad_peft.yaml](../../examples/llm_finetune/cohere/cohere_command_r_7b_squad_peft.yaml) |
| `DeciLMForCausalLM`                   | DeciLM                                | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc.                                    |
| `DeepseekForCausalLM`                 | DeepSeek                              | `deepseek-ai/deepseek-llm-7b-chat` etc.                                           |
| `DeepseekV3ForCausalLM` / `DeepseekV32ForCausalLM` | DeepSeek V3, DeepSeek V3.2        | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3.2` — example recipe: [deepseek_v32_hellaswag_pp.yaml](../../examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml) |
| `ExaoneForCausalLM`                   | EXAONE‑3                              | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.                                      |
| `FalconForCausalLM`                   | Falcon                                | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc. — example recipes: [falcon3_7b_instruct_squad.yaml](../../examples/llm_finetune/falcon/falcon3_7b_instruct_squad.yaml), [falcon3_7b_instruct_squad_peft.yaml](../../examples/llm_finetune/falcon/falcon3_7b_instruct_squad_peft.yaml) |
| `GemmaForCausalLM`                    | Gemma                                 | `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc.                                 |
| `Gemma2ForCausalLM`                   | Gemma 2                               | `google/gemma-2-9b`, etc. — example recipes: [gemma_2_9b_it_squad.yaml](../../examples/llm_finetune/gemma/gemma_2_9b_it_squad.yaml), [gemma_2_9b_it_squad_peft.yaml](../../examples/llm_finetune/gemma/gemma_2_9b_it_squad_peft.yaml) |
| `Gemma3ForCausalLM`                   | Gemma 3                               | `google/gemma-3-1b-it` etc. — example recipes: [gemma_3_270m_squad.yaml](../../examples/llm_finetune/gemma/gemma_3_270m_squad.yaml), [gemma_3_270m_squad_peft.yaml](../../examples/llm_finetune/gemma/gemma_3_270m_squad_peft.yaml) |
| `GlmForCausalLM`                      | GLM‑4                                 | `THUDM/glm-4-9b-chat-hf` etc. — example recipes: [glm_4_9b_chat_hf_squad.yaml](../../examples/llm_finetune/glm/glm_4_9b_chat_hf_squad.yaml), [glm_4_9b_chat_hf_hellaswag_fp8.yaml](../../examples/llm_finetune/glm/glm_4_9b_chat_hf_hellaswag_fp8.yaml) |
| `Glm4ForCausalLM`                     | GLM‑4‑0414                            | `THUDM/GLM-4-32B-0414` etc.                                                       |
| `Glm4MoeForCausalLM`                  | GLM‑4‑MoE                             | `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.7` — example recipes: [glm_4.5_air_te_deepep.yaml](../../examples/llm_finetune/glm/glm_4.5_air_te_deepep.yaml), [glm_4.7_te_deepep.yaml](../../examples/llm_finetune/glm/glm_4.7_te_deepep.yaml) |
| `GPTBigCodeForCausalLM`               | StarCoder, SantaCoder, WizardCoder    | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0` etc. |
| `GPTJForCausalLM`                     | GPT‑J                                 | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j` etc.                                  |
| `GPTNeoXForCausalLM`                  | GPT‑NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b` etc. |
| `GptOssForCausalLM`                   | GPT-OSS                               | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` — example recipes: [gpt_oss_20b.yaml](../../examples/llm_finetune/gpt_oss/gpt_oss_20b.yaml), [gpt_oss_120b.yaml](../../examples/llm_finetune/gpt_oss/gpt_oss_120b.yaml) |
| `GraniteForCausalLM`                  | Granite 3.0, Granite 3.1, PowerLM     | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b` etc. — example recipes: [granite_3_3_2b_instruct_squad.yaml](../../examples/llm_finetune/granite/granite_3_3_2b_instruct_squad.yaml), [granite_3_3_2b_instruct_squad_peft.yaml](../../examples/llm_finetune/granite/granite_3_3_2b_instruct_squad_peft.yaml) |
| `GraniteMoeForCausalLM`               | Granite 3.0 MoE, PowerMoE             | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b` etc. |
| `GraniteMoeSharedForCausalLM`         | Granite MoE Shared                    | `ibm-research/moe-7b-1b-active-shared-experts` (test model)                       |
| `GritLM`                              | GritLM                                | `parasail-ai/GritLM-7B-vllm`                                                    |
| `InternLMForCausalLM`                 | InternLM                              | `internlm/internlm-7b`, `internlm/internlm-chat-7b` etc.                          |
| `InternLM2ForCausalLM`                | InternLM2                             | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b` etc.                        |
| `InternLM3ForCausalLM`                | InternLM3                             | `internlm/internlm3-8b-instruct` etc.                                             |
| `JAISLMHeadModel`                     | Jais                                  | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3` etc. |
| `LlamaForCausalLM`                    | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B` etc. — example recipes: [llama3_2_1b_squad.yaml](../../examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml), [llama_3_3_70b_instruct_squad.yaml](../../examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad.yaml) |
| `MiniCPMForCausalLM`                  | MiniCPM                               | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16` etc.                 |
| `MiniCPM3ForCausalLM`                 | MiniCPM3                              | `openbmb/MiniCPM3-4B` etc.                                                        |
| `MistralForCausalLM`                  | Mistral, Mistral‑Instruct             | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1` etc. — example recipes: [mistral_7b_squad.yaml](../../examples/llm_finetune/mistral/mistral_7b_squad.yaml), [mistral_7b_squad_peft.yaml](../../examples/llm_finetune/mistral/mistral_7b_squad_peft.yaml) |
| `MixtralForCausalLM`                  | Mixtral‑8x7B, Mixtral‑8x7B‑Instruct   | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` etc. — example recipes: [mixtral-8x7b-v0-1_squad.yaml](../../examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad.yaml), [mixtral-8x7b-v0-1_squad_peft.yaml](../../examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad_peft.yaml) |
| `NemotronForCausalLM`                 | Nemotron‑3, Nemotron‑4, Minitron      | `nvidia/Minitron-8B-Base` etc.                                                    |
| `NemotronHForCausalLM`                | Nemotron-Nano-{9B,12B}                | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, `nvidia/NVIDIA-Nemotron-Nano-12B-v2` — example recipes: [nemotron_nano_9b_squad.yaml](../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml), [nemotron_nano_9b_squad_peft.yaml](../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad_peft.yaml) |
| `NemotronHForCausalLM`                | Nemotron-3-Nano-30B-A3B-BF16                | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| `OLMoForCausalLM`                     | OLMo                                  | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf` etc.                                   |
| `OLMo2ForCausalLM`                    | OLMo2                                 | `allenai/OLMo2-7B-1124` etc. — example recipes: [olmo_2_0425_1b_instruct_squad.yaml](../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad.yaml), [olmo_2_0425_1b_instruct_squad_peft.yaml](../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad_peft.yaml) |
| `OLMoEForCausalLM`                    | OLMoE                                 | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct` etc.              |
| `OrionForCausalLM`                    | Orion                                 | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat` etc.                   |
| `PhiForCausalLM`                      | Phi                                   | `microsoft/phi-1_5`, `microsoft/phi-2` etc. — example recipes: [phi_2_squad.yaml](../../examples/llm_finetune/phi/phi_2_squad.yaml), [phi_2_squad_peft.yaml](../../examples/llm_finetune/phi/phi_2_squad_peft.yaml) |
| `Phi3ForCausalLM`                     | Phi‑4, Phi‑3                          | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct` etc. — example recipes: [phi_4_squad.yaml](../../examples/llm_finetune/phi/phi_4_squad.yaml), [phi_4_squad_peft.yaml](../../examples/llm_finetune/phi/phi_4_squad_peft.yaml) |
| `Phi3SmallForCausalLM`                | Phi‑3‑Small                           | `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct` etc.   |
| `Qwen2ForCausalLM`                    | QwQ, Qwen2                            | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B` etc. — example recipes: [qwen2_5_7b_squad.yaml](../../examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml), [qwq_32b_squad_peft.yaml](../../examples/llm_finetune/qwen/qwq_32b_squad_peft.yaml) |
| `Qwen2MoeForCausalLM`                 | Qwen2MoE                              | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat` etc. — example recipe: [qwen1_5_moe_a2_7b_qlora.yaml](../../examples/llm_finetune/qwen/qwen1_5_moe_a2_7b_qlora.yaml) |
| `Qwen3ForCausalLM`                    | Qwen3                                 | `Qwen/Qwen3-8B` etc. — example recipes: [qwen3_0p6b_hellaswag.yaml](../../examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml), [qwen3_8b_squad_spark.yaml](../../examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml) |
| `Qwen3MoeForCausalLM`                 | Qwen3MoE                              | `Qwen/Qwen3-30B-A3B` etc. — example recipes: [qwen3_moe_30b_te_deepep.yaml](../../examples/llm_finetune/qwen/qwen3_moe_30b_te_deepep.yaml), [qwen3_moe_30b_lora.yaml](../../examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml) |
| `Qwen3NextForCausalLM`                | Qwen3‑Next                            | `Qwen/Qwen3-Next-80B-A3B-Instruct` — example recipe: [qwen3_next_te_deepep.yaml](../../examples/llm_finetune/qwen/qwen3_next_te_deepep.yaml) |
| `Step3p5ForCausalLM`                  | Step‑3.5                              | `stepfun-ai/Step-3.5-Flash` — example recipe: [step_3.5_flash_hellaswag_pp.yaml](../../examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml) |
| `StableLmForCausalLM`                 | StableLM                              | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2` etc.      |
| `Starcoder2ForCausalLM`               | Starcoder2                            | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b` etc. — example recipes: [starcoder_2_7b_squad.yaml](../../examples/llm_finetune/starcoder/starcoder_2_7b_squad.yaml), [starcoder_2_7b_hellaswag_fp8.yaml](../../examples/llm_finetune/starcoder/starcoder_2_7b_hellaswag_fp8.yaml) |
| `SolarForCausalLM`                    | Solar Pro                             | `upstage/solar-pro-preview-instruct` etc.                                          |
| `Mistral3ForConditionalGeneration`    | Ministral3 3B, 8B, 14B                | `mistralai/Ministral-3-8B-Instruct-2512`, `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Ministral-3-14B-Instruct-2512` |
| `Mistral3ForConditionalGeneration`    | Devstral-Small-2-24B                | `mistralai/Devstral-Small-2-24B-Instruct-2512` — example recipes: [devstral2_small_2512_squad.yaml](../../examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml), [devstral2_small_2512_squad_peft.yaml](../../examples/llm_finetune/devstral/devstral2_small_2512_squad_peft.yaml) |


## Fine-Tuning LLMs with NeMo AutoModel

The models listed above can be fine-tuned using NeMo AutoModel to adapt them to specific tasks or domains. We support two primary fine-tuning approaches:

1. **Parameter-Efficient Fine-Tuning (PEFT)**: Updates only a small subset of parameters (typically <1%) using techniques like Low-Rank Adaptation (LoRA). This is ideal for resource-constrained environments.

2. **Supervised Fine-Tuning (SFT)**: Updates all or most model parameters for deeper adaptation, suitable for high-precision applications.

Please see our [Fine-Tuning Guide](../guides/llm/finetune.md) to learn how you can apply both of these fine-tuning methods to your data.

:::{tip}
In these guides, we use the `SQuAD v1.1` dataset for demonstration purposes, but you can use your own data.

To do so, update the recipe YAML `dataset` / `validation_dataset` sections (for example `dataset._target_`, `dataset_name`/`path_or_dataset`, and `split`). See [LLM datasets](../guides/llm/dataset.md) and [dataset overview](../guides/dataset-overview.md).
:::