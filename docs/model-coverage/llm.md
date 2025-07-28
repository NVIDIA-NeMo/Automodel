# Large Language Models (LLMs)

## Introduction
Large Language Models (LLMs) power a variety of tasks such as dialogue systems, text classification, summarization, and more.
NeMo Automodel provides a simple interface for loading and fine-tuning LLMs hosted on the Hugging Face Hub.

## Run LLMs with NeMo Automodel
To run LLMs with NeMo Automodel, make sure you're using NeMo container version `25.07` or later. If the model you intend to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest version of NeMo Automodel by using:

```bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/Automodel.git
```

For other installation options (e.g., uv), please see our [Installation Guide](../guides/installation.md).

## Supported Models
NeMo Automodel interoperates with most LLMs available on the Hugging Face Hub. During preprocessing, it uses `transformers.AutoTokenizer`, which is sufficient for most LLM cases. If your model requires custom text handling, such as for reasoning tasks, you can override the default tokenizer during the data preparation stage.

The table below lists the main architectures we test against (FSDP2 combined with SFT/PEFT) and includes a representative checkpoint for each.


| Architecture                          | Models                                | Example HF Models                                                                 |
|---------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------|
| `AquilaForCausalLM`                   | Aquila, Aquila2                       | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.                                      |
| `BaiChuanForCausalLM`                 | Baichuan2, Baichuan                   | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.            |
| `BambaForCausalLM`                    | Bamba                                 | `ibm-ai-platform/Bamba-9B`                                                        |
| `ChatGLMModel` / `ChatGLMForConditionalGeneration` | ChatGLM                      | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`,  etc.                                   |
| `CohereForCausalLM` / `Cohere2ForCausalLM` | Command‑R                        | `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024`, etc.    |
| `DeciLMForCausalLM`                   | DeciLM                                | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc.                                    |
| `DeepseekForCausalLM`                 | DeepSeek                              | `deepseek-ai/deepseek-llm-7b-chat` etc.                                           |
| `ExaoneForCausalLM`                   | EXAONE‑3                              | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.                                      |
| `FalconForCausalLM`                   | Falcon                                | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.              |
| `GemmaForCausalLM`                    | Gemma                                 | `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc.                                 |
| `Gemma2ForCausalLM`                   | Gemma 2                               | `google/gemma-2-9b`, etc.                                                         |
| `Gemma3ForCausalLM`                   | Gemma 3                               | `google/gemma-3-1b-it` etc.                                                       |
| `GlmForCausalLM`                      | GLM‑4                                 | `THUDM/glm-4-9b-chat-hf` etc.                                                     |
| `Glm4ForCausalLM`                     | GLM‑4‑0414                            | `THUDM/GLM-4-32B-0414` etc.                                                       |
| `GPTBigCodeForCausalLM`               | StarCoder, SantaCoder, WizardCoder    | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0` etc. |
| `GPTJForCausalLM`                     | GPT‑J                                 | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j` etc.                                  |
| `GPTNeoXForCausalLM`                  | GPT‑NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b` etc. |
| `GraniteForCausalLM`                  | Granite 3.0, Granite 3.1, PowerLM     | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b` etc. |
| `GraniteMoeForCausalLM`               | Granite 3.0 MoE, PowerMoE             | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b` etc. |
| `GraniteMoeSharedForCausalLM`         | Granite MoE Shared                    | `ibm-research/moe-7b-1b-active-shared-experts` (test model)                       |
| `GritLM`                              | GritLM                                | `parasail-ai/GritLM-7B-vllm`.                                                     |
| `InternLMForCausalLM`                 | InternLM                              | `internlm/internlm-7b`, `internlm/internlm-chat-7b` etc.                          |
| `InternLM2ForCausalLM`                | InternLM2                             | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b` etc.                        |
| `InternLM3ForCausalLM`                | InternLM3                             | `internlm/internlm3-8b-instruct` etc.                                             |
| `JAISLMHeadModel`                     | Jais                                  | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3` etc. |
| `LlamaForCausalLM`                    | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B` etc. |
| `MiniCPMForCausalLM`                  | MiniCPM                               | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16` etc.                 |
| `MiniCPM3ForCausalLM`                 | MiniCPM3                              | `openbmb/MiniCPM3-4B` etc.                                                        |
| `MistralForCausalLM`                  | Mistral, Mistral‑Instruct             | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1` etc.            |
| `MixtralForCausalLM`                  | Mixtral‑8x7B, Mixtral‑8x7B‑Instruct   | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` etc.        |
| `NemotronForCausalLM`                 | Nemotron‑3, Nemotron‑4, Minitron      | `nvidia/Minitron-8B-Base` etc.                                                    |
| `OLMoForCausalLM`                     | OLMo                                  | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf` etc.                                   |
| `OLMo2ForCausalLM`                    | OLMo2                                 | `allenai/OLMo2-7B-1124` etc.                                                      |
| `OLMoEForCausalLM`                    | OLMoE                                 | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct` etc.              |
| `OrionForCausalLM`                    | Orion                                 | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat` etc.                   |
| `PhiForCausalLM`                      | Phi                                   | `microsoft/phi-1_5`, `microsoft/phi-2` etc.                                       |
| `Phi3ForCausalLM`                     | Phi‑4, Phi‑3                          | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct` etc. |
| `Phi3SmallForCausalLM`                | Phi‑3‑Small                           | `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct` etc.   |
| `Qwen2ForCausalLM`                    | QwQ, Qwen2                            | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B` etc.            |
| `Qwen2MoeForCausalLM`                 | Qwen2MoE                              | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat` etc.                      |
| `Qwen3ForCausalLM`                    | Qwen3                                 | `Qwen/Qwen3-8B` etc.                                                              |
| `Qwen3MoeForCausalLM`                 | Qwen3MoE                              | `Qwen/Qwen3-30B-A3B` etc.                                                         |
| `StableLmForCausalLM`                 | StableLM                              | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2` etc.      |
| `Starcoder2ForCausalLM`               | Starcoder2                            | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b` etc.   |
| `SolarForCausalLM`                    | Solar Pro                             | `upstage/solar-pro-preview-instruct` etc.                                          |




## Fine-Tuning LLMs with NeMo Automodel

The models listed above can be fine-tuned using NeMo Automodel to adapt them to specific tasks or domains. We support two primary fine-tuning approaches:

1. **Parameter-Efficient Fine-Tuning (PEFT)**: Updates only a small subset of parameters (typically <1%) using techniques like Low-Rank Adaptation (LoRA). This is ideal for resource-constrained environments. See our [PEFT Guide](../guides/llm/peft.md) for details.

2. **Supervised Fine-Tuning (SFT)**: Updates all or most model parameters for deeper adaptation, suitable for high-precision applications. See our [SFT Guide](../guides/llm/sft.md) for implementation details.

:::{tip}
In these guides, we use the `SQuAD v1.1` dataset for demonstation purposes, but you can specify your own data as needed.
:::

### Example: Fine-Tuning with SQuAD Dataset

We demonstrate fine-tuning using the Stanford Question Answering Dataset (SQuAD) as an example. SQuAD is a reading comprehension dataset where models learn to answer questions based on given context passages.

Key features of SQuAD:
- **v1.1**: All answers are present in the context (simpler for basic fine-tuning)
- **v2.0**: Includes unanswerable questions (more realistic but complex)

Sample data format:
```json
{
    "id": "5733be284776f41900661182",
    "title": "University_of_Notre_Dame",
    "context": "Architecturally, the school has...",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": {
        "text": ["Saint Bernadette Soubirous"],
        "answer_start": [515]
    }
}
```
This structure makes SQuAD ideal for training context-based question answering models. Both our PEFT and SFT guides use SQuAD v1.1 as an example, but you can substitute your own dataset as needed.

### Get Started with Fine-Tuning
To fine-tune any of the supported models:

1. Choose your approach:
   * For parameter-efficient tuning: Follow the [PEFT Guide](../guides/llm/peft.md)
   * For full model tuning: Follow the [SFT Guide](../guides/llm/sft.md)

2. Key steps in both guides:
   * Model and dataset configuration
   * Training recipe setup
   * Inference with fine-tuned models
   * Model sharing via Hugging Face Hub
   * Deployment with vLLM

3. Example launch commands:

```bash
# For PEFT
automodel finetune llm -c peft_guide.yaml

# For SFT
automodel finetune llm -c sft_guide.yaml
```

Both guides provide complete YAML configuration examples and explain how to:
  * Customize training parameters
  * Monitor progress
  * Save and share checkpoints
  * Deploy the fine-tuned model with optimized inference
