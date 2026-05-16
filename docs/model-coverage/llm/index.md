# Large Language Models (LLMs)

## Introduction
Large Language Models (LLMs) power a variety of tasks such as dialogue systems, text classification, summarization, and more.
NeMo AutoModel provides a simple interface for loading and fine-tuning LLMs hosted on the Hugging Face Hub.

## Run LLMs with NeMo AutoModel
To run LLMs with NeMo AutoModel, make sure you're using NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you intend to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest version of NeMo AutoModel by using:

```bash
pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/AutoModel.git
```

For other installation options (e.g., uv), see the [NeMo AutoModel Installation Guide](../../guides/installation.md).

## Supported Models

NeMo AutoModel supports the [AutoModelForCausalLM](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm) in the [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) category. During preprocessing, it uses `transformers.AutoTokenizer`, which is sufficient for most LLM cases. If your model requires custom text handling, override the tokenizer in your recipe YAML or provide a custom dataset `_target_`. See [LLM datasets](../../guides/llm/dataset.md) and [dataset overview](../../guides/dataset-overview.md).

| Owner | Model Family | Architectures |
|---|---|---|
| Meta | [Llama](meta/llama.md) | `LlamaForCausalLM` |
| Google | [Gemma](google/gemma.md) | `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM` |
| Qwen / Alibaba Cloud | [Qwen2](qwen/qwen2.md) | `Qwen2ForCausalLM` |
| Qwen / Alibaba Cloud | [Qwen2 MoE](qwen/qwen2-moe.md) | `Qwen2MoeForCausalLM` |
| Qwen / Alibaba Cloud | [Qwen3](qwen/qwen3.md) | `Qwen3ForCausalLM` |
| Qwen / Alibaba Cloud | [Qwen3 MoE](qwen/qwen3-moe.md) | `Qwen3MoeForCausalLM` |
| Qwen / Alibaba Cloud | [Qwen3-Next](qwen/qwen3-next.md) | `Qwen3NextForCausalLM` |
| DeepSeek | [DeepSeek](deepseek-ai/deepseek.md) | `DeepseekForCausalLM` |
| DeepSeek | [DeepSeek-V3](deepseek-ai/deepseek-v3.md) | `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` |
| DeepSeek | [DeepSeek V4 Flash](deepseek-ai/dsv4-flash.md) | `DeepseekV4ForCausalLM` |
| Mistral AI | [Mistral](mistralai/mistral.md) | `MistralForCausalLM` |
| Mistral AI | [Mixtral](mistralai/mixtral.md) | `MixtralForCausalLM` |
| Mistral AI | [Ministral3 / Devstral](mistralai/ministral3.md) | `Mistral3ForConditionalGeneration` |
| Microsoft | [Phi](microsoft/phi.md) | `PhiForCausalLM` |
| Microsoft | [Phi-3 / Phi-4](microsoft/phi3.md) | `Phi3ForCausalLM` |
| Microsoft | [Phi-3-Small](microsoft/phi3-small.md) | `Phi3SmallForCausalLM` |
| NVIDIA | [Nemotron](nvidia/nemotron.md) | `NemotronForCausalLM` |
| NVIDIA | [Nemotron-H](nvidia/nemotron-h.md) | `NemotronHForCausalLM` |
| NVIDIA | [Nemotron-Flash](nvidia/nemotron-flash.md) | `NemotronFlashForCausalLM` |
| NVIDIA | [Nemotron-Super](nvidia/nemotron-super.md) | `DeciLMForCausalLM` |
| THUDM / Zhipu AI | [ChatGLM](thudm/chatglm.md) | `ChatGLMModel` |
| THUDM / Zhipu AI | [GLM-4](thudm/glm4.md) | `GlmForCausalLM`, `Glm4ForCausalLM` |
| THUDM / ZAI | [GLM-4 MoE](thudm/glm4-moe.md) | `Glm4MoeForCausalLM`, `Glm4MoeLiteForCausalLM` |
| THUDM / ZAI | [GLM-5 / GLM-5.1](thudm/glm5-moe-dsa.md) | `GlmMoeDsaForCausalLM` |
| IBM | [Granite](ibm/granite.md) | `GraniteForCausalLM` |
| IBM | [Granite MoE](ibm/granite-moe.md) | `GraniteMoeForCausalLM`, `GraniteMoeSharedForCausalLM` |
| IBM | [Bamba](ibm/bamba.md) | `BambaForCausalLM` |
| Allen AI | [OLMo](allenai/olmo.md) | `OLMoForCausalLM` |
| Allen AI | [OLMo2](allenai/olmo2.md) | `OLMo2ForCausalLM` |
| Allen AI | [OLMoE](allenai/olmoe.md) | `OLMoEForCausalLM` |
| OpenAI | [GPT-OSS](openai/gpt-oss.md) | `GptOssForCausalLM` |
| OpenAI | [GPT-2](openai/gpt2.md) | `GPT2LMHeadModel` |
| EleutherAI | [GPT-J](eleutherai/gpt-j.md) | `GPTJForCausalLM` |
| EleutherAI | [GPT-NeoX / Pythia](eleutherai/gpt-neox.md) | `GPTNeoXForCausalLM` |
| BigCode | [StarCoder](bigcode/starcoder.md) | `GPTBigCodeForCausalLM` |
| BigCode | [StarCoder2](bigcode/starcoder2.md) | `Starcoder2ForCausalLM` |
| BAAI | [Aquila](baai/aquila.md) | `AquilaForCausalLM` |
| Baichuan Inc | [Baichuan](baichuan-inc/baichuan.md) | `BaiChuanForCausalLM` |
| Cohere | [Command-R](cohere/command-r.md) | `CohereForCausalLM`, `Cohere2ForCausalLM` |
| TII | [Falcon](tiiuae/falcon.md) | `FalconForCausalLM` |
| LG AI Research | [EXAONE](lgai-exaone/exaone.md) | `ExaoneForCausalLM` |
| InternLM | [InternLM](internlm/internlm.md) | `InternLMForCausalLM`, `InternLM2ForCausalLM`, `InternLM3ForCausalLM` |
| Inception AI | [Jais](inceptionai/jais.md) | `JAISLMHeadModel` |
| MiniMax | [MiniMax-M2](minimax/minimax-m2.md) | `MiniMaxM2ForCausalLM` |
| OpenBMB | [MiniCPM](openbmb/minicpm.md) | `MiniCPMForCausalLM`, `MiniCPM3ForCausalLM` |
| Moonshot AI | [Moonlight](moonshotai/moonlight.md) | `DeepseekV3ForCausalLM` |
| ByteDance Seed | [Seed](bytedance-seed/seed.md) | `Qwen2ForCausalLM` |
| Upstage | [Solar](upstage/solar.md) | `SolarForCausalLM` |
| OrionStar | [Orion](orionstar/orion.md) | `OrionForCausalLM` |
| Stability AI | [StableLM](stabilityai/stablelm.md) | `StableLmForCausalLM` |
| Stepfun AI | [Step-3.5](stepfun-ai/step-3-5.md) | `Step3p5ForCausalLM` |
| Parasail AI | [GritLM](parasail-ai/gritlm.md) | `GritLM` |
| Tencent | [Hy3-preview](tencent/hy3.md) | `HYV3ForCausalLM` |
| Xiaomi MiMo | [MiMo-V2-Flash](xiaomimimo/mimo-v2-flash.md) | `MiMoV2FlashForCausalLM` |

## Fine-Tuning LLMs with NeMo AutoModel

The models listed above can be fine-tuned using NeMo AutoModel. We support two primary fine-tuning approaches:

1. **Parameter-Efficient Fine-Tuning (PEFT)**: Updates only a small subset of parameters (typically <1%) using techniques like Low-Rank Adaptation (LoRA).
2. **Supervised Fine-Tuning (SFT)**: Updates all or most model parameters for deeper adaptation.

See the [Fine-Tuning Guide](../../guides/llm/finetune.md) to learn how to apply both methods to your data.

:::{tip}
In these guides, we use the `SQuAD v1.1` dataset for demonstration purposes, but you can use your own data. Update the recipe YAML `dataset` / `validation_dataset` sections accordingly. See [LLM datasets](../../guides/llm/dataset.md) and [dataset overview](../../guides/dataset-overview.md).
:::

```{toctree}
:hidden:

meta/llama
google/gemma
qwen/qwen2
qwen/qwen2-moe
qwen/qwen3
qwen/qwen3-moe
qwen/qwen3-next
deepseek-ai/deepseek
deepseek-ai/deepseek-v3
deepseek-ai/dsv4-flash
mistralai/mistral
mistralai/mixtral
mistralai/ministral3
microsoft/phi
microsoft/phi3
microsoft/phi3-small
nvidia/nemotron
nvidia/nemotron-h
nvidia/nemotron-flash
nvidia/nemotron-super
thudm/chatglm
thudm/glm4
thudm/glm4-moe
thudm/glm5-moe-dsa
ibm/granite
ibm/granite-moe
ibm/bamba
allenai/olmo
allenai/olmo2
allenai/olmoe
openai/gpt-oss
openai/gpt2
eleutherai/gpt-j
eleutherai/gpt-neox
bigcode/starcoder
bigcode/starcoder2
baai/aquila
baichuan-inc/baichuan
cohere/command-r
tiiuae/falcon
lgai-exaone/exaone
internlm/internlm
inceptionai/jais
minimax/minimax-m2
openbmb/minicpm
moonshotai/moonlight
bytedance-seed/seed
upstage/solar
orionstar/orion
stabilityai/stablelm
stepfun-ai/step-3-5
parasail-ai/gritlm
tencent/hy3
xiaomimimo/mimo-v2-flash
```
