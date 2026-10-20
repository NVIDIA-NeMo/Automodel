# Additional LLM fine-tuning examples

These configurations are extracted from #4102 for separate recipe review.
They have not been validated with GPU training or memory measurements.
They remain excluded from automatic GPU CI until that validation is recorded.

Set up the [AutoModel environment](https://docs.nvidia.com/nemo/automodel/nightly/get-started/installation),
then run from the repository root on one node with eight GPUs:

```bash
uv run automodel examples/llm_finetune/baai/aquila_7b_squad_peft.yaml --nproc-per-node 8
```

Each configuration includes its launch command and the Hugging Face revision used
when inspecting the checkpoint configuration. Access to gated checkpoints requires
accepting the publisher terms and authenticating with Hugging Face.
When changing GPU count, keep the global batch divisible by the local batch times
the data-parallel size. Training and validation may download the referenced datasets.

| Checkpoint | Configuration |
| --- | --- |
| `BAAI/Aquila-7B` | [aquila_7b_squad_peft.yaml](baai/aquila_7b_squad_peft.yaml) |
| `deepseek-ai/deepseek-llm-7b-chat` | [deepseek_llm_7b_chat_squad_peft.yaml](deepseek_ai/deepseek_llm_7b_chat_squad_peft.yaml) |
| `EleutherAI/gpt-j-6b` | [gpt_j_6b_squad_peft.yaml](eleutherai/gpt_j_6b_squad_peft.yaml) |
| `EleutherAI/gpt-neox-20b` | [gpt_neox_20b_squad_peft.yaml](eleutherai/gpt_neox_20b_squad_peft.yaml) |
| `ibm-ai-platform/Bamba-9B` | [bamba_9b_squad_peft.yaml](ibm_ai_platform/bamba_9b_squad_peft.yaml) |
| `inceptionai/jais-13b` | [jais_13b_squad_peft.yaml](inceptionai/jais_13b_squad_peft.yaml) |
| `internlm/internlm3-8b-instruct` | [internlm3_8b_instruct_squad_peft.yaml](internlm/internlm3_8b_instruct_squad_peft.yaml) |
| `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` | [exaone_3_0_7_8b_instruct_squad_peft.yaml](lgai_exaone/exaone_3_0_7_8b_instruct_squad_peft.yaml) |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | [llama_3_3_nemotron_super_49b_v1_squad_peft.yaml](nvidia/llama_3_3_nemotron_super_49b_v1_squad_peft.yaml) |
| `nvidia/Minitron-8B-Base` | [minitron_8b_base_squad_peft.yaml](nvidia/minitron_8b_base_squad_peft.yaml) |
| `OrionStarAI/Orion-14B-Base` | [orion_14b_base_squad_peft.yaml](orionstarai/orion_14b_base_squad_peft.yaml) |
| `parasail-ai/GritLM-7B-vllm` | [gritlm_7b_vllm_squad_peft.yaml](parasail_ai/gritlm_7b_vllm_squad_peft.yaml) |
| `stabilityai/stablelm-3b-4e1t` | [stablelm_3b_4e1t_squad_peft.yaml](stabilityai/stablelm_3b_4e1t_squad_peft.yaml) |
| `tiiuae/falcon-7b` | [falcon_7b_squad_peft.yaml](tiiuae/falcon_7b_squad_peft.yaml) |
| `upstage/solar-pro-preview-instruct` | [solar_pro_preview_instruct_squad_peft.yaml](upstage/solar_pro_preview_instruct_squad_peft.yaml) |
| `zai-org/chatglm3-6b` | [chatglm3_6b_squad_peft.yaml](zai_org/chatglm3_6b_squad_peft.yaml) |

The Orion example needs the legacy FlashAttention import compatibility fix split
from #4102 before its remote model code can be loaded with current Transformers.
