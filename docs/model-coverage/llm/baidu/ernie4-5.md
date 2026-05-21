# ERNIE 4.5

[ERNIE 4.5](https://huggingface.co/baidu) is Baidu's dense and Mixture-of-Experts language model family with long-context text checkpoints on Hugging Face.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architectures** | `Ernie4_5ForCausalLM`, `Ernie4_5_MoeForCausalLM` |
| **Parameters** | 0.36B dense; 21B total / 3B active MoE |
| **Context Length** | 131,072 tokens |
| **HF Org** | [baidu](https://huggingface.co/baidu) |
:::

## Available Models

- **ERNIE-4.5-0.3B-PT**: dense text checkpoint with 0.36B parameters.
- **ERNIE-4.5-21B-A3B-PT**: text MoE checkpoint with 21B total parameters and 3B activated parameters per token.

## Architectures

- `Ernie4_5ForCausalLM`: dense Hugging Face implementation path.
- `Ernie4_5_MoeForCausalLM`: custom NeMo AutoModel implementation with expert parallelism support.

## Example HF Models

| Model | HF ID |
|---|---|
| ERNIE 4.5 0.3B PT | [`baidu/ERNIE-4.5-0.3B-PT`](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT) |
| ERNIE 4.5 21B A3B PT | [`baidu/ERNIE-4.5-21B-A3B-PT`](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`ernie4_5_0p3b_hellaswag.yaml <../../../../examples/llm_finetune/ernie4_5/ernie4_5_0p3b_hellaswag.yaml>` | SFT -- ERNIE 4.5 0.3B on HellaSwag with the Hugging Face implementation |
| {download}`ernie4_5_21b_a3b_hellaswag.yaml <../../../../examples/llm_finetune/ernie4_5/ernie4_5_21b_a3b_hellaswag.yaml>` | SFT -- ERNIE 4.5 21B A3B on HellaSwag with TE attention and DeepEP |

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Run a dense recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/ernie4_5/ernie4_5_0p3b_hellaswag.yaml
```

**4. Run the MoE recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/ernie4_5/ernie4_5_21b_a3b_hellaswag.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2. Navigate to the AutoModel directory**:

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/ernie4_5/ernie4_5_21b_a3b_hellaswag.yaml
```
:::

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [baidu/ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT)
- [baidu/ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
