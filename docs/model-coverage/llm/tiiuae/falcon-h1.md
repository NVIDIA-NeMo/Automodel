# Falcon-H1

[Falcon-H1](https://falconllm.tii.ae/) is a hybrid Mamba-2 + attention model family from the Technology Innovation Institute (TII) in Abu Dhabi. Each decoder layer runs a Mamba-2 state-space mixer in parallel with standard attention off a shared input norm, sums their outputs into the residual stream, and follows with a SwiGLU MLP. µP (maximal-update parametrization) multipliers are applied throughout — on embeddings, attention keys, the SSM input/output, the MLP, and the LM head.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `FalconH1ForCausalLM` |
| **Parameters** | 0.5B – 34B |
| **HF Org** | [tiiuae](https://huggingface.co/tiiuae) |
:::

## Available Models

- **Falcon-H1-0.5B-Instruct**
- **Falcon-H1-1.5B-Deep-Instruct**
- **Falcon-H1-7B-Instruct**
- **Falcon-H1-34B-Instruct**

## Architecture

- `FalconH1ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Falcon-H1 0.5B Instruct | [`tiiuae/Falcon-H1-0.5B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct) |
| Falcon-H1 1.5B Deep Instruct | [`tiiuae/Falcon-H1-1.5B-Deep-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct) |
| Falcon-H1 7B Instruct | [`tiiuae/Falcon-H1-7B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct) |
| Falcon-H1 34B Instruct | [`tiiuae/Falcon-H1-34B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct) |

## Requirements

The Mamba-2 mixer uses the fused Triton kernel from [`mamba-ssm`](https://github.com/state-spaces/mamba), so a CUDA GPU plus `mamba-ssm` (and its `causal-conv1d` dependency) are required.

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo:**

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Load the model** via the standard interface:

```python
from nemo_automodel import NeMoAutoModelForCausalLM
model = NeMoAutoModelForCausalLM.from_pretrained("tiiuae/Falcon-H1-0.5B-Instruct")
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** Navigate to the AutoModel directory:

```bash
cd /opt/Automodel
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [tiiuae/Falcon-H1-0.5B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct)
- [tiiuae/Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct)
- [tiiuae/Falcon-H1-7B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct)
- [tiiuae/Falcon-H1-34B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct)