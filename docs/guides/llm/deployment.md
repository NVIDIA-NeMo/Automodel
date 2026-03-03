# Deploying Models with vLLM and SGLang

NeMo AutoModel saves every checkpoint in **native Hugging Face format** (Safetensors + `config.json` + tokenizer).
This means the same checkpoint directory that NeMo AutoModel writes can be loaded directly (without any **conversion step**) by any tool in the Hugging Face ecosystem, including the two most popular inference engines: [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).

Just point the engine at your checkpoint path and serve:

```bash
vllm serve checkpoints/epoch_0_step_100/model/consolidated/ --port 8000
```

Both engines expose an **OpenAI-compatible API**, so you can swap them in without changing client code.

:::{seealso}
- [Fine-Tuning Guide](finetune.md) — train or adapt a model before deployment.
- [Checkpointing Guide](../checkpointing.md) — checkpoint formats, consolidation, and Safetensors output.
:::

---

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| GPU | NVIDIA GPU with 8 GB+ VRAM (16 GB+ recommended for 1B+ models) |
| CUDA | 11.8+ |
| Python | 3.9+ |

Install either engine (or both):

```bash
pip install vllm        # vLLM
pip install "sglang[all]" # SGLang
```

:::{tip}
If you are inside the NeMo AutoModel Docker container, vLLM is already installed.
:::

---

## Checkpoint Requirements

vLLM and SGLang load models in **Hugging Face format**. A valid checkpoint directory looks like:

```text
my-checkpoint/
├── config.json
├── tokenizer.json (or tokenizer_config.json + tokenizer.model)
├── model.safetensors (or sharded model-00001-of-*.safetensors + model.safetensors.index.json)
└── generation_config.json  (optional)
```

NeMo AutoModel produces this layout automatically when you set `save_consolidated: true` and `model_save_format: safetensors` in the checkpoint config (the defaults).
See the [Checkpointing guide](../checkpointing.md) for details.

:::{important}
If your checkpoint directory contains **sharded DCP files** (`.distcp`) rather than consolidated Safetensors, you need to consolidate them first.
Use the AutoModel checkpoint utility or the recipe's `save_consolidated: true` flag to produce a Hugging Face-compatible directory.
:::

---

## Quick Local Test with a Small Model

The examples below use **`Qwen/Qwen2.5-0.5B-Instruct`** (~0.5 B parameters, ~1 GB on disk).
It runs comfortably on a single GPU with 4 GB+ VRAM and does not require any gated-access agreement, making it ideal for a quick smoke test.

:::{note}
If you already have access to `meta-llama/Llama-3.2-1B-Instruct`, you can substitute it in any of the commands below.
Just make sure your Hugging Face token is set:
```bash
export HF_TOKEN=hf_xxxxx
```
:::

### Deploy with vLLM

#### Option A: OpenAI-compatible API server

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

In a separate terminal, send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64
  }' | python3 -m json.tool
```

#### Option B: Python offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=64)
outputs = llm.generate(["What is the capital of France?"], sampling_params=params)
print(outputs[0].outputs[0].text)
```

### Deploy with SGLang

#### Option A: OpenAI-compatible API server

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

Query it with the same `curl` style (change the port to `30000`):

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64
  }' | python3 -m json.tool
```

#### Option B: Python engine API

```python
import sglang as sgl

llm = sgl.Engine(model_path="Qwen/Qwen2.5-0.5B-Instruct")
output = llm.generate("What is the capital of France?", sampling_params={"max_new_tokens": 64})
print(output["text"])
```

---

## Deploy a Local Checkpoint

After fine-tuning with NeMo AutoModel, pass the **consolidated checkpoint path** instead of a Hugging Face Hub name.

### vLLM

```bash
vllm serve /path/to/checkpoints/epoch_0_step_100/model/consolidated/ \
    --host 0.0.0.0 \
    --port 8000
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/checkpoints/epoch_0_step_100/model/consolidated/")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=64))
print(outputs[0].outputs[0].text)
```

### SGLang

```bash
python -m sglang.launch_server \
    --model-path /path/to/checkpoints/epoch_0_step_100/model/consolidated/ \
    --host 0.0.0.0 \
    --port 30000
```

### Separate tokenizer

If the tokenizer is not inside the checkpoint directory, point to it explicitly:

```bash
# vLLM
vllm serve /path/to/checkpoint --tokenizer /path/to/tokenizer

# SGLang
python -m sglang.launch_server \
    --model-path /path/to/checkpoint \
    --tokenizer-path /path/to/tokenizer
```

---

## Deploy a LoRA / PEFT Adapter

vLLM supports serving LoRA adapters on top of a base model without merging weights.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", enable_lora=True)

params = SamplingParams(temperature=0.7, max_tokens=64)
outputs = llm.generate(
    ["What is the capital of France?"],
    sampling_params=params,
    lora_request=LoRARequest("my-adapter", 1, "/path/to/lora/adapter"),
)
print(outputs[0].outputs[0].text)
```

Alternatively, use the NeMo AutoModel `vLLMHFExporter` helper:

```python
from nemo.export.vllm_hf_exporter import vLLMHFExporter

exporter = vLLMHFExporter()
exporter.export(model="/path/to/base/model", enable_lora=True)
exporter.add_lora_models(lora_model_name="my-adapter", lora_model="/path/to/lora/adapter")
print(exporter.forward(input_texts=["How are you?"], lora_model_name="my-adapter"))
```

---

## Common Configuration Flags

### vLLM

| Flag | Purpose |
|------|---------|
| `--tensor-parallel-size N` | Shard the model across N GPUs |
| `--dtype auto` | Auto-detect dtype (float16 / bfloat16) |
| `--max-model-len 4096` | Cap the maximum context length |
| `--gpu-memory-utilization 0.9` | Fraction of GPU memory to allocate |
| `--quantization awq` | Load a pre-quantized model (awq, gptq, etc.) |
| `--enforce-eager` | Disable CUDA graphs (useful for debugging) |

### SGLang

| Flag | Purpose |
|------|---------|
| `--tp N` | Tensor parallelism across N GPUs |
| `--dtype auto` | Auto-detect dtype |
| `--mem-fraction-static 0.85` | GPU memory fraction for KV cache |
| `--quantization awq` | Load a pre-quantized model |
| `--context-length 4096` | Maximum context length |

---

## Multi-GPU Deployment

For models that exceed the memory of a single GPU, increase the tensor-parallelism degree:

```bash
# vLLM on 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000

# SGLang on 4 GPUs
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 30000
```

---

## Docker Deployment

Run vLLM as a self-contained container:

```bash
docker run --gpus all -p 8000:8000 \
    -v /path/to/local/checkpoint:/model \
    vllm/vllm-openai:latest \
    --model /model
```

Or use the NeMo AutoModel container (vLLM is pre-installed):

```bash
docker run --gpus all -it --rm \
    --shm-size=8g \
    -p 8000:8000 \
    -v /path/to/checkpoint:/model \
    nvcr.io/nvidia/nemo-automodel:25.11.00 \
    vllm serve /model --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: config.json` | The checkpoint path must point to the directory that **contains** `config.json`. If you used NeMo AutoModel, this is the `model/consolidated/` subdirectory. |
| `torch.cuda.OutOfMemoryError` | Reduce `--max-model-len`, lower `--gpu-memory-utilization`, or increase `--tensor-parallel-size`. |
| Tokenizer not found | Pass `--tokenizer` (vLLM) or `--tokenizer-path` (SGLang) explicitly. |
| Gated model 401 error | Set `export HF_TOKEN=hf_xxxxx` or run `huggingface-cli login`. |
| Slow first request | The first request warms up CUDA graphs and the KV cache. Subsequent requests will be faster. |
