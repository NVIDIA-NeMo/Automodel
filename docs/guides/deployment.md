# Deployment

:::{tip}
**TL;DR** -- AutoModel saves checkpoints in standard Hugging Face format. Load them with `transformers`, serve with vLLM/SGLang, upload to HF Hub, or convert to GGUF for Ollama.
:::

## vLLM

Serve your fine-tuned model with vLLM for high-throughput inference:

```bash
# SFT checkpoint (full model)
vllm serve checkpoints/epoch_0_step_20/model/consolidated/

# LoRA adapter on top of base model
vllm serve Qwen/Qwen3-0.6B \
  --enable-lora \
  --lora-modules my-adapter=checkpoints/epoch_0_step_20/model/consolidated/
```

:::{tip}
vLLM supports continuous batching, PagedAttention, and multi-GPU tensor parallelism out of the box. Install: `pip install vllm`.
:::

## SGLang

```bash
python -m sglang.launch_server \
  --model-path checkpoints/epoch_0_step_20/model/consolidated/ \
  --port 8000
```

:::{tip}
SGLang supports RadixAttention for fast prefix caching. Install: `pip install sglang[all]`.
:::

## Hugging Face Hub

Upload your checkpoint so others can use it:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="checkpoints/epoch_0_step_20/model/consolidated/",
    repo_id="your-username/my-finetuned-model",
    repo_type="model",
)
```

:::{note}
Run `huggingface-cli login` first if you haven't authenticated.
:::

## Ollama (via GGUF)

Ollama requires GGUF format. Convert your checkpoint first:

```bash
# 1. Install llama.cpp conversion tools
pip install llama-cpp-python

# 2. Convert HF checkpoint to GGUF
python -m llama_cpp.convert \
  --model checkpoints/epoch_0_step_20/model/consolidated/ \
  --outfile model.gguf

# 3. Create Ollama Modelfile
echo 'FROM ./model.gguf' > Modelfile

# 4. Import into Ollama
ollama create my-model -f Modelfile

# 5. Run
ollama run my-model "What is the capital of France?"
```

:::{warning}
GGUF conversion may not support all architectures. Check [llama.cpp supported models](https://github.com/ggerganov/llama.cpp#description) for compatibility.
:::

## Hugging Face Transformers (Python)

Load directly in Python for scripted inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "checkpoints/epoch_0_step_20/model/consolidated/"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Summary

| Target | Format Needed | Extra Steps |
|--------|--------------|-------------|
| **vLLM** | HF Safetensors (default) | None |
| **SGLang** | HF Safetensors (default) | None |
| **HF Hub** | HF Safetensors (default) | `upload_folder()` |
| **Ollama** | GGUF | Convert with llama.cpp |
| **Python** | HF Safetensors (default) | None |
