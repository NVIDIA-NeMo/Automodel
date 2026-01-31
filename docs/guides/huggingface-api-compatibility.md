# Hugging Face compatibility (Transformers v4 & v5)

NeMo Automodel is built to work with the Hugging Face ecosystem.
In practice, compatibility comes in two layers:

- **API compatibility**: for many workflows you can keep your existing `transformers` code and swap in NeMo Automodel “drop-in” wrappers (`NeMoAutoModel*`, `NeMoAutoTokenizer`) with minimal changes.
- **Artifact compatibility**: NeMo Automodel produces **Hugging Face–compatible checkpoints** (config + tokenizer + safetensors) that can be loaded by Hugging Face Transformers and downstream tools (vLLM, SGLang, etc.).

This page summarizes what “HF compatibility” means in NeMo Automodel, calls out differences you should be aware of, and provides side-by-side examples.

## Transformers version compatibility: v4 and v5

### Transformers v4 (current default)

NeMo Automodel currently pins Hugging Face Transformers to the **v4** major line (see `pyproject.toml`, currently `transformers<=4.57.5`).

This means:

- NeMo Automodel is primarily tested and released against **Transformers v4.x**
- New model releases on the Hugging Face Hub that require a newer Transformers may require upgrading NeMo Automodel as well (similar to upgrading `transformers` directly)

### Transformers v5 (forward-compatibility + checkpoint interoperability)

Transformers **v5** introduces breaking changes across some internal utilities (e.g., cache APIs) and adds/reshapes tokenizer backends for some model families.

NeMo Automodel addresses this in two complementary ways:

- **Forward-compatibility shims**: NeMo Automodel includes small compatibility patches to smooth over known API differences across Transformers releases (for example, cache utility method names). The built-in recipes apply these patches automatically.
- **Backports where needed**: for some model families, NeMo Automodel may vendor/backport Hugging Face code that originated in the v5 development line so users can run those models while staying on a pinned v4 dependency.
- **Stable artifact format**: NeMo Automodel checkpoints are written in Hugging Face–compatible `save_pretrained` layouts (config + tokenizer + safetensors). These artifacts are designed to be loadable by both Transformers **v4** and **v5** (and non-Transformers tools that consume HF-style model repos).

:::{note}
If you are running Transformers v5 in another environment, you can still use NeMo Automodel–produced consolidated checkpoints with Transformers’ standard loading APIs. For details on the checkpoint layouts, see {doc}`checkpointing <./checkpointing>`.
:::

## What’s “drop-in” vs what’s different

### Drop-in (same mental model as Transformers)

- **Load by model ID or local path**: `from_pretrained(...)`
- **Standard HF config objects**: `AutoConfig` / `config.json`
- **Tokenizers**: standard `PreTrainedTokenizerBase` behavior, including `__call__` to create tensors and `decode`/`batch_decode`
- **Generation**: `model.generate(...)` and the usual generation kwargs

### Differences (where NeMo Automodel adds value or has constraints)

- **Performance features**: NeMo Automodel can automatically apply optional kernel patches/optimizations (e.g., SDPA selection, Liger kernels) while keeping the public model API the same.
- **Distributed training stack**: NeMo Automodel’s recipes/CLI are designed for multi-GPU/multi-node fine-tuning with PyTorch-native distributed features (FSDP2, pipeline parallelism, etc.).
- **CUDA expectation**: NeMo Automodel’s `NeMoAutoModel*` wrappers are primarily intended for GPU workflows.

:::{important}
`NeMoAutoModelForCausalLM.from_pretrained(...)` currently assumes CUDA is available (it uses `torch.cuda.current_device()` internally). If you need CPU-only inference, use Hugging Face `transformers` directly.
:::

## API mapping (Transformers ⇔ NeMo Automodel)

### API name mapping

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo Automodel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>transformers.AutoModelForCausalLM</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForCausalLM</code></td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForImageTextToText</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForImageTextToText</code></td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForSequenceClassification</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForSequenceClassification</code></td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForTextToWaveform</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForTextToWaveform</code></td>
    </tr>
    <tr>
      <td><code>transformers.AutoTokenizer.from_pretrained(...)</code></td>
      <td><code>nemo_automodel.NeMoAutoTokenizer.from_pretrained(...)</code></td>
    </tr>
    <tr>
      <td><code>model.generate(...)</code></td>
      <td><code>model.generate(...)</code></td>
    </tr>
    <tr>
      <td><code>model.save_pretrained(path)</code></td>
      <td><code>model.save_pretrained(path, checkpointer=...)</code></td>
    </tr>
  </tbody>
</table>
:::

### Load a model and tokenizer

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo Automodel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <pre><code>import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

model = model.to("cuda").eval()</code></pre>
      </td>
      <td>
        <pre><code>import torch
from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoTokenizer

model_id = "gpt2"

tokenizer = NeMoAutoTokenizer.from_pretrained(model_id)
model = NeMoAutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

model = model.eval()</code></pre>
      </td>
    </tr>
  </tbody>
</table>
:::

### Text generation

This snippet assumes you already have a `model` and `tokenizer` (see the loading snippet above).

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo Automodel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <pre><code>import torch

prompt = "Write a haiku about GPU kernels."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))</code></pre>
      </td>
      <td>
        <pre><code>import torch

prompt = "Write a haiku about GPU kernels."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))</code></pre>
      </td>
    </tr>
  </tbody>
</table>
:::

## Side-by-side examples

### Tokenizers (Transformers vs NeMo Automodel)

NeMo Automodel provides `NeMoAutoTokenizer` as a Transformers-like auto-tokenizer with a small registry for specialized backends (and a safe fallback when no specialization is needed).

:::{list-table}
:header-rows: 1
:widths: 1 1

* - Hugging Face (`transformers`)
  - NeMo Automodel (`nemo_automodel`)
* - :::{code-block} python
      from transformers import AutoTokenizer

      tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    :::
  - :::{code-block} python
      from nemo_automodel import NeMoAutoTokenizer

      # Default: use NeMo Automodel's dispatch logic (custom backend if registered,
      # otherwise a HF-compatible fallback).
      tok = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

      # If you want the raw HF tokenizer (no wrapping/dispatch):
      # tok = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", force_hf=True)
    :::
:::

## Checkpoints: save in Automodel, load everywhere

NeMo Automodel training recipes write checkpoints in Hugging Face–compatible layouts, including consolidated safetensors that you can load directly with Transformers:

- See {doc}`checkpointing <./checkpointing>` for checkpoint formats and example directory layouts.
- See {doc}`model coverage overview <../model-coverage/overview>` for notes on how model support depends on the pinned Transformers version.

If your goal is: **train/fine-tune in NeMo Automodel → deploy in the HF ecosystem**, the recommended workflow is to enable consolidated safetensors checkpoints and then load them with the standard HF APIs or downstream inference engines.
