# 🤗 Transformers API Compatibility

NeMo AutoModel is built to work with the 🤗 Hugging Face ecosystem.
In practice, compatibility comes in two layers:

- **API compatibility**: for many workflows, you can keep your existing `transformers` code and swap in NeMo AutoModel “drop-in” wrappers (`NeMoAutoModel*`, `NeMoAutoTokenizer`) with minimal changes.
- **Artifact compatibility**: NeMo AutoModel produces **Hugging Face-compatible checkpoints** (config + tokenizer + safetensors) that can be loaded by Hugging Face Transformers and downstream tools (vLLM, SGLang, etc.).

This page summarizes what "HF compatibility" means in NeMo AutoModel, calls out differences you should be aware of, and provides side-by-side examples.

## Transformers Version Compatibility: v4 and v5

### Transformers v4 (Current Default)

NeMo AutoModel currently pins Hugging Face Transformers to the **v4** major line (see `pyproject.toml`, currently `transformers<=4.57.5`).

This means:

- NeMo AutoModel is primarily tested and released against **Transformers v4.x**
- New model releases on the Hugging Face Hub that require a newer Transformers may require upgrading NeMo AutoModel as well (similar to upgrading `transformers` directly)

### Transformers v5 (Forward-Compatibility and Checkpoint Interoperability)

Transformers **v5** introduces breaking changes across some internal utilities (e.g., cache APIs) and adds/reshapes tokenizer backends for some model families.

NeMo AutoModel addresses this in two complementary ways:

- **Forward-compatibility shims**: NeMo AutoModel includes small compatibility patches to smooth over known API differences across Transformers releases (for example, cache utility method names). The built-in recipes apply these patches automatically.
- **Backports where needed**: for some model families, NeMo AutoModel may vendor/backport Hugging Face code that originated in the v5 development line so users can run those models while staying on a pinned v4 dependency.
- **Stable artifact format**: NeMo AutoModel checkpoints are written in Hugging Face-compatible `save_pretrained` layouts (config + tokenizer + safetensors). These artifacts are designed to be loadable by both Transformers **v4** and **v5** (and non-Transformers tools that consume HF-style model repos).

:::{note}
If you are running Transformers v5 in another environment, you can still use NeMo AutoModel-produced consolidated checkpoints with Transformers' standard loading APIs. For details on the checkpoint layouts, see [checkpointing](checkpointing.md).
:::

## Drop-In Compatibility and Key Differences

### Drop-In (Same Mental Model as Transformers)

- **Load by model ID or local path**: `from_pretrained(...)`
- **Standard HF config objects**: `AutoConfig` / `config.json`
- **Tokenizers**: standard `PreTrainedTokenizerBase` behavior, including `__call__` to create tensors and `decode`/`batch_decode`
- **Generation**: `model.generate(...)` and the usual generation kwargs

### Differences (Where NeMo AutoModel Adds Value or Has Constraints)

- **Performance features**: NeMo AutoModel can automatically apply optional kernel patches/optimizations (e.g., SDPA selection, Liger kernels, DeepEP, etc.) while keeping the public model API the same.
- **Distributed training stack**: NeMo AutoModel's recipes/CLI are designed for multi-GPU/multi-node fine-tuning with PyTorch-native distributed features (FSDP2, pipeline parallelism, etc.).
- **CUDA expectation**: NeMo AutoModel's `NeMoAutoModel*` wrappers are primarily optimized for NVIDIA GPU workflows, and offer support for CPU workflows as well.

:::{important}
`NeMoAutoModelForCausalLM.from_pretrained(...)` currently assumes CUDA is available (it uses `torch.cuda.current_device()` internally). If you need CPU-only inference, use Hugging Face `transformers` directly.
:::

## API Mapping (Transformers and NeMo AutoModel)

### API Name Mapping

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 45%;">🤗 Hugging Face (<code>transformers</code>)</th>
      <th style="width: 45%;">NeMo AutoModel (<code>nemo_automodel</code>)</th>
      <th style="width: 10%;">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>transformers.AutoModelForCausalLM</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForCausalLM</code></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForImageTextToText</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForImageTextToText</code></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForSequenceClassification</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForSequenceClassification</code></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><code>transformers.AutoModelForTextToWaveform</code></td>
      <td><code>nemo_automodel.NeMoAutoModelForTextToWaveform</code></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><code>transformers.AutoTokenizer.from_pretrained(...)</code></td>
      <td><code>nemo_automodel.NeMoAutoTokenizer.from_pretrained(...)</code></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><code>model.generate(...)</code></td>
      <td><code>model.generate(...)</code></td>
      <td>🚧</td>
    </tr>
    <tr>
      <td><code>model.save_pretrained(path)</code></td>
      <td><code>model.save_pretrained(path, checkpointer=...)</code></td>
      <td>🚧</td>
    </tr>
  </tbody>
</table>
:::

## Side-by-Side Examples

### Load a Model and Tokenizer (Transformers v4)

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">🤗 Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo AutoModel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="vertical-align: top;">
        <div class="highlight"><pre><code>import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)</code></pre></div>
      </td>
      <td style="vertical-align: top;">
        <div class="highlight"><pre><code>import torch
from nemo_automodel import NeMoAutoModelForCausalLM, NeMoAutoTokenizer

model_id = "gpt2"

tokenizer = NeMoAutoTokenizer.from_pretrained(model_id)
model = NeMoAutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)</code></pre></div>
      </td>
    </tr>
  </tbody>
</table>
:::

### Text Generation

This snippet assumes you already have a `model` and `tokenizer` (see the loading snippet above).

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">🤗 Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo AutoModel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="vertical-align: top; padding-top: 0;">
        <div class="highlight" style="margin-top: 0;"><pre style="margin: 0;"><code>import torch

prompt = "Write a haiku about GPU kernels."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))</code></pre></div>
      </td>
      <td style="vertical-align: top; padding-top: 0;">
        <div class="highlight" style="margin-top: 0;"><pre style="margin: 0;"><code>import torch

prompt = "Write a haiku about GPU kernels."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))</code></pre></div>
      </td>
    </tr>
  </tbody>
</table>
:::


### Tokenizers (Transformers vs NeMo AutoModel)

NeMo AutoModel provides `NeMoAutoTokenizer` as a Transformers-like auto-tokenizer with a small registry for specialized backends (and a safe fallback when no specialization is needed).

:::{raw} html
<table>
  <thead>
    <tr>
      <th style="width: 50%;">🤗 Hugging Face (<code>transformers</code>)</th>
      <th style="width: 50%;">NeMo AutoModel (<code>nemo_automodel</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="vertical-align: top;">
        <div class="highlight"><pre><code>from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")</code></pre></div>
      </td>
      <td style="vertical-align: top;">
        <div class="highlight"><pre><code>from nemo_automodel import NeMoAutoTokenizer

tok = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")</code></pre></div>
      </td>
    </tr>
  </tbody>
</table>
:::

## Checkpoints: Save in NeMo AutoModel, Load Everywhere

NeMo AutoModel training recipes write checkpoints in Hugging Face-compatible layouts, including consolidated safetensors that you can load directly with Transformers:

- See [checkpointing](checkpointing.md) for checkpoint formats and example directory layouts.
- See [model coverage](../model-coverage/overview.md) for notes on how model support depends on the pinned Transformers version.

If your goal is: **train/fine-tune in NeMo AutoModel → deploy in the HF ecosystem**, the recommended workflow is to enable consolidated safetensors checkpoints and then load them with the standard HF APIs or downstream inference engines.
