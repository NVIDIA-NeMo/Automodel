# Fine-Tune Qwen3-Omni for ASR (Wu Chinese Example)

This guide walks through end-to-end ASR fine-tuning of
`Qwen/Qwen3-Omni-30B-A3B-Instruct` on a HuggingFace audio dataset, using the
NeMo AutoModel VLM training stack. The running example is the gated
`yuekai/WenetSpeech_Wu_1k` (Wu Chinese / Shanghainese), but the same recipe and
tools work for any HF dataset that exposes ``{audio, text}`` columns.

The path has three stages:

1. **Train** the thinker sub-model with the `FinetuneRecipeForVLM` recipe.
2. **Convert** the NEMO-saved thinker checkpoint into a HuggingFace-compatible
   Qwen3-Omni export (so `transformers.AutoModel*` and vLLM can load it).
3. **Decode** the test set with vLLM and compute CER against the reference.

To set up your environment to run NeMo AutoModel, follow the
[Installation Guide](https://github.com/NVIDIA-NeMo/Automodel#-install-nemo-automodel).
The bundled DTensor / vLLM-worker venvs at `/opt/ray_venvs/` are missing
`torchcodec` and `qwen_omni_utils` — every component below is written to run
without them.

---

## 1. Data

### Dataset

The example uses [`yuekai/WenetSpeech_Wu_1k`](https://huggingface.co/datasets/yuekai/WenetSpeech_Wu_1k),
a gated 1 000-hour corpus of Wu Chinese speech. Each row has:

| Column   | Type                                | Notes                                  |
|----------|-------------------------------------|----------------------------------------|
| `audio`  | `Audio(decode=True)` struct         | `{"bytes": <wav>, "path": ...}` after `cast_column(decode=False)` |
| `text`   | `string`                            | Chinese transcript (Wu / Mandarin mix) |
| `prompt` | `string` (optional)                 | Per-sample instruction; ignored by this recipe |

Access requires an HF access token with explicit approval on the dataset page.

```bash
hf auth login --force          # if the cached token lacks access
hf auth whoami                 # confirms which account is logged in
```

The recipe's dataset builder will then load the dataset via the standard
`datasets.load_dataset` path.

### Built-in builder: `make_hf_audio_asr_dataset`

`nemo_automodel.components.datasets.vlm.datasets.make_hf_audio_asr_dataset`
returns a HuggingFace `Dataset` whose `__getitem__` lazily produces a single
`{"conversation": [...]}` dict suitable for `qwen3_omni_asr_collate_fn`. The
key design points:

* **No `torchcodec`**: the audio column is cast to `Audio(decode=False)` and
  decoded inside the lazy transform with `soundfile.read(io.BytesIO(...))`
  (mono mix + `float32` cast + optional `scipy.signal.resample_poly`). The
  pattern matches `result/decode_vllm.py`.
* **`with_transform` for lazy decoding**: building the dataset object is a
  constant-time metadata read; audio decode + chat-template assembly only run
  inside dataloader workers when a batch is fetched. Startup time for the
  builder is independent of split size.
* **Configurable prompt shape**: by default the conversation is the minimal
  `user(audio) → assistant(transcript)` shape (both `system_prompt` and
  `user_prompt` default to `None`). Setting either or both expands the
  conversation: `system_prompt="..."` adds a `system` turn, `user_prompt="..."`
  prepends a text item before the audio inside the user turn. Whitespace-only
  prompts are treated as absent.
* **Dataset-agnostic**: the builder accepts any HuggingFace audio dataset that
  exposes an audio column (cast to `Audio(decode=False)`) and a transcript
  column. Defaults (`audio_column="audio"`, `text_column="text"`,
  `name=None`) cover AMI, LibriSpeech, GigaSpeech, and WenetSpeech out of the
  box; per-dataset overrides go in the recipe YAML.

```python
from nemo_automodel.components.datasets.vlm.datasets import (
    make_hf_audio_asr_dataset,
)

dataset = make_hf_audio_asr_dataset(
    path_or_dataset="yuekai/WenetSpeech_Wu_1k",
    split="train[:5000]",
    sampling_rate=16000,
    system_prompt=None,
    user_prompt=(
        "请将这段上海话（吴语）语音逐字转写成纯文本。"
        "只输出转写文本，不要加任何引导语、解释或标点。"
    ),
)
# dataset[0]["conversation"] yields:
#   [
#     {"role": "user",      "content": [{"type": "text", "text": "请将…"},
#                                       {"type": "audio", "audio": np.ndarray}]},
#     {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
#   ]
```

### Built-in collate: `qwen3_omni_asr_collate_fn`

`nemo_automodel.components.datasets.vlm.collate_fns.qwen3_omni_asr_collate_fn`
batches the lazy samples into model inputs without depending on
`qwen_omni_utils`:

* Walks each conversation for `{"type": "audio", "audio": <ndarray>}` items
  and feeds the raw waveforms straight to `Qwen3OmniMoeProcessor`'s
  `WhisperFeatureExtractor` (skip the `process_mm_info` helper entirely).
* Validates and coerces every audio payload through
  `_validate_and_coerce_audio_payload` (1-D `float32`; otherwise raise
  `ValueError` naming the sample index and offending shape/dtype).
* Pins `padding_side="right"` so the recipe's `count_tail_padding` token
  accounting works correctly.
* Reuses `build_labels_from_template` (marker-based; `Qwen3OmniMoeProcessor`
  is in `_IMSTART_TEMPLATE_PROCESSORS`) and emits pre-shifted labels.

The collate is selected via the YAML's `dataloader.collate_fn._target_`; it is
intentionally **not** registered in the global `COLLATE_FNS` map so the
existing `Qwen3OmniMoeProcessor → qwen3_omni_collate_fn` mapping keeps
serving non-ASR VLM users that *do* have `qwen_omni_utils` installed.

### Using a different HF audio dataset

The Wu YAML is one of two example configs:
`examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml` is a sibling that points at
the public AMI meeting corpus and shows the minimal YAML diff required to
swap datasets. To target your own dataset, set `dataset.path_or_dataset` and
override the defaults below only when the dataset diverges:

| Dataset                                 | `path_or_dataset`                    | `name`            | `text_column` |
|-----------------------------------------|--------------------------------------|-------------------|---------------|
| `yuekai/WenetSpeech_Wu_1k` (gated)      | `${oc.env:WENETSPEECH_WU_PATH}`      | —                 | `text` (default) |
| `edinburghcstr/ami`                     | `edinburghcstr/ami`                  | `ihm` or `sdm`    | `text` (default) |
| `openslr/librispeech_asr`               | `openslr/librispeech_asr`            | optional config   | `text` (default) |
| `speechcolab/gigaspeech`                | `speechcolab/gigaspeech`             | optional config   | `text` (default) |
| `mozilla-foundation/common_voice_*`     | `mozilla-foundation/common_voice_18_0` | language code (e.g. `zh-CN`) | **`sentence`** |

YAML override snippet for AMI / SDM far-field training:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_hf_audio_asr_dataset
  path_or_dataset: edinburghcstr/ami
  name: sdm           # near-field is `ihm`
  split: train
  sampling_rate: 16000
  system_prompt: null
  user_prompt: "Transcribe the English audio into text."
```

YAML override snippet for CommonVoice (note `text_column: sentence`):

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_hf_audio_asr_dataset
  path_or_dataset: mozilla-foundation/common_voice_18_0
  name: zh-CN
  text_column: sentence
  split: train
  sampling_rate: 16000
```

Audio columns are universally named `audio` across these datasets, so the
default `audio_column="audio"` rarely needs an override.

---

## 2. Train

### Example config

`examples/vlm_finetune/qwen3_omni_asr/wenetspeech_wu_sft.yaml` is a
ready-to-run full fine-tune for the 30B-A3B Omni model on a single 8-GPU
node. The defaults:

| Section            | Setting                                                  |
|--------------------|----------------------------------------------------------|
| `recipe`           | `FinetuneRecipeForVLM`                                   |
| `distributed`      | `fsdp2`, `ep_size=8`, `tp=cp=pp=1`                       |
| `freeze_config`    | `freeze_vision_tower=true`, `freeze_audio_tower=false`, `freeze_language_model=false`  |
| `step_scheduler`   | `global_batch_size=64`, `local_batch_size=8`, `ckpt_every_steps=200`, `max_steps=2000` (validated on 8x H100) |
| `optimizer`        | `AdamW(lr=2.0e-5, betas=[0.9, 0.95], weight_decay=0.0)`  |
| `checkpoint`       | `result/checkpoints/...`, `model_save_format=safetensors`, `save_consolidated=true` |
| `wandb`            | enabled by default; project / name / dir via env vars     |
| `dataset`          | `make_hf_audio_asr_dataset` with `WENETSPEECH_WU_PATH` env var |

`peft:` is intentionally omitted — both the language model and the audio
tower are trainable, vision tower stays frozen. With `ep_size=8` the MoE
experts are sharded across all 8 GPUs.

`examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml` is a sibling config that
swaps the dataset to `edinburghcstr/ami` (subset `ihm`) and uses an English
transcription `user_prompt`. Unlike the Wu YAML it ships with a higher-
throughput `step_scheduler` (`global_batch_size=64`, `local_batch_size=8`,
`ckpt_every_steps=200`) — those numbers were measured on 8x H100 80GB:
~1.4 step/s steady-state, ~36–40 GB peak / GPU, one epoch over the ~69k
post-1.0s-filter AMI IHM train clips in ~22 min (vs ~2 h at `local_batch_size=1`).
The reason batch can be pushed this high is that peak memory on this MoE is
dominated by FSDP/expert all-gather (~36 GB), not by activations.
See "Using a different HF audio dataset" above for column overrides on other
datasets.

### Launch

`examples/vlm_finetune/qwen3_omni_asr/train.sh` is the canonical entry point.
It validates `NPROC_PER_NODE=8` and `WENETSPEECH_WU_PATH` up front, then
`exec`s the standard NeMo AutoModel launcher. Pick the YAML by setting the
`CONFIG` env var (the CLI takes the YAML as a positional argument; do *not*
try to pass `--config-file <yaml>` at the end — it is silently ignored):

```bash
# WandB: as long as WANDB_API_KEY is set in the environment (either via
# `wandb login`, or by exporting WANDB_API_KEY=<key> from your shell rc),
# the run streams online and shows up in the configured project. To dry-run
# without uploading, set:
#   export WANDB_MODE=offline
# and `wandb sync result/wandb/<run-dir>` later.
export WANDB_PROJECT=${WANDB_PROJECT:-qwen3-omni-asr-wenetspeech-wu}

# Wu (default): the Wu YAML resolves `${oc.env:WENETSPEECH_WU_PATH}` from
# this var, so it is genuinely required.
export WENETSPEECH_WU_PATH=yuekai/WenetSpeech_Wu_1k   # gated HF id or local mirror
examples/vlm_finetune/qwen3_omni_asr/train.sh \
    --dataset.split 'train[:5000]'                    # any CLI override is forwarded

# AMI / any public dataset (path is hard-coded in the YAML): the
# WENETSPEECH_WU_PATH precondition still fires, so set a stub.
CONFIG=examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml \
WENETSPEECH_WU_PATH=unused \
    examples/vlm_finetune/qwen3_omni_asr/train.sh
```

The launcher pins the Python interpreter to
`/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python`
(set `PY=...` to override) and runs
`torch.distributed.run --nproc_per_node=8 --nnodes=1 -m nemo_automodel.cli.app <yaml>`.

### What gets saved

Every `ckpt_every_steps` steps the recipe writes a consolidated checkpoint
under `result/checkpoints/qwen3_omni_asr_wenetspeech_wu/epoch_E_step_S/`:

```
epoch_0_step_99/
├── config.yaml                # snapshot of the recipe config
├── losses.json
├── dataloader/                # StatefulDataLoader state for restart
├── optim/                     # AdamW state (~30 GB / shard for 30B FT)
├── rng/                       # PyTorch + numpy + python RNG state
├── step_scheduler.pt
└── model/
    ├── shard-XXXXX-model-00001-of-00001.safetensors  # DCP sharded
    ├── consolidated/                                  # **HF-format export**
    │   ├── config.json                               # thinker subtree only
    │   ├── model.safetensors.index.json
    │   ├── model-00001-of-00013.safetensors
    │   └── ...
    └── chat_template.jinja, tokenizer*.json, processor_config.json
```

The `consolidated/` directory is the artefact you continue with for
inference. It already holds the trained weights and the right tokenizer +
processor — but its `config.json` is the *thinker sub-model only*
(`model_type=qwen3_omni_moe_thinker`), which neither `transformers.AutoConfig`
nor vLLM 0.20.0 recognises as a top-level architecture.

### Resume

`--checkpoint.restore_from <ckpt_dir>` reloads the model state, optimizer,
RNG, and dataloader position. PEFT (when used) is loaded via
`_set_peft_state_dict`; full-FT checkpoints are loaded directly into the
sharded model parts. The recipe does not require the conversion step below
for restart — only for *external* inference tooling.

---

## 3. Convert: thinker → HF-compatible Omni

NeMo currently maps `Qwen3OmniMoeForConditionalGeneration` to a custom
*thinker-only* class (the parent Omni model in HF has `thinker / code2wav /
talker` sub-modules; this recipe only needs the thinker for ASR). The saved
`consolidated/config.json` therefore carries
`model_type=qwen3_omni_moe_thinker`, which is **not registered as a top-level
architecture** in `transformers.CONFIG_MAPPING`. The resulting symptom for any
downstream loader is:

```text
ValueError: The checkpoint you are trying to load has model type
`qwen3_omni_moe_thinker` but Transformers does not recognize this architecture.
```

### Tool: `tools/wrap_thinker_ckpt_as_omni.py`

`tools/wrap_thinker_ckpt_as_omni.py` rewraps the thinker checkpoint as a
full Qwen3-Omni export by:

1. Renaming + copying the trained `thinker.*` shards into the output dir.
2. Copying the un-trained `code2wav.*` and `talker.*` shards verbatim from
   the cached HF base model (these were never modified during ASR training).
3. Writing a merged `model.safetensors.index.json` over all three buckets.
4. Replacing the bogus `config.json` with the base model's
   (`model_type=qwen3_omni_moe`, `architectures=["Qwen3OmniMoeForConditionalGeneration"]`).
5. Copying the rest of the HF metadata (tokenizer, processor, generation
   config, chat template) from base; the recipe-saved `chat_template.jinja`
   wins if present.

Memory footprint stays at roughly one shard (~5 GB) at a time — there is no
full-model materialisation.

```bash
python tools/wrap_thinker_ckpt_as_omni.py \
    --ckpt-dir   result/checkpoints/qwen3_omni_asr_wenetspeech_wu/epoch_0_step_199/model/consolidated \
    --base-dir   ~/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/<rev> \
    --out-dir    /tmp/qwen3_omni_asr_step_199_wrapped
```

The output directory is a drop-in replacement for the public Qwen3-Omni
snapshot — only the `thinker.*` weights differ.

### Sanity-check the export

```bash
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/tmp/qwen3_omni_asr_step_199_wrapped')
print('model_type:', cfg.model_type)               # qwen3_omni_moe
print('architectures:', cfg.architectures)         # ['Qwen3OmniMoeForConditionalGeneration']
"
```

### About `merge_lora_streaming.py`

If you trained with a `peft:` block instead of full FT, the saved checkpoint
is an adapter (no consolidated weights). Use
`tools/merge_lora_streaming.py` to merge the adapter into the base model;
the output of that tool is also already a HF-format Omni directory and
**does not require** `wrap_thinker_ckpt_as_omni.py`. Pick the right tool by
inspecting the checkpoint:

| You see in the `model/` dir                  | Use                            |
|----------------------------------------------|--------------------------------|
| `adapter_model.safetensors` + `adapter_config.json` (PEFT) | `merge_lora_streaming.py` |
| `consolidated/model-00001-of-NN.safetensors` (full FT) | `wrap_thinker_ckpt_as_omni.py` |

---

## 4. Decode

### vLLM (recommended)

`result/decode_vllm.py` runs ASR inference with vLLM over any HF audio
dataset and emits both transcripts and a CER report. It expects the
HF-format Omni export from the conversion step above. The bundled
`result/decode_asr.sh` is a thin wrapper that fixes the prompt and dataset
defaults:

```bash
cd result

# Point at the wrapped export instead of the base model.
sed -i 's|HF_MODEL_PATH=.*|HF_MODEL_PATH=/tmp/qwen3_omni_asr_step_199_wrapped|' decode_asr.sh

bash decode_asr.sh vllm
```

The script invokes `decode_vllm.py` with:

| Flag                              | Value (default)                                    |
|-----------------------------------|----------------------------------------------------|
| `--hf-model-path`                 | wrapped export (or base model)                     |
| `--dataset`                       | `yuekai/WenetSpeech-Wu-ASR-Bench`                  |
| `--subset` / `--split`            | `default` / `test`                                 |
| `--id-column`                     | `utt_id`                                           |
| `--prompt`                        | the Wu transcription instruction                   |
| `--batch-size`                    | 16                                                 |
| `--max-new-tokens`                | 256                                                |
| `--tensor-parallel-size`          | 2                                                  |
| `--log-dir`                       | `exp/wenetspeech_wu_test_<TAG>_vllm`               |

Outputs:

```
result/exp/wenetspeech_wu_test_<TAG>_vllm/
├── recogs.txt   # alternating <utt_id>:\tref=… / <utt_id>:\thyp=… lines
├── errs.txt     # %CER, ins/del/sub counts, per-utt diff, top-K substitutions
└── log.txt      # full vLLM + decode log
```

### Scoring

`result/score_recogs.py` computes CER from an existing `recogs.txt` using
the same `kaldialign`-based alignment as `decode_vllm.py`:

```bash
python result/score_recogs.py result/exp/<run>/recogs.txt
```

Punctuation in hypotheses is stripped before scoring (`，。！？：；、…`
etc.) because the WenetSpeech reference is unpunctuated.

### Known runtime caveats

* `kaldialign` is required by `decode_vllm.py` / `score_recogs.py`. The
  bundled vLLM-worker venv does not ship it; install with
  `pip install kaldialign` into whichever venv runs the decode.
* First-time vLLM startup for the 30B MoE involves a **flashinfer JIT
  compile of the cutlass fused-MoE kernel** (5-10 minutes of `nvcc` +
  `gcc`). Subsequent runs hit `/root/.cache/flashinfer` and are instant.
  During the compile vLLM emits `INFO ... [shm_broadcast.py:681] No
  available shared memory broadcast block found in 60 seconds` once a
  minute — this is expected.
* If a previous decode was `SIGKILL`-ed mid-handshake, orphan
  `VLLM::Worker_TP*` processes may still hold 30+ GB on GPU 0/1. Use
  `nvidia-smi --query-compute-apps=pid --format=csv` to find them and
  `kill -KILL <pid>` directly before relaunching.

### Result observed on the example

| Model                                                  | %CER   |
|--------------------------------------------------------|--------|
| Base Qwen3-Omni-30B-A3B-Instruct + Wu prompt           | 42.80% |
| FT step 199 (200 steps, full FT, audio_tower trainable)| **20.87%** |

(`yuekai/WenetSpeech-Wu-ASR-Bench`, 4 851 utterances, vLLM TP=2,
`max_new_tokens=256`, greedy.) The audio tower being unfrozen is essential —
LoRA-on-LM-only with frozen audio tower under-performs on low-resource
dialects.

---

## 5. Tips and Pitfalls

| Symptom                                                                 | Likely cause                                                                                            | Fix                                                                                                                              |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `DatasetNotFoundError` on `WENETSPEECH_WU_PATH`                         | HF token does not have access to the gated dataset                                                       | `hf auth login --force` with an account that has been granted access; or point at a local mirror                                |
| `AssertionError: trainable_params cannot be empty`                      | LoRA `target_modules` pattern did not match anything (e.g. used `thinker.model.layers.*`)                | Inspect `model.named_modules()` post-load; the NEMO custom thinker class exposes LM layers at `model.layers.*` (no `thinker.` prefix) |
| `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'to'` | Pre-existing bug; `get_audio_features()` returns a named output, the forward code called `.to()` on it directly | Fixed in `nemo_automodel/components/models/qwen3_omni_moe/model.py`; regression covered by `tests/unit_tests/models/qwen3_omni_moe/test_qwen3_omni_moe_model.py` |
| `ValueError: model type 'qwen3_omni_moe_thinker' not recognized`        | Loading the NEMO consolidated checkpoint directly with HF or vLLM                                        | Run `tools/wrap_thinker_ckpt_as_omni.py` to wrap the thinker as a full Omni export                                              |
| `ModuleNotFoundError: torchcodec`                                       | Some other ASR/audio loader is being imported                                                            | The built-in dataset builder uses soundfile only; check that no `Audio(decode=True)` path is invoked outside the recipe          |
| OOM during full FT                                                      | 30B + audio tower trainable plus AdamW states                                                            | Drop `step_scheduler.global_batch_size` to 4; or enable activation checkpointing via the `moe:` sub-config; or freeze audio tower |
| First vLLM run hangs at "No available shared memory broadcast block"    | flashinfer JIT compiling cutlass MoE kernels                                                             | Wait 5-10 min on first run; subsequent runs use `/root/.cache/flashinfer`                                                       |

---

## Reference: files added or relied on by this guide

| Path                                                                       | Role                                                          |
|----------------------------------------------------------------------------|---------------------------------------------------------------|
| `nemo_automodel/components/datasets/vlm/datasets.py`                       | `make_hf_audio_asr_dataset` (lazy `with_transform` builder) |
| `nemo_automodel/components/datasets/vlm/collate_fns.py`                    | `qwen3_omni_asr_collate_fn` (no `qwen_omni_utils`)              |
| `examples/vlm_finetune/qwen3_omni_asr/wenetspeech_wu_sft.yaml`             | Full-FT Wu Chinese ASR recipe (gated dataset)                 |
| `examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml`                        | Full-FT English ASR recipe on AMI (public; sibling demo)      |
| `examples/vlm_finetune/qwen3_omni_asr/train.sh`                            | Launcher (validates `NPROC_PER_NODE=8` + `WENETSPEECH_WU_PATH`) |
| `tools/wrap_thinker_ckpt_as_omni.py`                                       | Convert thinker checkpoint → HF Omni export                   |
| `tools/merge_lora_streaming.py`                                            | Stream-merge a LoRA adapter into the base model (peft-free)   |
| `result/decode_vllm.py`                                                    | vLLM-backed decode + CER scoring                              |
| `result/decode_asr.sh`                                                     | Wrapper script with the Wu prompt + bench dataset defaults    |
| `result/score_recogs.py`                                                   | CER scoring from an existing `recogs.txt`                     |
