# Gemma 4 E2B AfriInstruct migration handoff

This document is the operational handoff for continuing the Gemma 4 E2B
AfriInstruct fine-tuning project on another machine. It records the decisions,
implemented files, observed results, known pitfalls, and exact commands needed
to rebuild the data and continue training without departing from NeMo
AutoModel's existing architecture.

## Current status

As of 2026-07-14:

- GitHub fork: `https://github.com/u25724267-svg/Automodel`
- Working branch: `u25724267-svg/feat/gemma4-afriinstruct`
- Pipeline implementation commit before this handoff: `598d7a10dd774a78b055bb58673f398d793f820b`
- Model: `google/gemma-4-E2B-it`
- Dataset: `llama-lang-adapt/AfriInstruct-Data`
- Training approach: text-only supervised fine-tuning through the existing VLM
  recipe, starting with LoRA
- CPU data preparation: completed successfully on the old machine
- Exact token-count precomputation: run on the old machine, but the new machine
  should rebuild it with its newly prepared dataset
- GPU qualification: not started
- Training checkpoints: none
- Immediate continuation point: rebuild the environment and data, validate
  them, then run the 20-step single-GPU LoRA qualification gate

Do not look for a checkpoint to resume. No training step has run yet.

## Architecture decision

Gemma 4 E2B is a multimodal conditional-generation architecture even though
this project uses text-only AfriInstruct examples. It must remain on the
framework's VLM path:

- recipe: `FinetuneRecipeForVLM`
- model wrapper: `NeMoAutoModelForImageTextToText`
- recipe implementation: `nemo_automodel/recipes/vlm/finetune.py`
- dataset builder:
  `nemo_automodel.components.datasets.vlm.datasets.make_meta_dataset`

Do not move this project to `NeMoAutoModelForCausalLM`, an LLM recipe, or the
Gemma 7B SQuAD example. The existing
`examples/vlm_finetune/gemma4/gemma4_2b_peft.yaml` recipe was the closest
reference, but it targets a different dataset. This project reuses its
framework path and supplies AfriInstruct-specific data and configuration.

No new model class, recipe class, trainer, dataset component, CLI route, or
notebook was required. The pipeline is deliberately YAML-driven and composed
from existing framework components.

## Implemented files

The feature branch contains:

- `tools/prepare_afriinstruct.py`
  - streams AfriInstruct through Hugging Face Datasets;
  - normalizes `instruction`, `output`, `lang`, `task`, and `source`;
  - converts each record to the existing `messages` conversation schema;
  - skips malformed records;
  - applies optional language, task, and source filters;
  - performs deterministic hash sampling;
  - removes exact duplicate instruction/output pairs using disk-backed SQLite;
  - creates a deterministic validation split;
  - writes bounded JSONL shards partitioned by language and task;
  - writes train/validation meta manifests and `summary.json`;
  - refuses to overwrite a nonempty output directory.
- `examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml`
  - LoRA qualification and primary training recipe.
- `examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_full.yaml`
  - full language-backbone SFT recipe, to be used only after LoRA succeeds.
- `examples/vlm_finetune/gemma4/README_afriinstruct.md`
  - compact user-facing workflow.
- `tests/unit_tests/tools/test_prepare_afriinstruct.py`
  - CPU tests for preparation, sampling, filtering, deduplication, splitting,
    manifests, and overwrite protection.
- `tests/unit_tests/recipes/test_gemma4_afriinstruct_configs.py`
  - asserts the Gemma 4 VLM, text-only, freezing, PEFT, and checkpoint policy.

This handoff supplements the compact README with migration state and failure
history.

## Recipe invariants

Preserve these settings unless a measured qualification result justifies a
new versioned experiment:

- `google/gemma-4-E2B-it`
- `torch.bfloat16`
- SDPA attention
- `use_cache: false`
- `output_hidden_states: true`
- FSDP2
- `tp_size: 1`, `cp_size: 1`, `pp_size: 1`
- activation checkpointing disabled
- embeddings frozen
- vision tower frozen
- audio tower frozen
- language model trainable
- no fake images
- packed text sequence length 4096
- fused linear cross-entropy with logit soft-capping 30
- local batch size 1 and global batch size 8

Gemma 4 E2B/E4B shares keys and values across trailing layers. Activation
recomputation is intentionally disabled for this configuration. Do not enable
activation checkpointing merely to address an out-of-memory condition. Use a
machine with more memory or qualify a separately reviewed configuration.

The LoRA recipe uses rank 16, alpha 32, no dropout, and excludes vision,
audio, image-encoder, and LM-head modules. Its learning rate is `2e-4`. The
full language-backbone recipe has no PEFT section and uses `2e-5`.

## Observed results on the old machine

The representative preparation command scanned the complete source train
split and retained approximately one percent:

```text
input_records:       8,949,088
sampled_out_records: 8,859,042
duplicate_records:         397
malformed_records:           0
written_records:        89,649
train_records:          88,783
validation_records:        866
```

The Hugging Face log resolved the dataset to source revision
`0a6325e6806ffcd7b3baefcc3c6b2de435ac6e1e`. The current utility does not
expose a dataset revision flag, so a later upstream dataset revision may
produce different counts. Record the revision shown by the new run and treat
the new `summary.json` as the source of truth.

The retained data was not balanced. Hash sampling preserved the source
distribution:

```text
Multitask:                 78,156
QA:                         5,159
Sentiment Analysis:         2,462
translation:                1,579
news_topic_classification:    936
Summarization:                722
NER:                          578
POS:                           57
```

`xP3` contributed 83,315 of the 89,649 retained records. Large language
groups included Swahili, Igbo, Yoruba, Kinyarwanda, Xhosa, Zulu, Chichewa,
and Shona. Report evaluation by language, task, and source rather than relying
only on an aggregate loss.

The preparation took about 39 minutes on the old machine. A one-percent
sample still scans all 8.95 million source rows; the option reduces retained
and training data, not source download time.

Validation completed before migration:

- example YAML lint: 438 YAML files passed;
- focused CPU tests: 7 passed;
- framework `make_meta_dataset` loader: 88,783 train and 866 validation;
- conversation role contract: `user` followed by `assistant`;
- `git diff --check`: passed.

No GPU training result exists yet because the old GPU was occupied.

## New-machine prerequisites

Use the NVIDIA NeMo AutoModel 26.04 container. Do not copy `.venv` or
`/opt/venv` from the old machine.

Before starting:

```bash
nvidia-smi
docker --version
df -h
```

Confirm that Docker can see the GPU and that the machine has enough local
storage for the Gemma weights, Hugging Face cache, generated dataset, and
checkpoints.

Clone the feature branch:

```bash
git clone \
  --branch u25724267-svg/feat/gemma4-afriinstruct \
  https://github.com/u25724267-svg/Automodel.git
cd Automodel
git branch --show-current
git log -1 --oneline
```

Create persistent host directories:

```bash
mkdir -p "$HOME/afriinstruct" "$HOME/hf-cache" "$HOME/checkpoints" "$HOME/wandb"
```

Start the container with the checkout and persistent state mounted:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g \
  -v "$HOME/Automodel:/opt/Automodel" \
  -v "$HOME/afriinstruct:/data/afriinstruct" \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/checkpoints:/checkpoints" \
  -v "$HOME/wandb:/logs/wandb" \
  nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

All remaining commands in this document run inside the container unless
explicitly stated otherwise.

## Synchronize the container environment

The checkout is bind-mounted over the source bundled in the image. Apply the
repository-provided container patch before syncing dependencies:

```bash
cd /opt/Automodel
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --all-groups --extra vlm
```

The patch intentionally modifies `pyproject.toml` and `uv.lock` in the mounted
checkout so `uv` uses NVIDIA's baked PyTorch stack. These are machine-local
container changes. Do not commit them to the feature branch.

Do not use `pip`, do not install individual PyTorch or torchvision wheels,
and do not add `--extra cuda` for this pipeline. The CUDA extra previously
failed on optional compiled packages and is not required for this Gemma VLM
recipe.

Verify the repaired environment:

```bash
/opt/venv/bin/python -c \
  "import torch, torchvision, transformers, huggingface_hub; \
print('torch:', torch.__version__); \
print('torchvision:', torchvision.__version__); \
print('transformers:', transformers.__version__); \
print('huggingface_hub:', huggingface_hub.__version__); \
from transformers import Gemma4Config; \
print('Gemma4Config OK')"
```

The working old-container combination was:

```text
torch: 2.12.0a0+0291f960b6.nv26.04.48445190
torchvision: 0.26.0a0+48956e05.nv26.04.48445190
transformers: 5.12.1
huggingface_hub: 1.9.2
Gemma4Config OK
```

Exact NVIDIA development hashes can change with a refreshed image, but
PyTorch and torchvision must be matching builds.

## Authenticate with Hugging Face

The account must have accepted the access terms for
`google/gemma-4-E2B-it`. Authenticate inside the container:

```bash
/opt/venv/bin/hf auth login
```

Alternatively, set the token without putting it in shell history:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
```

Never put the token in YAML, Markdown, Git, or a command pasted into an issue.

Confirm model access before doing more work:

```bash
/opt/venv/bin/python -c \
  "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('google/gemma-4-E2B-it'); \
print('Gemma tokenizer access OK')"
```

## Rebuild AfriInstruct

First run the bounded schema smoke test. This validates network access,
source schema, conversion, manifests, and local write permissions:

```bash
/opt/venv/bin/python tools/prepare_afriinstruct.py \
  --output-dir /data/afriinstruct/e2b-schema-smoke-v1 \
  --max-samples 1000 \
  --validation-fraction 0.05 \
  --shard-size 250
```

`--max-samples` takes the first matching records from an ordered source. It is
only a schema smoke test and must never be used as the training dataset.

Create the representative subset:

```bash
/opt/venv/bin/python tools/prepare_afriinstruct.py \
  --output-dir /data/afriinstruct/e2b-representative-v1 \
  --sample-fraction 0.01 \
  --validation-fraction 0.01 \
  --shard-size 50000
```

Expected behavior:

- the command may run for tens of minutes or several hours;
- repeated Hugging Face `302` and `206 Partial Content` responses are normal;
- an unauthenticated warning is nonfatal, although `HF_TOKEN` improves rate
  limits;
- `--sample-fraction 0.01` still scans every source row;
- success prints `Prepared ... records` and creates `summary.json`,
  `train_meta.json`, and `validation_meta.json`;
- if interrupted, use a new versioned output directory rather than trying to
  overwrite a partial nonempty directory.

Inspect the new summary:

```bash
cat /data/afriinstruct/e2b-representative-v1/summary.json
```

Confirm that all expected languages, tasks, and sources are represented. Do
not require exact equality with the old counts if the upstream source revision
changed, but investigate large unexplained differences.

## Precompute exact Gemma token counts

The framework meta loader can estimate lengths as characters divided by
three, but the proper pipeline uses the repository's existing token-count
precomputation utility. Run it after data preparation:

```bash
/opt/venv/bin/python scripts/precompute_tokens.py \
  --meta /data/afriinstruct/e2b-representative-v1/train_meta.json \
  --processor google/gemma-4-E2B-it \
  --inplace \
  --workers 8

/opt/venv/bin/python scripts/precompute_tokens.py \
  --meta /data/afriinstruct/e2b-representative-v1/validation_meta.json \
  --processor google/gemma-4-E2B-it \
  --inplace \
  --workers 8
```

Reduce `--workers` if host memory is constrained. The utility loads one
tokenizer per worker.

In-place precomputation replaces JSONL files through temporary files. When it
runs as container root, the resulting files may be mode `0600` and appear as
`nobody:nogroup` from a user-namespaced host. This is acceptable when training
continues inside the same root container. Perform loader validation inside the
container.

## Validate data and recipes

Set persistent paths:

```bash
export AFRIINSTRUCT_DATA_DIR=/data/afriinstruct/e2b-representative-v1
export AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct
export AFRIINSTRUCT_WANDB_DIR=/logs/wandb
```

Run repository validation:

```bash
uv run python tools/lint_example_yamls.py
uv run pytest \
  tests/unit_tests/tools/test_prepare_afriinstruct.py \
  tests/unit_tests/recipes/test_gemma4_afriinstruct_configs.py \
  -q
```

Load the exact generated data through the same framework builder used by the
recipe:

```bash
/opt/venv/bin/python -c \
  "from nemo_automodel.components.datasets.vlm.datasets import make_meta_dataset; \
root='/data/afriinstruct/e2b-representative-v1'; \
train=make_meta_dataset(root+'/train_meta.json', split='train'); \
validation=make_meta_dataset(root+'/validation_meta.json', split='validation'); \
assert train and validation; \
assert train[0]['conversation'][0]['role']=='user'; \
assert train[0]['conversation'][1]['role']=='assistant'; \
print('framework loader OK:', len(train), 'train,', len(validation), 'validation')"
```

There should be no `missing '_text_tokens'` warnings after precomputation. For
the old dataset revision, the expected result was:

```text
framework loader OK: 88783 train, 866 validation
```

The loader materializes the selected dataset in host memory. Observe setup
memory on the target machine before increasing the sample fraction.

## Run the 20-step LoRA qualification gate

Check that the target GPU is available immediately before launch:

```bash
nvidia-smi
```

Use a unique checkpoint directory so another experiment cannot be
auto-detected or resumed accidentally:

```bash
export AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct/qual-20-v1
```

Launch one process on one GPU:

```bash
/opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps 20 \
  --step_scheduler.ckpt_every_steps 10 \
  --step_scheduler.val_every_steps 10 \
  --lr_scheduler.lr_warmup_steps 2
```

The qualification gate passes only when:

- model and processor loading complete;
- packed dataset construction completes;
- training and validation losses are finite;
- gradient norms are finite;
- steps advance through 20;
- checkpoints are written at the requested cadence;
- the newest checkpoint can resume successfully.

Test explicit resume using the same checkpoint directory:

```bash
/opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --checkpoint.restore_from LATEST \
  --step_scheduler.max_steps 25 \
  --step_scheduler.ckpt_every_steps 5 \
  --step_scheduler.val_every_steps 5 \
  --lr_scheduler.lr_warmup_steps 2
```

Verify in the log that the restored optimizer, scheduler, RNG, and step state
continue from the checkpoint rather than starting at step zero.

## Run the qualified LoRA job

Do not launch the full 10,000-step default immediately after the first forward
pass. First record qualification throughput, peak GPU memory, host setup
memory, and validation behavior. Then choose a new versioned checkpoint
directory for the real run:

```bash
export AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct/lora-v1
```

For one GPU:

```bash
/opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1
```

For multiple GPUs in one node, set `--nproc-per-node` to the number of GPUs.
The current recipe keeps TP, CP, and PP at one; additional processes provide
data parallelism through the existing FSDP2 setup. Do not claim TP, CP, or PP
support from this recipe without separate framework validation.

Enable W&B only after `WANDB_API_KEY` is configured:

```bash
/opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --wandb.enable true
```

Retain `training.jsonl`, `validation.jsonl`, resolved configuration, dataset
`summary.json`, source revision, environment versions, and checkpoints for
each experiment.

## Promotion to full SFT

The full language-backbone recipe is:

```text
examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_full.yaml
```

It is not the next automatic step. Promote only after LoRA demonstrates:

- stable optimization and checkpoint resume;
- a reproducible improvement on held-out evaluation;
- acceptable generation quality across target languages and tasks;
- sufficient GPU memory for the much larger optimizer and gradient state.

The full recipe keeps embeddings and vision/audio towers frozen, updates the
complete language backbone, uses a lower learning rate, and requests a final
consolidated SafeTensors export for Hugging Face compatibility.

Launch only after choosing a separate checkpoint directory:

```bash
export AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct/full-v1

/opt/venv/bin/automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_full.yaml \
  --nproc-per-node 1
```

Increase the process count only after measuring memory and validating the same
topology with the LoRA gate.

## Evaluation requirements

Compare the pretrained model, LoRA checkpoints, and any full-SFT checkpoint on
fixed, source-disjoint examples. Report results by language, task, and source.
At minimum, check:

- task-appropriate metrics;
- aggregate and per-language validation loss;
- instruction following;
- empty or whitespace-only generations;
- repetition loops;
- missing EOS;
- abrupt truncation;
- unexpected language switching;
- retention on general instruction-following prompts.

Reuse the staged convergence and inference-quality process in
`examples/convergence/tulu3/` while keeping the Gemma 4 model and these recipe
settings.

## Troubleshooting history

### `Gemma4Config` import failure

Observed wrapper error:

```text
ModuleNotFoundError: Could not import module 'Gemma4Config'
```

The real nested error was:

```text
RuntimeError: operator torchvision::nms does not exist
```

PyTorch was loaded from `/opt/venv` while an incompatible torchvision was
loaded from `/usr/local`. The supported fix was not another Transformers
install. It was:

```bash
cd /opt/Automodel
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --all-groups --extra vlm
```

After synchronization, `torch`, `torchvision`, and `Gemma4Config` imported
successfully.

### Long AfriInstruct preparation

Thirty minutes or more is normal. Representative hash sampling requires a
complete streaming pass over roughly 8.95 million train records. Monitor the
process and output directory; do not restart while it is making progress.

### Hugging Face authentication warning

Unauthenticated dataset requests can still work but have lower rate limits.
Model access requires the correct account permissions and accepted Gemma
terms. Configure authentication inside the target container.

### Missing `_text_tokens`

This warning is nonfatal but means exact length metadata was not prepared.
Run `scripts/precompute_tokens.py` on both manifests before qualification.

### Nonempty output directory

The preparation utility intentionally refuses to overwrite existing output.
Use a new versioned directory or deliberately archive and remove a confirmed
partial run outside the utility. Never delete a potentially valid dataset
without inspecting it.

### Out-of-memory during training

The recipe already uses local batch size one, freezes non-language towers, and
uses LoRA. Do not enable activation checkpointing for this Gemma 4 E2B recipe.
Record the failing phase and peak allocation, then move to a larger GPU or
qualify a separately reviewed sequence-length or topology change.

## Guardrails for the next agent

- Stay on the existing NeMo AutoModel VLM recipe and component APIs.
- Do not create a replacement trainer, dataset component, model class, or
  notebook for the core workflow.
- Do not use the first `--max-samples` records for training.
- Do not silently change sampling, validation, deduplication, or packing
  policies; create a new versioned experiment and record the reason.
- Do not commit `HF_TOKEN`, caches, datasets, checkpoints, `.venv`,
  container-patched `pyproject.toml`, or container-patched `uv.lock`.
- Use `uv`; do not add `pip install` commands.
- Do not install a separate PyTorch/torchvision pair over the NVIDIA
  container stack.
- Do not start full SFT before LoRA qualification, resume, and evaluation
  pass.
- Keep each dataset and checkpoint directory versioned and immutable after it
  becomes an experiment input.
- Preserve `summary.json` and resolved run configuration with every result.
- Review the licenses of the aggregated AfriInstruct source datasets before
  publishing or redistributing derived artifacts.

## Continuation checklist

- [ ] Clone the feature branch on the new machine.
- [ ] Confirm Docker, GPU visibility, disk space, and persistent mounts.
- [ ] Apply the container PyTorch patch and run the VLM-only `uv` sync.
- [ ] Verify matching PyTorch/torchvision and `Gemma4Config` import.
- [ ] Authenticate with Hugging Face and confirm Gemma tokenizer access.
- [ ] Run the 1,000-record schema smoke test.
- [ ] Build the one-percent representative dataset.
- [ ] Inspect and preserve `summary.json` plus the resolved source revision.
- [ ] Precompute exact tokens for train and validation manifests.
- [ ] Run YAML lint and both focused CPU test files.
- [ ] Load train and validation through `make_meta_dataset` with no token
      warnings.
- [ ] Run the 20-step LoRA qualification gate.
- [ ] Resume the qualification checkpoint to step 25.
- [ ] Record throughput, memory, losses, and checkpoint evidence.
- [ ] Launch a versioned LoRA run only after every gate passes.
- [ ] Evaluate by language, task, and source.
- [ ] Consider full SFT only after LoRA produces measurable held-out gains.
