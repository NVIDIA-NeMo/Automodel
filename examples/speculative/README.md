# Speculative decoding: draft-model training in NeMo AutoModel

This directory trains the **draft models** used for speculative decoding. A draft
proposes several future tokens cheaply, the frozen **target** model verifies them
in one forward pass, and every accepted token is a step the target never had to
run autoregressively. The faster and more accurate the draft, the higher the
acceptance length and the larger the inference speedup.

AutoModel trains the draft; you then serve `draft + target` together in an
inference engine (SGLang or vLLM). The training code lives in
`nemo_automodel/components/speculative/`, the recipes in
`nemo_automodel/recipes/llm/`, and ready-to-run configs in this folder.

## Supported methods

| Method | What the draft does | Recipe (`recipe:` key) | Configs |
|---|---|---|---|
| **EAGLE-1** | Single decoder block that predicts the next target hidden state; supervised by SmoothL1 hidden-state loss plus a token loss through the frozen target `lm_head`. | `TrainEagle1Recipe` | `eagle1/` |
| **EAGLE-2** | Same training objective as EAGLE-1 (it differs only in the inference-time tree policy), so the recipe is a thin subclass. | `TrainEagle2Recipe` | `eagle2/` |
| **EAGLE-3** | Draft with its own `lm_head` over a (optionally compressed) draft vocab, fed three auxiliary target hidden states, trained with a **test-time-training (TTT)** unroll. | `TrainEagle3Recipe` | `eagle3/` |
| **EAGLE-3.1** | EAGLE-3 plus two drafter toggles (`fc_norm`, `norm_output`), matching the vLLM EAGLE-3.1 architecture. | `TrainEagle3Recipe` | `eagle3_1/` |
| **P-EAGLE** | Parallel-drafting EAGLE-3: predicts all `num_depths` tokens in one forward over a COD-subsampled sequence instead of the TTT unroll. Serves on vLLM only. | `TrainEagle3Recipe` (`parallel_drafting: true`) | `p-eagle/` |
| **DFlash** | Block-parallel drafting: drafts a whole block of `block_size` tokens in one non-causal "denoising" forward over `[anchor, MASK, MASK, ...]`. | `TrainDFlashRecipe` (LLM) / `DiffusionLMSFTRecipe` (DLLM SFT) | `dflash/`, `../dllm_sft/*dflash*` |
| **Domino** | DFlash backbone plus a serial GRU correction head (`prefix_gru` / `embed_proj`) that refines each block position on the previous ones. | `TrainDominoRecipe` | `dflash/qwen3_domino.yaml` |
| **JetSpec** | DFlash backbone trained as a *causal* parallel tree drafter: causal in-block attention plus forward-KL distillation against the target distribution. | `TrainJetSpecRecipe` | `jetspec/` |
| **DSpark** | Semi-autoregressive parallel drafting: a parallel backbone drafts the block, a lightweight serial Markov head adds intra-block dependency, and a confidence head predicts per-position acceptance. | `TrainDSparkRecipe` | `dspark/` |

EAGLE-1/2/3 keep their separate code paths: `*_v12.py` files are the **EAGLE-1/2**
("v1/v2") implementation; the unsuffixed files are the EAGLE-3/3.1 path. Inside
EAGLE-3, `fc_norm`/`norm_output` upgrade to 3.1 and `parallel_drafting` upgrades
to P-EAGLE, all from the same draft class. Domino and JetSpec reuse the DFlash
draft class and checkpoint format; only their training wrappers (and, for
Domino, the extra head weights) differ.

## Supported target models

A target's `config.architectures` string selects the draft architecture through a
registry (`eagle/registry.py`, `dflash/registry.py`, `dspark/registry.py`).
Capability is per registry, not per (method, target) pair: the EAGLE-3.1
(`fc_norm` / `norm_output`) and P-EAGLE (`parallel_drafting`) toggles ride on the
same EAGLE-3 dense draft, so they apply to any target the EAGLE-3 registry maps.
The shipped example configs only cover a subset.

- **EAGLE-1/2/3 (and the EAGLE-3.1 / P-EAGLE toggles on top of EAGLE-3)**:
  `LlamaForCausalLM`, `Phi3ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`.
- **gpt-oss** (`GptOssForCausalLM`): EAGLE-3 only, via a dedicated draft class.
- **DeepSeek-V3** (`DeepseekV3ForCausalLM`): EAGLE-3 only, via a dedicated MLA
  draft class (eager attention; sequence packing not supported yet).
- **DFlash / Domino / JetSpec**: `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`.
- **DSpark**: `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`,
  `DeepseekV4ForCausalLM`, GLM-5.2 (`GlmMoeDsaForCausalLM`), Gemma4
  (`Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`),
  and MiniMax M3 VL (`MiniMaxM3SparseForConditionalGeneration`, incl. a
  multimodal data path via `recipe_args.multimodal: true`).

Qwen3-MoE is handled exactly like a dense target: the draft only consumes
post-block hidden states, never per-expert routing. gpt-oss uses a dedicated
draft class that reuses the target's YaRN rotary embedding but keeps the on-disk
`architectures` string as the Llama EAGLE-3 draft so inference engines load it
unchanged. The large-MoE DSpark targets (DeepSeek-V4, GLM-5.2, MiniMax M3) load
frozen through the same expert-parallel / FSDP paths their finetune recipes use;
see the per-target notes in `dspark/README_*.md`.

## Quick start

The CLI is `automodel` (alias `am`); it drives `torchrun` internally.

```bash
automodel examples/speculative/eagle3/llama_eagle3_mvp.yaml --nproc-per-node 8
```

Override any config key inline:

```bash
automodel examples/speculative/eagle3/llama_eagle3_perfectblend.yaml --nproc-per-node 8 --recipe_args.micro_batch_size=2
```

The DFlash **DLLM SFT** configs under `../dllm_sft/` use the standard AutoModel
SFT entry script instead:

```bash
torchrun --nproc-per-node 8 examples/dllm_sft/finetune.py -c examples/dllm_sft/qwen3_4b_dflash.yaml
```

Each config family ships an `*_mvp.yaml` (tiny single-GPU smoke with placeholder
data paths) and a `*_perfectblend.yaml` (real run on
`frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct`).

## Operators, kernels, and attention backends

Beyond the methods themselves, the subsystem supports several compute backends.
Pick them through config; all degrade gracefully when a dependency is missing.

### Draft attention backend

| Method | Backends | How to select |
|---|---|---|
| EAGLE-3 / 3.1 | `eager`, `flash_attention_2` | `recipe_args.draft_attn_implementation` (default `eager`) |
| EAGLE-1/2 | `eager` only | n/a |
| P-EAGLE | `flex_attention` (compiled when CUDA + head_dim ≥ 16, else eager flex) | automatic |
| DFlash / Domino / JetSpec | `flex_attention`, `sdpa` | `recipe_args.attention_backend` (default `flex_attention`) |
| DSpark | `flex_attention` (Qwen3 / Gemma4 / MiniMax M3 drafts), `sdpa` (V4 / GLM MLA drafts, fixed) | `recipe_args.attention_backend` |

For EAGLE-3, FlashAttention-2 is real FA2 over the TTT attention pattern: FA2
computes the `T×T` causal block and returns `softmax_lse`, and the diagonal
extension columns for cached TTT steps are merged in log space via `logaddexp`.
The draft declares `_supports_flash_attn = True`, FA2 availability is probed
defensively (`_HAS_FA`), and requesting `flash_attention_2` without `flash-attn`
installed raises rather than silently falling back. A ready example is
`eagle3/llama_eagle3_mvp_flash_attn.yaml`. FA2 requires a right-padded attention
mask (enforced at runtime).

### Fused Triton soft cross-entropy

EAGLE-3/P-EAGLE supervise the draft with a masked soft cross-entropy that uses a
**fused Triton kernel** when Triton is available and the logits are on CUDA
(`components/loss/soft_ce.py`, `components/loss/triton/soft_cross_entropy.py`),
falling back to a pure-PyTorch `log_softmax` path otherwise. The masked reduction
normalizes by valid-position count.

### Draft-vocab compression (d2t / t2d)

EAGLE-3 can shrink the draft `lm_head` to `draft_vocab_size < target_vocab_size`.
Training carries two tensors: `selected_token_ids` (draft index to target id, the
"d2t" direction) and `selected_token_mask` (a boolean membership mask over the
full target vocab, the "t2d" direction). The mapping is built and cached by
`components/datasets/llm/eagle3.py`; set `recipe_args.draft_vocab_size` to enable,
or point `recipe_args.selected_token_ids_path` at a precomputed map. `t2d` is
unset when the draft vocab is uncompressed.

### FP8 draft training

All spec-decode recipes (EAGLE-1/2, EAGLE-3 / P-EAGLE, DFlash / Domino /
JetSpec, DSpark) accept the same top-level `fp8:` block as the SFT recipes (see
`components/quantization/fp8.py`). When enabled, the draft's `nn.Linear` layers
are swapped to torchao `Float8Linear` before the DDP / FSDP2 wrap, so the
draft's forward and backward GEMMs run in fp8. Requires SM89+ (H100 or newer);
`emulate: true` runs the fp8 numerics on older GPUs for testing. The frozen
target model is never converted (it already supports fp8-quantized checkpoints
via dequant-on-load in the DSpark V4 / GLM path). Linears with a weight dim not
divisible by 16 are skipped automatically; use `filter_fqns` to exclude more
(e.g. `["lm_head"]`). On DSpark's FSDP2 path, `enable_fsdp_float8_all_gather`
plus `precompute_float8_dynamic_scale_for_fsdp` additionally amortize the
per-step scale computation, mirroring the SFT recipes. See
`eagle3/qwen3_eagle3_fp8.yaml`.

**Pair `fp8:` with `compile:`.** The recipes also accept the SFT recipes'
top-level `compile:` block (`CompileConfig`); the draft is compiled in place
(`nn.Module.compile()`, so checkpoint keys are unchanged) after the fp8 swap.
This matters for fp8 throughput: Float8Linear's per-GEMM cast/scale ops are
memory-bound and only pay off once inductor fuses them into the GEMM
prologue. In eager mode fp8 draft training is typically SLOWER than bf16
(measured 0.76x on an H100 EAGLE-3 run); compiled, the same A/B measured
fp8 at 1.03x over bf16 with byte-equivalent convergence. Expect the fp8
gain to scale with draft GEMM size: a single 4096-wide EAGLE-3 layer sits
near the float8 break-even point, while wider or deeper drafts (DSpark on
V4/GLM-scale targets) benefit more. `compile:` also works without `fp8:`
as a plain draft speedup (measured ~1.34x over the eager bf16 baseline in
the same run).

### LoRA draft adaptation (EAGLE-3 only)

The EAGLE-3 recipe accepts the SFT recipes' `peft:` block (`PeftConfig`). The
base draft is frozen and only `lora_A`/`lora_B` adapters train; checkpoints are
adapter-only (`adapter_model.safetensors` via the standard PEFT checkpoint
path). This is for adapting an existing draft to a new domain or dataset:
point `recipe_args.draft_weights_path` at the consolidated safetensors export
of a trained draft to warm-start the base weights (adapters over a randomly
initialized draft are pointless; `draft_weights_path` also works without
`peft:` for full-FT continued training). With a compressed draft vocab the
base run's token mapping must be reused via `selected_token_ids_path` (the
frozen `lm_head` rows are tied to it); a differing mapping fails fast at
load. The FINAL checkpoint of a LoRA run additionally exports the merged
draft to `model/consolidated` (serve-ready, same layout as full-FT runs), so
no external merge step is needed. Not supported with `parallel_drafting`
(P-EAGLE trains `mask_hidden` and the embeddings, which the LoRA freeze would
lock), with `freeze_embeddings: false` (same freeze conflict), with `fp8:`,
or in the DFlash-family / DSpark / EAGLE-1/2 recipes (rejected explicitly;
their drafts carry trainable non-LoRA heads that the freeze would silently
lock, and only EAGLE-3 implements the warm start). See
`eagle3/qwen3_eagle3_lora.yaml`.

## Target backends

The frozen target produces the supervision signal (aux hidden states plus the
target distribution). EAGLE-3 supports three ways to run it.

| Backend | `recipe_args.target_model_backend` | When to use |
|---|---|---|
| **Co-located (default)** | `colocated` | Target and draft share the same GPUs. Simplest; default for every config. |
| **Remote** | `remote` | Target served on separate GPUs/host; training streams supervision over HTTP (control) + NCCL (data, with a binary wire fallback). Numerically identical to co-located. |
| **Offline cache** | set `cached_target_path` | Precompute target outputs once to disk, then train without the target loaded. Disk-heavy and largely superseded by the remote backend. |

Remote serving (`eagle3/llama_eagle3_remote.yaml`): start a server first, then
point training at it.

```bash
python -m nemo_automodel.components.speculative.serve_target --target meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8001
```

```yaml
recipe_args:
  target_model_backend: remote
  remote_urls: ["http://localhost:8001"]
  target_prefetch_depth: 1
```

Offline cache is produced by `precompute_eagle3.py`
(`python -m nemo_automodel.components.speculative.precompute_eagle3 --target-model ... --input-data ... --output-dir ...`),
then consumed via `cached_target_path`. DSpark also supports a text-only offline cache via `precompute_dspark.py`
for HF-loadable single-process text targets.

For DSpark targets too large to fit on one node (DeepSeek-V4-Flash, GLM-5.2), a
**distributed** precompute (`precompute_dspark_dist.py`) loads the target frozen
through the same expert-parallel + FSDP2 path as online training and writes the
identical cache. It is config-driven and launched with `torchrun` like multi-node
training; each rank forwards its own contiguous slice of the dataset and writes its
own global-indexed shards straight into the shared `cache_output_dir` (no merge step):

```bash
torchrun --nnodes=4 --node-rank=0 --nproc_per_node=8 \
  --master-addr=<NODE0_IP> --master-port=29500 \
  -m nemo_automodel.recipes.llm.precompute_dspark_dist \
  -c examples/speculative/dspark/deepseek_v4_flash_dspark_precompute.yaml
```

Then set the matching training config's `recipe_args.cached_target_path` to that
`cache_output_dir` to train the draft with no live target. See
`dspark/deepseek_v4_flash_dspark_precompute.yaml` and
`dspark/glm_5.2_dspark_precompute.yaml`.

EAGLE drafters learn best when the assistant turns in the training data are
produced by the **same model** that will serve as the inference target. Most
public chat datasets were generated by other models, so their assistant tokens
are off-distribution for the drafter. `components/speculative/regenerate.py`
replaces those answers with fresh ones from the target model.

Use it when you want a drafter for a specific target but your only conversational
data came from a different model, or when you have a curated prompts dataset
(ShareGPT, UltraChat, an internal corpus) and want its answer distribution
aligned with the target. If a regenerated set already exists on the Hub (for
example `frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct`), skip this
and point `recipe_args.train_data_path` straight at it.

### Two-step flow

The script talks to an OpenAI-compatible chat-completion endpoint, so the target
must already be served. The examples use SGLang; vLLM or any other
OpenAI-compatible server works too.

Step 1, start the target server (use `--tp 2` or higher to shard a multi-GPU
target):

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

Step 2, regenerate against the running server:

```bash
python -m nemo_automodel.components.speculative.regenerate --input-data Aeala/ShareGPT_Vicuna_unfiltered --output-dir ./regenerated/sharegpt_llama31_8b --target-server http://localhost:30000/v1 --model meta-llama/Llama-3.1-8B-Instruct --concurrency 64 --shard-size 1000
```

For each sample the script loads the `messages` column (HF Hub id, local
parquet, or JSON/JSONL, same loader as `ChatDataset`), drops every trailing
`assistant` turn while keeping the leading `system / user / ...` context
(intermediate assistant turns in multi-turn conversations are kept), calls
`/v1/chat/completions` on the target with that prompt, appends the response as the
new assistant turn, and writes the rebuilt conversations to `shard-NNNNNN.parquet`
files of `--shard-size` rows each.

The run is resumable: rerun with the same `--output-dir` and `--resume` to skip
shards already on disk. A `manifest.json` guards resume, so changing the input
dataset, split, target model, or shard sizing fails fast instead of silently
mixing incompatible shards.

The output is a parquet dataset with a `messages` column, exactly what
`ChatDataset` (used by `build_eagle3_dataloader`) consumes. Point the recipe at
it:

```yaml
recipe_args:
  target_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  train_data_path: ./regenerated/sharegpt_llama31_8b
  val_data_path: null
```

### Regeneration tuning knobs

| Flag | Default | Notes |
|---|---|---|
| `--concurrency` | 32 | In-flight requests; raise to saturate the target server. |
| `--shard-size` | 1000 | Smaller shards mean more frequent checkpointing and more files. |
| `--max-new-tokens` | 1024 | Cap per-answer length. |
| `--temperature` | 0.0 | Greedy by default; drafters are typically trained against argmax answers. |
| `--top-p` | 1.0 | Only relevant with `temperature > 0`. |
| `--timeout-s` | 600 | Per-request timeout; bump for very long generations. |
| `--max-retries` | 3 | Retries on 5xx, 429, and transport errors with exponential backoff. |
| `--split` | `train` | Supports HF slice syntax, e.g. `train[:10000]`. |
| `--shuffle-seed` | unset | Optional shuffle before slicing. |

### Regeneration pitfalls

- **Wrong model name.** `--model` is the name sent in the OpenAI payload; it must
  match what the server serves. SGLang uses `--model-path` as the served name by
  default, so mirror `--served-model-name` here if you set it.
- **Server not warm.** Send one curl request to the server first; otherwise the
  script retries then fails on the first batch.
- **Tokenizer mismatch.** The regenerated dataset is consumed by `ChatDataset`,
  which applies the target model's chat template at training time. Make sure the
  recipe's tokenizer comes from the same model id you used for `--model`, or the
  loss-mask alignment silently drifts.

Datasets are consumed by `ChatDataset`: a `messages` list of `{role, content}`,
or a `conversations` column (ShareGPT or OpenAI style) that is auto-converted.

## Serve and benchmark a trained draft

### SGLang

After training, serve `target + draft` through SGLang:

```bash
python -m nemo_automodel.components.speculative.serve_sglang --target meta-llama/Llama-3.1-8B-Instruct --draft /path/to/run/epoch_0_step_1000/model --algorithm EAGLE3 --num-steps 3 --num-draft-tokens 4
```

`serve_sglang.py` resolves the consolidated `model/` directory, rewrites the
draft `architectures` to SGLang's canonical name, and regenerates SGLang's
speculative token map from `eagle_meta.pt` when needed. SGLang is not bundled; the
tool exits with an install hint if it is missing.

Measure acceptance length and speedup against the running server:

```bash
python -m nemo_automodel.components.speculative.bench_sglang --server http://localhost:30000 --model meta-llama/Llama-3.1-8B-Instruct --input-data <prompts-dataset> --baseline-server http://localhost:30001
```

It reports `accept_length` (mean tokens per verify step), `acceptance_rate`,
output throughput, and a `speedup` ratio versus an optional non-speculative
baseline server. Point it at a freshly started server, since SGLang reports a
server-cumulative average.

### vLLM

`serve_vllm.py` is the vLLM companion; it serves EAGLE-3, P-EAGLE (vLLM's
parallel-drafting runtime, vLLM >= 0.16), and the DFlash family (vLLM's `dflash`
method, vLLM >= 0.20):

```bash
python -m nemo_automodel.components.speculative.serve_vllm --target Qwen/Qwen3-8B --draft /path/to/run/epoch_0_step_1000 --port 8000
```

The speculative method is auto-detected from the draft config (`dflash` for a
DFlash-family draft, else `eagle3`), the draft `architectures` are rewritten to
vLLM's registered names, and `--num-speculative-tokens` defaults to `num_depths`
(P-EAGLE) or `block_size - 1` (DFlash). Pass `--print-only` to inspect the
resolved `vllm serve` command without launching. DFlash notes: vLLM reserves
`1 + K` query tokens per sequence, so a large block size may need a smaller
`--max-num-seqs` (forward it via the trailing extra args, e.g.
`-- --max-num-seqs 32`); a JetSpec draft carries `dflash_config.causal=true` so
vLLM matches its causal in-block attention (pass `--dflash-causal` for
checkpoints trained before the recipe stamped it); Domino drafts are rejected
because their GRU correction head has no vLLM runtime.

`bench_vllm.py` measures the same metrics against a running vLLM server, reading
the spec-decode counters from vLLM's Prometheus `/metrics` endpoint. The counters
are snapshotted before and after the workload and differenced, so the numbers
cover exactly the benchmark's own requests:

```bash
python -m nemo_automodel.components.speculative.bench_vllm --server http://localhost:8000 --model Qwen/Qwen3-8B --input-data <prompts-dataset> --baseline-server http://localhost:8001
```

## Inference-engine compatibility

| Draft | SGLang | vLLM |
|---|---|---|
| EAGLE-1/2/3, EAGLE-3.1 | yes | yes |
| P-EAGLE | no (tracked upstream) | yes (parallel-drafting runtime, >= 0.16) |
| DFlash, JetSpec | no | yes (`dflash` method, >= 0.20) |
| Domino | no | no (the GRU correction head has no engine runtime) |
| DSpark | no | not yet (runtime in development upstream, unreleased) |

`serve_sglang.py` rejects P-EAGLE drafts with an actionable error; serve those on
vLLM.

## Config reference (EAGLE-style schema)

EAGLE-1/2/3/3.1, P-EAGLE, and the LLM DFlash recipe share one schema. The DFlash
**DLLM SFT** configs under `../dllm_sft/` use the standard AutoModel SFT schema
(`step_scheduler` / `model._target_` / `dataset._target_` / `dllm` / `dflash`
blocks) instead.

### Top-level sections

| Section | Purpose |
|---|---|
| `recipe` | Recipe class name (required). |
| `recipe_args` | Main training block (below). |
| `dist_env` | `backend` (nccl), `timeout_minutes`. |
| `distributed` | Optional; only for MoE / large targets. `strategy: fsdp2`, `tp_size`, `pp_size`, `cp_size`, `ep_size`, `activation_checkpointing`, `sequence_parallel`. Absent means DDP. |
| `optimizer` | `lr`, `betas`, `weight_decay`, optional `warmup_ratio` (0.05), `min_lr_ratio` (0.1). |
| `checkpoint` | `enabled`, `checkpoint_dir`, `model_save_format: safetensors`, `save_consolidated`, optional `restore_from` (`LATEST` / subdir / path). |
| `fp8` | Optional; torchao FP8 draft training, same surface as the SFT recipes. See "FP8 draft training". |
| `compile` | Optional; in-place torch.compile of the draft (`CompileConfig`). Strongly recommended with `fp8`. |
| `peft` | Optional, EAGLE-3 only; LoRA draft adaptation (`PeftConfig`). See "LoRA draft adaptation". |
| `wandb` | Optional; `project`, `entity`, `name`. |

### `recipe_args` common to all methods

`target_model_name_or_path`, `train_data_path`, `val_data_path`, `train_split`,
`val_split`, `output_dir`, `seq_length`, `micro_batch_size`,
`grad_accumulation_steps`, `num_workers`, `num_epochs`, `freeze_embeddings`,
`trust_remote_code`, `shuffle_seed`, `log_every_steps`, `max_grad_norm`. Optional
checkpoint cadence: `ckpt_every_steps`, `save_checkpoint_every_epoch`.

### Method-specific `recipe_args`

| Key | Methods | Notes |
|---|---|---|
| `draft_num_hidden_layers` | EAGLE-1/2, DFlash | Stacked draft decoder layers. |
| `hidden_loss_weight`, `token_loss_weight` | EAGLE-1/2 | Defaults 1.0 / 0.1. |
| `ttt_steps` | EAGLE-3 / 3.1 | TTT unroll depth; integer ≥ 1 (required). |
| `draft_vocab_size` | EAGLE-3 family | Compress the draft `lm_head`; omit for full vocab. |
| `selected_token_ids_path` | EAGLE-3 family | Reuse a cached draft-vocab map. |
| `aux_layer_ids` | EAGLE-3 | Override the default low/mid/high recipe `[1, n//2-1, n-4]`. |
| `draft_attn_implementation` | EAGLE-3 | `eager` (default) or `flash_attention_2`. |
| `fc_norm`, `norm_output` | EAGLE-3.1 | Both default false; either alone is a valid intermediate config. |
| `draft_weights_path` | EAGLE-3 | Warm-start the draft from a consolidated safetensors export (file or directory); required for meaningful LoRA adaptation. |
| `target_model_backend`, `remote_urls`, `target_prefetch_depth`, `remote_timeout`, `remote_max_retries` | EAGLE-3 remote | See Target backends. |
| `cached_target_path` | EAGLE-3 / DSpark offline | Path to a cache produced by `precompute_eagle3.py`, `precompute_dspark.py`, or (large sharded targets) `precompute_dspark_dist.py`. |
| `parallel_drafting`, `num_depths`, `num_draft_layers`, `down_sample_ratio`, `down_sample_ratio_min`, `mask_token_id`, `sequence_partitions` | P-EAGLE | `mask_token_id` is required (no default). `sequence_partitions > 1` splits each sequence by dependency lineage to bound long-context memory. |
| `block_size`, `num_anchors`, `loss_decay_gamma`, `mask_token_id`, `target_layer_ids`, `attention_backend` | DFlash (LLM recipe) | Block drafting knobs. |
| `emb_dim`, `gru_hidden_dim`, `pure_draft_prefix_len`, `shift_label` | Domino | Correction-head knobs on top of the DFlash set. |
| `kd_temperature`, `kd_chunk_size` | JetSpec | Forward-KL distillation knobs on top of the DFlash set. |
| `markov_rank`, `markov_head_type`, `confidence_head_alpha`, `confidence_head_with_markov`, `ce_loss_alpha` | DSpark | Markov / confidence-head knobs on top of the block-drafting set (`block_size`, `num_anchors`, `mask_token_id`, `target_layer_ids`, ...). |

## Directory layout

Configs in this folder:

```
examples/speculative/
  eagle1/    eagle2/    eagle3/    eagle3_1/    p-eagle/
  dflash/                     # DFlash + Domino (qwen3_domino.yaml) configs
  jetspec/   dspark/
  README.md                   # this file (includes the dataset regeneration guide)
examples/dllm_sft/            # DFlash DLLM SFT configs (standard SFT schema)
```

Implementation:

```
nemo_automodel/components/speculative/
  eagle/        core(.py/_v12), draft_llama(.py/_v12), draft_gpt_oss,
                draft_deepseek, backend, registry, target(.py/_v12),
                peagle_*, remote/
  dflash/       core, domino_core, jetspec_core, draft_qwen3, registry, target
  dspark/       core, draft_qwen3, draft_deepseek_v4, draft_glm_5_2,
                draft_gemma4, draft_minimax_m3, markov_head, registry, target
  regenerate.py            # dataset regeneration with the target model
  precompute_eagle3.py     # offline target-output cache
  serve_target.py          # remote target server (HTTP + NCCL)
  serve_sglang.py          # serve a trained draft via SGLang
  serve_vllm.py            # serve a trained draft via vLLM (EAGLE-3 / P-EAGLE / DFlash)
  bench_common.py          # shared chat-completions workload machinery
  bench_sglang.py          # acceptance-length / speedup benchmark (SGLang)
  bench_vllm.py            # acceptance-length / speedup benchmark (vLLM)
nemo_automodel/recipes/llm/
  train_eagle1.py  train_eagle2.py  train_eagle3.py  peagle_recipe.py
  train_dflash.py  train_domino.py  train_jetspec.py  train_dspark.py
```
