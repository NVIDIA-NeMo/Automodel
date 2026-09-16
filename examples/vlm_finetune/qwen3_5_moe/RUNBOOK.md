# Runbook — Qwen3.6-35B-A3B agentic SFT

One file for the whole run: what the recipe is, what must not be changed, how to bring it
up on 1/2/4 nodes, what every measured configuration cost, and how to export a
base-identical checkpoint.

**v4_88k, v5_130k and v6_137k are the same recipe.** They differ only in the corpus
(§4); the model, parallelism, masking, launchers and export path are shared. Everything
below applies to all of them unless a corpus is named.

Merged from the four original notes (runbook; 2×B300 bring-up 2026-09-11; 8×H200 sweep
2026-09-12; 2×8×H200 multi-node 2026-09-14), keeping only the conclusions that still
matter.

---

## 1. What this run is

Full-parameter SFT of `Qwen/Qwen3.6-35B-A3B` on a private agentic-trajectory corpus,
FSDP2 + expert parallelism + the DeepEP dispatcher. The model is a VLM with a frozen
vision tower; its text backbone is hybrid — **30 GatedDeltaNet linear-attention layers
and 10 full-attention layers** (`layer_types` cycles `[linear ×3, full]`), 256 experts /
8 active, `head_dim: 256`, `vocab_size: 248320`, `mtp_num_hidden_layers: 1`.

Three requirements. Do not trade any away for throughput without asking:

1. **Only the final assistant turn is supervised.** The corpora are turn-exploded (v4:
   87,552 rows over 9,835 unique trajectories), so supervising every turn would weight
   early turns by their ~9× duplication.
2. **Over-length rows are dropped, never truncated** — the supervised turn is at the tail.
3. **The exported checkpoint must stay architecturally identical to the base** and load
   with plain `transformers` as `Qwen3_5MoeForConditionalGeneration`.

---

## 2. Do not undo these

Each was verified against the source and each fails **silently** if reverted.

| Do not | Why |
|---|---|
| Pass `max_length` to the collator | `default_collate_fn` flips to `padding="max_length"` once it is set (`vlm/collate_fns.py:1294`); a 163-token row would pad to 40,960. The cap is enforced offline instead. |
| Set `text_config.mtp_expert_hf_layout` | Qwen3.6-35B-A3B stores MTP experts **fused**; unset, the adapter infers it. (The sibling 122B config sets `split`.) |
| Set `num_nextn_predict_layers: 0` | `self.mtp = None`, so the 19 `mtp.*` tensors never reach the export while the copied `config.json` still declares `mtp_num_hidden_layers: 1` → missing keys on load. Breaks requirement 3. |
| Enable packing while keeping `attn: te` | The model declares `_packed_cp_attn_backends = ("sdpa",)` and `supports_thd: False`; TE has no route to the 4-D block-causal mask, so attention bleeds across documents with no error. |
| Run without `flash-linear-attention` | GatedDeltaNet falls back to a pure-PyTorch reference path behind a bare `except ImportError`, **no warning** (`qwen3_5_moe/cp_linear_attn.py:126-142`) — silently degrades 30 of 40 layers. |
| Use `experts: gmm`/`te` without a DeepEP-family dispatcher | `BackendConfig.__post_init__` silently rewrites the pair to `(torch_mm, torch)`. |
| Narrow `_resolve_markers` to the thinking-enabled suffix only | See §3. |

---

## 3. Masking: the think block (decided 2026-09-11)

Rung 2 failed 8/8 on the first bring-up: `_resolve_markers` derived only the
`enable_thinking=True` suffix `<think>\n` (ids `[248068, 198]`), but a corpus without
`reasoning_content` renders every final turn as `<think>\n\n</think>\n\n` + content, and
`\n\n` is a single token (271) — so the trim never fired and the whole empty think block
stayed in the loss.

Fix in `affine/dataset.py`: derive **both** suffixes and trim whichever the supervised span
starts with, longest first. `affine/check_masking.py` accepts either.

**This is the one place the corpus changes the semantics of the run:**

- **No `reasoning_content` (v4_88k):** the whole empty block is masked, only the action
  text is supervised, and the model is trained for `enable_thinking=False` inference —
  **serve it with thinking disabled.**
- **With `reasoning_content` (v5_130k, v6_137k):** the final turn renders `<think>\n` +
  reasoning + `</think>\n\n` + content; only the opening tag is masked, so **the reasoning
  is supervised and the model is thinking-enabled** — the opposite of the v4 run.

Re-run rungs 2 and 3 whenever the corpus format changes. `affine/find_reasoning_rows.py`
answers which case a corpus is in.

---

## 4. The corpora

All are pre-filtered at 40,960 tokens with `affine/prefilter.py` (512 val rows,
seed 1234) and differ only in content.

| Corpus | rows | kept at 40,960 | mean / p50 tokens | tokens per epoch | reasoning |
|---|---|---|---|---|---|
| `vuhaian/v4_88k` | 87,552 | 87,279 (99.69%) | 11,747 / 8,805 | 1.028 B | no — thinking-disabled |
| `vuhaian/v5_130k` | ~130 k | 128,575 train | 4,664 / 3,349 | 0.600 B | yes |
| `vuhaian/v6_137k` | 137,030 | 135,491 (98.88%) | 4,179 / 2,793 | 0.564 B | yes |

v4 length distribution (Qwen3.6 tokenizer, all 87,552 rows):

| p10 | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|
| 1,537 | 3,091 | 8,805 | 18,570 | 27,271 | 31,120 | 37,097 | 45,024 | 68,378 |

Keep-rate by cap on v4: 16,384 → 70.4% · 24,576 → 85.9% · **40,960 → 99.7%** ·
49,152 → 99.96%. v6 has more long rows than v5 (949 above 32k vs 577) but the same
worst-case batch shape, so memory behaviour carries over unchanged between them.

**Signal density (v4):** supervised final-turn lengths p50 163, p90 844, p99 1,685,
max 1,801, mean 324 → 28.3M supervised tokens against 1.03B processed per epoch (2.8%),
an artifact of last-turn-only supervision on turn-exploded data. See §12.4.

---

## 5. Environment

```bash
# HF token. This repo has NO dotenv support; the repo-root .env is HF_TOKEN="hf_...".
set -a; . ./.env; set +a
export HF_HOME=/mnt/fast/hf          # 26 shards, ~67 GB — fast local disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hf download Qwen/Qwen3.6-35B-A3B     # pre-stage so ranks don't race
```

**Use the container.** `deep_ep` needs nvshmem + rdma-core + a CUDA arch list; a source
build is painful. On the multi-node cluster the image is built from `deepep564/`:

```bash
docker build -t nemo-automodel:26.08.00-deepep564 examples/vlm_finetune/qwen3_5_moe/deepep564
```

It is `nvcr.io/nvidia/nemo-automodel:26.08.00` (torch 2.13.0a0+nv26.06, CUDA 13.3,
TE 2.14.1, FLA 0.4.2, DeepEP 1.2.1+4214430, transformers 5.12.1) plus two changes:
**DeepEP PR #564 backported** (required for any `ep_size` spanning nodes, §8.1) and
**torchao pinned to 0.14.0** (§8.3).

From source instead: `uv sync --locked --all-groups --extra moe --extra vlm --extra
vlm-media`. If you drop `--all-groups`, keep `--group dev`: `FusedLinearCrossEntropy`
imports `cut_cross_entropy`, which lives only there, and the run dies with an
`ImportError` *after* the model has loaded. `--all-groups` also builds MagiAttention,
unused here.

### Bare-metal traps (hit on the B300 box; an exact `uv sync` reverts the fixes)

- Outside `uv.lock` and needed on Blackwell: `nvidia-ml-py` (DeepEP imports `pynvml`),
  `transformer-engine{,-cu13,-torch}==2.18.0`, `nvidia-cudnn-cu13>=9.23`. Build
  `transformer-engine-torch` with `NVTE_WITH_NCCL_EP=0` (its NCCL-EP code includes a
  header torch 2.10 lacks) and run `uv pip install` from **outside** the repo.
- Do **not** apt-install `libnvidia-ml-dev`: it pulls `libnvidia-compute-*` and wedges
  dpkg in a container. CUDA 13's `cuda-nvml-dev-13-0` already provides `nvml.h`.
- The CUDA toolkit's `lib64` must not be on `LD_LIBRARY_PATH` at runtime — its cuBLASLt
  shadows the pip wheel's and TE fails to import
  (`undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`).
- TE prefers a "system" cuDNN. Export
  `CUDNN_HOME=<venv>/lib/python3.12/site-packages/nvidia/cudnn` and put the venv's
  `nvidia/cudnn/lib` and `nvidia/nccl/lib` first on `LD_LIBRARY_PATH`, or TE silently
  picks `UnfusedDotProductAttention` (§6).

---

## 6. Attention on Blackwell (SM 10.x)

Keep `attn: te`, but TE's fused attention at `head_dim=256` on SM 10.x needs **TE ≥ 2.18
and cuDNN ≥ 9.23** (TE PR #3056). Without them TE logs `FusedAttention=False` and
materializes `[heads, seq, seq]` scores — measured **259 GiB extra at 32k tokens**;
rung 6 died allocating 45.5 GiB in `backward()`.

Single GPU, attention fwd+bwd at 32k tokens:

| path | time | extra memory |
|---|---|---|
| TE 2.15 unfused | 591–1216 ms | ~259 GiB |
| PyTorch SDPA flash (`--model.backend.attn sdpa`) | 93.5 ms | 2.6 GiB |
| **TE 2.18 + cuDNN 9.26 fused** | **26.6 ms** | **2.6 GiB** |

Verify with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`: the log must say
`Selected backend = FusedAttention`. (A patch to `components/attention/utils.py` that let
SDPA keep its flash kernel on right-padded batches was written, tested and reverted in
favour of the package fix.) H200 needs none of this.

---

## 7. Rungs

Run in order; 1–4 need no GPU and catch most failures.

| # | Command | Pass |
|---|---|---|
| 1 kernels | `python -c "import fla, causal_conv1d, deep_ep, transformer_engine"` | no ImportError — a failure here means a silent slow path, not a crash |
| 2 masking | `python examples/vlm_finetune/qwen3_5_moe/affine/check_masking.py --n 8` | exactly **1** supervised run per row, decoding to the final assistant message, generation-prompt `<think>` prefix excluded (§3). On v4, supervised counts p50 ≈ 163, p90 ≈ 844, max ≈ 1,801 — thousands per row means the wrapper is inert |
| 3 pre-filter | `python examples/vlm_finetune/qwen3_5_moe/affine/prefilter.py --max-seq-len 40960 --out data/<corpus>_filtered` | the keep rate in §4. A different one means the tokenizer or template changed |
| 4 config parse | `load_yaml_config(<config>)` | no `TypeError: Unexpected ... field(s)`. The VLM `dataloader` allowlist is `{shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor, drop_last}` + `collate_fn`/`_target_` — deliberately **no `batch_size`** (it comes from `step_scheduler.local_batch_size`) |
| 5 tiny proxy | `torchrun --nproc-per-node=2 -m nemo_automodel.recipes.llm.train_ft -c tests/functional_tests/parallelism/qwen3_5_moe_proxy.yaml` | 6 steps, finite decreasing loss |
| 6 smoke | `automodel <config> --nproc-per-node 8 --step_scheduler.max_steps 5 --lr_scheduler.lr_warmup_steps 1 --checkpoint.enabled false` | checklist below |
| 7 probe | `--step_scheduler.max_steps 200` | loss trends down; kill at ~150 and resume via `checkpoint.restore_from`, curve rejoins |
| 8 full run | no overrides | watch `checkpoints/<name>/training.jsonl` |
| 9 export | §10 | base-identical export loads and generates |

Any run shorter than the recipe's warmup **must** pass `--lr_scheduler.lr_warmup_steps 1`
(or `--lr_scheduler.lr_decay_steps N`): `lr_decay_steps` defaults to the step count and
`OptimizerParamScheduler` asserts `lr_warmup_steps < lr_decay_steps`
(`components/optim/scheduler.py:100`), so `max_steps 5` alone fails at setup *after* the
model has loaded.

Rung 6 checklist — check **all**:

| Check | Expected |
|---|---|
| DeepEP engaged | `grep "Falling back to standard GroupedExperts" smoke.log` finds nothing (`moe/layers.py:783`) |
| Backend survived validation | logged `backend.experts == "te"`, `dispatcher == "deepep"` |
| **Weights actually loaded** | step-0 loss ≈ **0.8–1.8**. ≈ **12.4** is `ln(248320)` = random init, i.e. the state-dict adapter matched nothing |
| Masking survived | `num_label_tokens` > 0 and stable |
| Memory | judge by `nvidia-smi`, not `torch` (§9) |

Routing health via `moe_metrics` does **not** work here: it is read only by
`recipes/llm/train_ft.py`, not the VLM recipe, and only logs to W&B.

---

## 8. Failures and fixes

### 8.1 DeepEP internode dispatch: Xid 31, then `timeout (dispatch CPU)`
Every GPU on both nodes logs `Xid 31 ... MMU Fault ... faulted @ 0x0` in the first MoE
layer, then `RuntimeError: DeepEP error: timeout (dispatch CPU)`. NCCL, `ibv_rc_pingpong`
and intra-node DeepEP all pass. Cause: the image's DeepEP (`4214430`, hybrid-ep branch)
predates [DeepEP PR #564](https://github.com/deepseek-ai/DeepEP/pull/564), which fixes the
RC-QP layout and device-state RDC for NVSHMEM ≥ 3.5 (the image ships 3.6.5); same defect
as NVIDIA-NeMo/RL #4027, and even hybrid-ep HEAD lacks it. Fix: the backport in
`deepep564/` (#564's `configs.cuh` + `ibgda_device.cuh` onto the hybrid-ep tree, plus
`docker/common/deepep.patch` for the CCCL include path, without which the build fails on
`fatal error: cuda/std/tuple`). After it, the 16-rank comms check passes (8192-token
dispatch+combine: 3.2 ms).

### 8.2 `torch.optim.AdamW`: mixed Tensor/DTensor in `_foreach_mul_`
At `ep_size == world_size` the `experts: te` weights are plain tensors while everything
else is FSDP-sharded. Fix: `optimizer.foreach: false`. Moot under `torchao.optim.AdamW8bit`.

### 8.3 `torchao.optim.AdamW8bit` needs torchao 0.14.0
0.17.0 fails on FSDP2 DTensor params inside the compiled step (`aten.view.dtype`
unimplemented; a different failure under `force_eager`). 0.14.0 matches `torch.optim.AdamW`
to 3.4e-3 relative over 3 steps. Pinned with `pip install --no-deps torchao==0.14.0`; its
"Skipping import of cpp extensions" warning is harmless.

The 8-bit optimizer is the memory win: block-wise quantized moments ≈ 8.7 GiB/GPU at EP8
on 8 GPUs vs ~17.4 GiB for bf16 moments. It needed one framework fix — torchao keeps `lr`
as a tensor and raises `RuntimeError: lr was changed to a non-Tensor object` when a
scheduler assigns a float, so `OptimizerParamScheduler.step` now updates a tensor `lr` in
place with `fill_()`. DCP save/load round-trips the quantized state bit-identically.

### 8.4 `CUDA driver error: invalid argument` from the static Triton launcher
The 8 ranks of a node share one container and one `/tmp/torchinductor_root`; concurrent
compiles corrupt kernels another rank then loads (reproduced 2/2 shared, 0/2 per-rank).
Fix: the launcher sets `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_rank${LOCAL_RANK}`.

### 8.5 Length-grouped batching is slow until the Triton cache is warm
The first 100-step run averaged 21 s/step against an expected 8, bimodal (4 s and 31 s for
same-sized batches). `py-spy` found 12–13% of wall time in **Triton autotune benchmarks**
for FLA's `l2norm_fwd` / `layer_norm_gated_fwd`: their autotune keys include a
batch-length term, length grouping gives every step a new length, and each new key costs
~30 s on the rank that hits it — which stalls all ranks at the next collective. Ruled out
first, with measurements: cross-rank batch misalignment (<0.5%), the dataloader, the
optimizer (2.6%). Fix: a **shared, persistent** Triton cache (`/mnt/fast/triton_cache`
bind-mounted, `TRITON_CACHE_DIR=/triton_cache`) while the inductor cache stays per rank —
the race is in inductor's, not Triton's. The first epoch after any change in batch shape
still pays ~25 stalls (~12 min); after that they are gone. **Judge speed only from a warm
run.**

### 8.6 Smaller items
- `global_batch_size` must be a multiple of `local_batch_size × world_size`
  (`step_scheduler.py:117`): on 16 GPUs lbs 4 → gbs 64, lbs 2 → 32; on 32 GPUs lbs 4 → 128.
- **Killing a run needs SIGKILL.** Ranks re-parent to PID 1 and hold ~100 GB through
  SIGTERM; the next launch then dies on `EADDRINUSE` (29500) while the memory sampler
  silently records the *previous* run's peak. Drain to <2000 MiB first —
  `teardown.sh` does this and waits.
- **A failed rank does not end the run**: survivors block in a collective holding all
  their memory until the 120-minute NCCL timeout. `oom_watchdog.sh` tails every
  node's log and tears down immediately.
- `save_consolidated: final` logs `v4_compatible=False`; a transformers-v4 load needs
  `--checkpoint.v4_compatible=True`.
- Syncing the tree from Windows: `git archive` emits CRLF unless run with
  `-c core.autocrlf=false`, and a CR in a launcher or YAML breaks the run confusingly
  (`sed -i 's/\r$//'`).

---

## 9. Measurements

**Read throughput correctly.** `tps` in the log is sequence positions/s summed over all
ranks *including interior padding*; `num_label_tokens` is supervised tokens only. `tps` is
comparable only between runs on the same data (longer rows amortize the fixed per-step
cost: the same config reads ~48k tok/s on a 40k-row corpus and ~22k on the real one) and
it **understates length grouping**, whose whole benefit is removing padding. **Use s/step**
— every step consumes exactly `global_batch_size` samples however they are batched.

`nvidia-smi` runs ~20–30 GiB above `torch.cuda.max_memory_allocated` (NCCL/DeepEP buffers,
cuDNN/cuBLAS workspaces, CUDA context), none of it reclaimable — **judge headroom by smi.**

### 9.1 Single node, 8 × H200 (v4, 50 steps each, gbs 32, EP8)

| lbs | batching | AC | s/step | samp/s | label tok/s | torch peak | smi peak | val @49 |
|---|---|---|---|---|---|---|---|---|
| 4 | length-grouped | full | **8.01** | 3.99 | 1,242 | 106.4 GiB | 131.3 GiB | 0.5759 |
| 2 | length-grouped | full | 9.79 | 3.27 | 1,073 | 75.3 GiB | 88.8 GiB | 0.5718 |
| **2** | **random** | **full** | **16.62** | 1.92 | 604 | 72.4 GiB | 86.9 GiB | **0.5605** |
| 4 | random | full | 18.97 | 1.69 | 529 | 105.4 GiB | 122.2 GiB | 0.5603 |
| 2 | random | selective | 16.2 † | — | — | 112.5 GiB | 130.4 GiB | — |

† stopped at step 25. The bold row is the single-node default.

- **lbs 2 beats 1** (16.62 vs 19.53 s/step, same val loss) and, with random batching,
  beats 4 — the collator pads to the longest sample in each micro-batch, so padding to the
  longest of four costs more than the larger batch gains. Padding factor vs lbs 1:
  1.47× at lbs 2, 1.95× at 4, 2.37× at 8.
- **Full AC over `selective`** (there is no intermediate rung): selective bought 2.4% for
  **+40 GiB** and reached 139.6 of 140.4 GiB on worst-case batches.
  `activation_checkpointing_scope` does not help — the vision tower is frozen and already
  skipped.
- Worst case, `longest256.parquet` (256 rows of 38.8–40.9k tokens, every batch worst case):
  lbs 2 random 96.9 GiB smi (comfortable); lbs 4 random 138.7 GiB (98.8%); lbs 4 grouped
  139.5 GiB (99.4%); lbs 2 selective 139.6 GiB; lbs 4 selective **OOM**. With random
  batching the smi footprint climbs toward the ceiling over steps; with grouping it is
  pinned there from step 1, because the sampler places the longest rows together by
  construction rather than by luck.
- Gradient clipping at `max_norm: 1.0` fires on **every step of every arm** — the clip, not
  the schedule, sets the effective step size.
- Checkpointing and validation work at lbs 4: sharded save ~80 GB in ~64 s, consolidated
  export 26 shards / ~67 GiB; validation adds no memory beyond the training peak.

### 9.2 Two nodes, 16 × H200 (v4, length-grouped, AdamW8bit, full AC)

| mesh | lbs / gbs | warm s/step | samples/s | val @99 | smi peak |
|---|---|---|---|---|---|
| EP16 | 4 / 64 | ~14–15 | 4.3 | 0.5132 | 132.6 GiB (95%) |
| EP8 | 2 / 32 | 6.5 | 4.9 | 0.5267 | 77 GiB (55%) |
| **EP8** | **4 / 64** | **7.1** | **9.0** | **0.5135** | **123.7 GiB (88%)** |
| EP16 | 8 / 128 | — | — | — | **OOM on step 0** |

- **Two nodes scale better than linearly with EP8** — 9.0 vs 4.0 samples/s (2.3×).
  Sharding dense params, grads and 8-bit state 16 ways instead of 8 frees the memory lbs 4
  needs, and each node's expert traffic stays on NVLink. What crosses the fabric is the
  dense all-gathers and the expert-grad all-reduce, each well under 0.2 s at 3.3 Tbps.
- **EP16 buys nothing here.** Expert parallelism across nodes only pays when the experts
  don't fit in one node; they do (77 GiB at lbs 2). EP16 just puts half of every
  dispatch/combine on RoCE — `py-spy` shows 22% of forward time in `internode_dispatch`.
- Validation loss is identical across EP8/EP16 at matched batch, as it should be.
- grad_norm 1.6–12.5 against the 1.0 clip, so the 2-node config halves peak LR to 5e-6.
  This is a judgement, not an A/B. (The later v6 runs from base use 5e-5 with a WSD
  schedule instead.)
- **To go faster, add nodes:** with EP8 the expert all-to-all never leaves a node, so
  4 nodes at gbs 128 is near-linear (~18 samples/s, ~1.4 h/epoch on v4). The v5/v6 configs
  do exactly this: 32 GPUs, `ep_size: 8`, MoE mesh `(ep_shard=4, ep=8)`, lbs 4 / gbs 128.
- What will not help: lbs 8 (OOM), selective AC, sequence packing (incompatible with the
  TE masking this recipe depends on), larger gbs via gradient accumulation.

### 9.3 End-to-end 266-step run (2 nodes, EP8, lbs 4, checkpointing + validation live)

| steps | mean loss | grad_norm | s/step | torch peak |
|---|---|---|---|---|
| 0–49 | 0.830 | 5.2 | 8.6 | 94 GiB |
| 50–99 | 0.655 | 2.4 | 9.0 | 83 GiB |
| 100–149 | 0.633 | 2.8 (one spike 18.3) | 9.9 | 90 GiB |
| 150–199 | 0.604 | 2.4 | 9.6 | 88 GiB |
| 200–266 | ~0.58 | 2.3 | 9.3 | 50 GiB |

Validation 0.5567 @99 → 0.5318 @199; smi peak 112.7 GiB. Autotune stalls still cost ~25%
of wall time (~14 per 50 steps); excluding them, 6.4–7 s/step. Checkpoint at step 199:
~30 s save, ~75 s pause including validation; 16 model safetensors (67 GiB, all 1,045
tensors), 16 optimizer DCP shards (13 GB), dataloader/rng/scheduler, `LATEST` and
`LOWEST_VAL` symlinks.

**The checkpoint bug this run found:** with `moe.reshard_after_forward: false` (the
single-node value) the step-200 save died with `KeyError: 11` inside DCP's
`get_optimizer_state_dict` — model shards written, optimizer not. At `ep_size 8` on 16
GPUs the experts are FSDP-sharded over the `ep_shard` mesh, FSDP2 reshards only in the
post-backward hook, so the forward-only validation that precedes each save leaves the
expert params unsharded and DCP's identity mapping fails on the first expert weight. The
single-node recipe never sees it (at `ep_size == world_size` the experts are not
FSDP-wrapped). **Set `moe.reshard_after_forward: true` on multi-node**; it costs no
throughput and lowers peak memory 124 → 113 GiB (and ~97 GiB smi at 32 GPUs on v5_130k).

---

## 10. Export (requirement 3)

The stock consolidation (`checkpoint.save_consolidated: final`) merges weights correctly
but is **not** a drop-in for the base: 60 tensors stay fp32 (`A_log`, `dt_bias` in the 30
GDN layers, marked "intrinsically fp32" in `.hf_metadata`, so even `CAST_DTYPE=bf16`
leaves them); `config.json`, `generation_config.json` and the tokenizer files are
re-serialized from the runtime config (`use_cache: false`, `output_hidden_states: true`,
renamed vision `model_type`, dropped sampling defaults); and `preprocessor_config.json`,
`video_preprocessor_config.json`, `vocab.json`, `merges.txt` are never written.

```bash
bash examples/vlm_finetune/qwen3_5_moe/export_hf.sh <ckpt>/model <out>   # ~40 s
python examples/vlm_finetune/qwen3_5_moe/affine/check_export.py <out>              # rung 9
```

The export script remaps those 60 entries to bf16, consolidates with `--cast-dtype bf16`,
overlays the base snapshot's metadata files, fixes the index `total_size`, and asserts
name/shape/dtype parity for all 1,045 tensors plus byte equality of `config.json`,
`generation_config.json`, `tokenizer_config.json`.

`affine/check_export.py` loads the result with plain `transformers`
(`Qwen3_5MoeForConditionalGeneration`, 35.107 B params), verifies the 333 frozen vision
tensors are bit-identical to the base, verifies trained tensors differ (after 200 steps
max |Δ| ≈ 5e-4), and generates. Expected layout: **1,045 tensors** — 692
`model.language_model.*`, 333 `model.visual.*`, 19 `mtp.*`, 1 `lm_head.weight`. The HF
class has no MTP module, so `state_dict()` reports 1,026 and the 19 `mtp.*` are ignored on
load, exactly as with the base. (`A_log` shows no change after 200 steps: its fp32 updates
are below bf16 resolution.)

---

## 11. Running multi-node

Cluster: `h200-32c-260914` in GCP `us-west1-c`, `a3-ultragpu-8g` nodes (8 × H200,
224 vCPUs, 32 local NVMe, 2 gVNICs + 8 RoCE NICs `gpu{0..7}rdma0`). node-2 (10.30.0.2) is
torchrun rank 0 and the NFS server; node-3 is 10.30.0.4.

Per-node storage: the 32 local SSDs are one RAID0 at `/mnt/fast` — **not in `/etc/fstab`**,
so after a reboot re-assemble (`mdadm --assemble --scan`) and re-mount; spot preemption
wipes it along with the image, data, Triton cache and checkpoints. `/mnt/fast/hf` is
`HF_HOME` (local per node); node-2 exports `/mnt/fast/shared` over NFS (`nconnect=16`)
holding the checkout, `.env`, `data/`, `logs/`, `checkpoints/` — so the recipe's relative
paths resolve identically on every node, which DCP needs (each rank writes its own shards,
rank 0 consolidates by reading all of them). Hugging Face rate-limits node-2's public IP
(HTTP 429) even with a token: do all Hub traffic from node-3; the launcher sets
`HF_HUB_OFFLINE=1`. Docker's data root is on `/mnt/fast` too (46 GB image).

```bash
# 2 nodes: rank 0 (node-2) first, from /mnt/fast/shared/Automodel
bash examples/vlm_finetune/qwen3_5_moe/launch_node.sh 0 [--key.sub value ...]
bash examples/vlm_finetune/qwen3_5_moe/launch_node.sh 1 [--key.sub value ...]

# 4 nodes: NNODES=4 per node, or fan out over SSH from rank 0's node
bash examples/vlm_finetune/qwen3_5_moe/launch_cluster.sh [--key.sub value ...]

# 16-rank NCCL + DeepEP smoke test (pass = COMM_CHECK_PASS on rank 0)
ENTRY=examples/vlm_finetune/qwen3_5_moe/affine/comm_check.py bash ...launch_node.sh 0

# worst-case memory probe
bash ... 0 --step_scheduler.max_steps 5 --lr_scheduler.lr_warmup_steps 1 \
  --checkpoint.enabled false --dataset.path_or_dataset data/v4_88k_longest/train.parquet
```

The launcher runs `torchrun` inside the container, tees rank 0 to
`logs/<timestamp>_node<rank>.log`, and takes recipe overrides after the node rank. Knobs:
`IMAGE`, `CONFIG`, `RUN_NAME`, `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `TORCHRUN_ARGS`,
`ENTRY`, `NCCL_DEBUG`, `NVSHMEM_DEBUG`, `TRITON_PRINT_AUTOTUNING`. It sets Google's gIB
NCCL plugin (`/usr/local/gib/scripts/set_nccl_env.sh`),
`NCCL_SOCKET_IFNAME=...=enp0s19`, `NVSHMEM_IB_GID_INDEX=3` (RoCE v2), per-rank
`NVSHMEM_HCA_LIST=gpu{N}rdma0` so each GPU uses its own rail, and
`--privileged --network host --ipc host --ulimit memlock=-1`.

**Always sample `nvidia-smi` alongside the run**
(`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > memlog.csv`);
`launch_cluster.sh` does it for you on every node.

> The cluster account is shared. Other sessions have installed packages under
> `/mnt/fast/shared` and once deleted `/mnt/fast/hf/hub` (67 GB of base weights) with no
> `sudo` and no log. Keep a second copy of anything you cannot re-download.

---

## 12. If it OOMs

In this order, re-running rung 6 after each: lower the pre-filter cap to 32,768 and
re-filter (still ~96.6% of v4 rows) · `--distributed.cp_size 2` with
`--step_scheduler.global_batch_size 16` (dp4/cp2 keeps `dp_size * cp_size = 8`, so
`ep_size: 8` stays legal) · `--distributed.moe.reshard_after_forward true` ·
confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Fixed memory per GPU at EP8 on 8 GPUs: params 8.8 + grads 8.7 + AdamW8bit moments ≈ 8.7
GiB, plus activations, against 141 GiB. Halve the fixed part for each doubling of the
world size; on 2 GPUs (EP2) multiply it by 4.

---

## 13. Open questions

1. **Resume from a multi-node checkpoint** (`checkpoint.restore_from`) has never been
   exercised. Needed before trusting spot recovery.
2. **The LR** is a judgement everywhere, not an A/B: clipping fires every step at 1e-5 in
   every configuration measured, the 2-node v4 config dropped to 5e-6, and the v6 runs
   from base use 5e-5 with a WSD schedule.
3. **Length-grouped vs random, ~300 steps at lbs 2.** Grouping is 1.70× faster but showed
   a consistently worse validation loss (0.5718 vs 0.5605) over 50 steps of a single seed,
   entirely inside LR warmup — weak evidence either way. It ships enabled on multi-node
   and disabled in the single-node recipe. `affine/check_sampler.py` shows what the sampler
   actually yields epoch over epoch (only the per-epoch chunk shuffle is random; batch
   membership stays length-homogeneous).
4. **Signal density.** 2.8% of processed tokens are supervised (§4). Collapsing to the
   unique full trajectories and supervising every turn would give equivalent coverage for
   ~a ninth of the compute. The user has been told and chose the current design; do not
   change it unilaterally.
5. **Sequence packing.** Rejected, not impossible: `packing_format: thd` emits `seq_lens`
   but no `attention_mask`, and the model's THD branch nulls the mask, so the 30 GDN
   layers treat a whole pack as one sequence and bleed across samples silently; `neat` +
   TE passes the document-id mask to TE as a *padding* mask, which also bleeds; `neat` +
   SDPA is correct but slower than today. Packing also bypasses the last-turn masking
   (labels are built inside `PreTokenizedDatasetWrapper`, whose only hook runs before
   labels exist). Making it work needs a model change plus a `label_post_hook`.
6. **FP8.** `te_fp8` works on Hopper elsewhere in this repo and is the next large speedup.
   No fp8 reference exists for this model family — an experiment after a green bf16
   baseline, not a default.

---

## 14. Files

| File | Purpose |
|---|---|
| `qwen3_6_35b_1node_ep8.yaml` | single-node 8×H200 recipe (EP8, lbs 2, random batching) |
| `qwen3_6_35b_ddp2.yaml` | pure-DDP 2-GPU variant (`ep_size: 1`, `dispatcher: torch`) — probe only; each rank materializes the whole model on CPU first (~13 min, single-threaded) and the fit was never confirmed |
| `qwen3_6_35b_4node_ep8_base.yaml` | 4-node run from the base model on v6_137k (5 epochs, WSD, peak LR 5e-5) |
| `affine/dataset.py` | dataset adapter (`make_affine_dataset`) + `last_turn_collate_fn` — the masking, §3 |
| `affine/check_masking.py` | rung 2 |
| `affine/check_export.py` | rung 9 |
| `affine/export_hf_helper.py` | offline consolidation used by `export_hf.sh` |
| `affine/dump_sample.py` | dump one training-ready sample with the supervised span marked |
| `affine/check_sampler.py` | what `LengthGroupedSampler` actually yields, epoch over epoch |
| `affine/find_reasoning_rows.py` | scan a corpus for rows carrying real reasoning (decides which case §3 is in) |
| `affine/sample_rows.py` | row sampler for the subset ablations |
| `affine/comm_check.py` | multi-rank NCCL all-reduce + DeepEP dispatch/combine smoke test |
| `export_hf.sh` | base-identical HF export of a sharded checkpoint (§10) |
| `launch_node.sh` | per-node launcher (container, gIB NCCL, NVSHMEM rails, caches); `NNODES` sets the node count |
| `launch_cluster.sh` | fans that launcher out over SSH and samples `nvidia-smi` |
| `oom_watchdog.sh`, `teardown.sh` | immediate teardown on OOM; drain GPUs and free port 29500 |
| `deepep564/` | patched container: DeepEP PR #564 backport + torchao 0.14 (§5, §8.1) |
| `affine/prefilter.py` | rung 3: the offline length filter and train/val split |
