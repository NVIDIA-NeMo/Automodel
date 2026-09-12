# Runbook — Qwen3.6-35B-A3B SFT on `vuhaian/v4_88k` (8×H200)

**Audience:** a Claude Code session with shell access on the GPU VM.
**Goal:** take `examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml` from a
cold VM to a finished, HF-loadable checkpoint.

Everything here was derived by reading this checkout and measuring the real dataset on
a machine with no GPU. Treat every "expected" value as a hypothesis to confirm, and stop
at the first rung that disagrees.

**Rungs 1-6 have since been run on 2 × B300.** See `BRINGUP_2xB300_v4_88k.md` for what
held, what did not (rung 2 failed and the masking was fixed; TE silently fell back to
unfused attention on Blackwell), the measured throughput and memory ceilings, and the
environment gotchas.

**The recipe has since been swept on 8 × H200.** See `SWEEP_8xH200_v4_88k.md` for the
measurements behind `local_batch_size: 2` and full activation checkpointing, the
worst-case memory ceilings per configuration, why `tok/s` is the wrong metric to compare
configs with, and the length-grouped batching option (1.70× faster, shipped disabled).

---

## 0. What this run is

Full-parameter SFT of `Qwen/Qwen3.6-35B-A3B` on a private agentic-trajectory corpus,
one node of 8×H200, FSDP2 + expert parallelism (`ep_size: 8`) + the DeepEP dispatcher.

Three properties the user requires. Do not trade any of them away for throughput
without asking:

1. **Only the final assistant turn is supervised.** The corpus is turn-exploded —
   87,552 rows over just 9,835 unique trajectories — so supervising every assistant
   turn would weight early turns by their ~9× duplication factor.
2. **Over-length rows are dropped, never truncated.** The supervised turn is at the
   tail; right-truncation deletes exactly the thing being trained on.
3. **The exported checkpoint must stay architecturally identical to the base** and load
   with plain `transformers` as `Qwen3_5MoeForConditionalGeneration`.

The model is a VLM with a frozen vision tower. Its text backbone is hybrid: **30
GatedDeltaNet linear-attention layers and 10 full-attention layers** (`layer_types`
cycles `[linear ×3, full]`), 256 experts / 8 active, `head_dim: 256`,
`vocab_size: 248320`, `mtp_num_hidden_layers: 1`.

---

## 1. Do not undo these

Each was verified against the source and each fails **silently** if reverted.

| Do not | Why |
|---|---|
| Pass `max_length` to the collator | `default_collate_fn` flips to `padding="max_length"` as soon as it is set (`vlm/collate_fns.py:1294`). A 163-token row would be padded to 40,960. The cap is enforced offline instead. |
| Set `text_config.mtp_expert_hf_layout` | The sibling `qwen3_5_122b_128k_ep8cp32.yaml` sets `split`, but Qwen3.6-35B-A3B stores MTP experts **fused** (`mtp.layers.0.mlp.experts.gate_up_proj`). Unset, the adapter infers it. |
| Set `num_nextn_predict_layers: 0` | `self.mtp = None`, so the 19 `mtp.*` tensors never reach the export while the copied `config.json` still declares `mtp_num_hidden_layers: 1` → missing keys on load. Breaks requirement 3. |
| Enable packing while keeping `attn: te` | `supports_sequence_packing` *accepts* `te`, and `supports_cp_with_sequence_packing` short-circuits to `True` at `cp_size <= 1` — but the model declares `_packed_cp_attn_backends = ("sdpa",)` and `supports_thd: False`. TE has no route to the 4-D block-causal mask, so attention would bleed across document boundaries with no error. |
| Run without `flash-linear-attention` | The GatedDeltaNet layers fall back to a pure-PyTorch reference path behind a bare `except ImportError`, **with no warning** (`qwen3_5_moe/cp_linear_attn.py:126-142`). That silently degrades 30 of 40 layers. |
| Use `experts: gmm`/`te` without a DeepEP-family dispatcher | `BackendConfig.__post_init__` silently rewrites the pair to `(torch_mm, torch)`. |
| Narrow `_resolve_markers` to the thinking-enabled suffix only | Final turns render `<think>\n\n</think>\n\n`, which never contains the `<think>\n` token pair, so the trim silently stops firing and the empty think block goes back into the loss. See rung 2. |

---

## 2. Environment

```bash
# HF token. This repo has NO dotenv support — nothing reads .env.
# The repo-root .env is of the form HF_TOKEN="hf_...", so parse it, do not `source` it raw.
set -a; . ./.env; set +a
[ -n "$HF_TOKEN" ] || { echo "HF_TOKEN not set"; exit 1; }

export HF_HOME=/mnt/fast/hf          # 26 safetensors shards, ~70 GB — use fast local disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Preferred: the container, which ships DeepEP and TransformerEngine prebuilt (`deep_ep`
is a git dependency needing nvshmem + rdma-core + a CUDA arch list — a painful source
build).

```bash
docker run --gpus all -it --rm --ipc=host --shm-size=32g \
  -e HF_TOKEN -e HF_HOME=/hf -v /mnt/fast/hf:/hf \
  -v $PWD:/opt/Automodel -w /opt/Automodel \
  nvcr.io/nvidia/nemo-automodel:26.06.00
# If bind-mounting this checkout, run this BEFORE uv sync or uv reinstalls torch and breaks TE:
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --all-groups --extra all
```

From source instead:

```bash
uv sync --locked --all-groups --extra moe --extra vlm --extra vlm-media
# moe -> cuda (TransformerEngine, nv-grouped-gemm, causal-conv1d)
#      + fla (flash-linear-attention)   <- 30 of 40 layers depend on this
#      + deep_ep                        <- NOT in the `all` extra
```

If you drop `--all-groups` (it also pulls the `magi` group, a long MagiAttention source
build this run does not use), keep `--group dev`: `FusedLinearCrossEntropy` imports
`cut_cross_entropy`, which is only in the `dev` group. Without it the run fails with an
`ImportError` at loss construction, after the model has already loaded.

Pre-stage the weights so eight ranks do not race the same download:

```bash
hf download Qwen/Qwen3.6-35B-A3B
```

---

## 3. Rungs

Run in order. Do not skip ahead — rungs 1–4 need no GPU and catch most failures.

### Rung 1 — kernel sanity (seconds)

```bash
python -c "import fla, causal_conv1d, deep_ep, transformer_engine; print('kernels ok')"
```

**Pass:** no ImportError. Any failure here means a silent slow path, not a crash later —
this is the single easiest thing to miss.

### Rung 2 — masking assertion (CPU, minutes) — *most important correctness check*

```bash
python examples/vlm_finetune/qwen3_5_moe/check_masking_v4_88k.py --n 8
```

**Pass:** for every conversation, the baseline collator supervises as many runs as there
are assistant turns, the wrapper leaves exactly **1** run, that run decodes to the final
assistant message, and the generation-prompt `<think>` prefix is excluded. Supervised
token counts should look like the final-turn distribution: **p50 ≈ 163, p90 ≈ 844,
max ≈ 1,801**. If they instead look like thousands of tokens per row, the wrapper is not
taking effect.

The script prints both derived generation-prompt suffixes: `'<think>\n\n</think>\n\n'`
(thinking disabled, ids `[248068, 271, 248069, 271]`) and `'<think>\n'` (thinking
enabled, ids `[248068, 198]`).

> **Masking decision (2026-09-11): the empty think block is masked.** The current
> corpus has no `reasoning_content`, so the chat template renders every final
> assistant turn as `<think>\n\n</think>\n\n` + content. `\n\n` is a single token
> (271), so the thinking-enabled `<think>\n` suffix never matches; the first
> version of the wrapper looked only for that one, trimmed nothing, and left the
> whole block in the loss (rung 2 failed 8/8). The wrapper now strips whichever
> known suffix the final span starts with, longest first. On this corpus that is
> the full empty block, so **only the content is supervised and the model is
> trained for `enable_thinking=False` inference.** Serve it with thinking disabled.

### Rung 3 — pre-filter (CPU, ~10 min)

```bash
python scripts/prefilter_v4_88k.py --max-seq-len 40960 --out data/v4_88k_filtered
```

**Pass:** ~**99.7% kept** (~260 rows dropped of 87,552); writes `train.parquet` +
`val.parquet`. A wildly different keep rate means the tokenizer or chat template
changed — stop and re-measure before training.

### Rung 4 — config parse (CPU, seconds)

```bash
python -c "
from nemo_automodel.components.config.loader import load_yaml_config
c = load_yaml_config('examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml')
print(c.to_yaml_dict())"
```

**Pass:** no `TypeError: Unexpected ... field(s)`. The VLM `dataloader` allowlist is
exactly `{shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor,
drop_last}` plus `collate_fn` / `_target_` — there is deliberately **no `batch_size`**
(it comes from `step_scheduler.local_batch_size`).

### Rung 5 — tiny proxy, 2 GPUs (minutes)

```bash
torchrun --nproc-per-node=2 -m nemo_automodel.recipes.llm.train_ft \
  -c tests/functional_tests/parallelism/qwen3_5_moe_proxy.yaml
```

**Pass:** 6 steps, finite decreasing loss. Validates the Qwen3.5-MoE block and EP
plumbing against this driver/torch stack without touching 70 GB of weights.

### Rung 6 — 5-step smoke, 8 GPUs

```bash
automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml \
  --nproc-per-node 8 \
  --step_scheduler.max_steps 5 \
  --lr_scheduler.lr_warmup_steps 1 \
  --checkpoint.enabled false \
  2>&1 | tee smoke.log
```

The warmup override is required for any run shorter than the recipe's 50 warmup steps:
`lr_decay_steps` defaults to the total step count, and `OptimizerParamScheduler` asserts
`lr_warmup_steps < lr_decay_steps` (`components/optim/scheduler.py:100`), so
`max_steps 5` alone fails at setup, after the model has already loaded.

Check **all** of:

| Check | Expected |
|---|---|
| DeepEP engaged | `grep "Falling back to standard GroupedExperts" smoke.log` finds **nothing** (`moe/layers.py:783`) |
| Backend survived validation | logged `backend.experts == "te"` and `dispatcher == "deepep"` — `BackendConfig.__post_init__` rewrites invalid pairs silently |
| **Weights actually loaded** | step-0 loss ≈ **0.8–1.8**. ≈ **12.4** is `ln(248320)` = random init, i.e. the state-dict adapter matched nothing |
| Masking survived | `num_label_tokens` > 0 and stable across steps |
| Memory | peak ≤ ~95 GiB/GPU (predicted ~80) |
| Routing | with `moe_metrics.enabled`, expert load not collapsing onto a few experts |

**Record tokens/s here.** It is the baseline for deciding whether the packing variant
(see §5) is worth its cost.

### Rung 7 — 200-step convergence probe

```bash
automodel <config> --nproc-per-node 8 --step_scheduler.max_steps 200
```

**Pass:** loss trends down; `grad_norm` mostly under the 1.0 clip (persistent clipping →
halve the LR); validation loss at 100/200 decreasing. Then kill at ~150 and resume via
`checkpoint.restore_from` and confirm the curve rejoins.

### Rung 8 — full run

Remove the overrides. ~2,728 optimizer steps/epoch (87.3k rows ÷ global batch 32),
`num_epochs: 2`. Watch `checkpoints/qwen3_6_35b_v4_88k_ep8/training.jsonl`.

### Rung 9 — export check (the hard requirement)

`save_consolidated: final` copies the **original** `config.json` and
`generation_config.json` and saves the tokenizer — but **not** the processor configs.

```bash
BASE=$HF_HOME/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/
OUT=checkpoints/qwen3_6_35b_v4_88k_ep8/consolidated
cp $BASE/preprocessor_config.json $BASE/video_preprocessor_config.json $OUT/

python -c "
import transformers, json, glob
m = transformers.AutoModelForImageTextToText.from_pretrained('$OUT', dtype='bfloat16', device_map='auto')
print(type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, 'B')"
```

**Pass:** loads as `Qwen3_5MoeForConditionalGeneration`, ~35.3B params, **1045 tensors**
(692 `model.language_model.*`, 333 `model.visual.*`, 19 `mtp.*`, 1 `lm_head.weight`).
The vision tower was frozen, so those 333 tensors must be bit-identical to base. If
`mtp.*` is missing, MTP was disabled somewhere — that violates requirement 3.

---

## 4. If it OOMs

In this order, re-running rung 6 after each:

1. `--packed_sequence` untouched; lower the pre-filter cap to 32,768 and re-filter
   (still keeps ~96.6% of rows).
2. `--distributed.cp_size 2 --step_scheduler.global_batch_size 16` — dp4/cp2 keeps
   `dp_size * cp_size = 8`, so `ep_size: 8` stays legal.
3. `--distributed.moe.reshard_after_forward true`.
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (should already be set).

Predicted fixed memory per GPU on 8 GPUs: params 8.8 + grads 8.7 + optimizer moments
≈ 8.7 GiB with `torchao.optim.AdamW8bit` (1 byte per moment per param; torch AdamW on
the bf16 params would keep bf16 moments, ≈ 17.4 GiB), plus activations. H200 has 141 GiB.
On 2 GPUs (EP2) multiply the fixed part by 4.

**Blackwell (B200/B300, SM 10.x): keep `attn: te`, but TE's fused attention needs
TE >= 2.18 and cuDNN >= 9.23.** cuDNN fused-attention backward for head_dim 256 on
SM 10.x arrived in TE 2.18 (TE PR #3056) and needs cuDNN 9.23 for BSHD. Without it TE
silently selects `UnfusedDotProductAttention`, which materializes `[heads, seq, seq]`
scores: ~45 GiB for a ~39k-token row, ~259 GiB at 32k in isolation.

- Install TE 2.18.0 (build `transformer-engine-torch` with `NVTE_WITH_NCCL_EP=0`: its
  NCCL-EP code includes a header that torch 2.10 lacks) and `nvidia-cudnn-cu13>=9.23`.
  Both are outside `uv.lock`, so an exact `uv sync` reverts them.
- Point TE at the pip cuDNN: `export CUDNN_HOME=<venv>/lib/python3.12/site-packages/nvidia/cudnn`.
  TE prefers a "system" cuDNN and otherwise dlopens `libcudnn.so`, which on the NGC-style
  base image is an apt cuDNN 9.8. Also put the venv's `nvidia/cudnn/lib` and
  `nvidia/nccl/lib` first on `LD_LIBRARY_PATH`.
- Confirm with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`: expect `Selected backend = FusedAttention`.
  Measured on B300 at 32k tokens (16/2 heads, d=256, fwd+bwd): TE fused 26.6 ms / 2.6 GiB,
  vs. PyTorch SDPA flash 93.5 ms / 2.6 GiB.

---

## 5. Known-open questions — ask before deciding

- **Packing.** Not enabled. It would fit ~4.2 samples per 40,960-token pack with zero
  padding, but it forces `attn: sdpa` *and* bypasses the masking wrapper (labels are
  built inside `PreTokenizedDatasetWrapper`, and `dataloader.collate_fn` is overridden
  by the packing branch at `vlm/loader.py:277`). Enabling it needs a ~6-line
  `label_post_hook` mirroring the existing `post_tokenize_hook_fn`. Only worth it if
  rung 6 throughput disappoints.
- **FP8.** `te_fp8` works on Hopper (`examples/llm_benchmark/qwen/qwen3_moe_30b_te_fp8.yaml`)
  and would be the next large speedup. No fp8 reference exists for this model family —
  treat as an experiment after a green bf16 baseline, not a default.
- **Signal density.** 28.3M supervised tokens against 1.03B processed per epoch (2.8%),
  an artifact of last-turn-only supervision on turn-exploded data. Collapsing to the
  9,835 full trajectories and supervising every turn would give equivalent coverage for
  roughly a ninth of the compute. The user has been told and chose the current design;
  do not change it unilaterally.
- **Dataset revision.** The corpus is migrating to a format where the `THOUGHT:` prose
  moves into `reasoning_content` and only the action stays in `content`. Measured
  impact: sequences ~3% shorter, keep-rate at 40,960 goes 99.73% → 99.82%, and the
  masking design is unaffected (the wrapper derives the generation-prompt prefixes from
  the tokenizer, so it adapts automatically). Note the semantics change, though: final
  turns will render `<think>\n` + reasoning, so only `<think>\n` is masked and the
  reasoning becomes supervised — the model is then trained for thinking-enabled
  inference instead of thinking-disabled (see the rung 2 decision note). Re-run rungs 2
  and 3 when it lands.

---

## 6. Reference numbers (measured, current corpus)

Full sequence token lengths over all 87,552 rows, Qwen3.6 tokenizer:

| p10 | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|
| 1,537 | 3,091 | 8,805 | 18,570 | 27,271 | 31,120 | 37,097 | 45,024 | 68,378 |

mean 11,747 → **1.028B tokens/epoch**. Final-turn (supervised) lengths: p50 163,
p90 844, p99 1,685, max 1,801, mean 324 → **28.3M supervised tokens/epoch**.

Keep-rate by cap: 16,384 → 70.4% · 24,576 → 85.9% · **40,960 → 99.7%** · 49,152 → 99.96%.
