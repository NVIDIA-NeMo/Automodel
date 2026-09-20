# Bring-up — `neat` packing + TransformerEngine on 2 × B300

What to run, in order, to qualify the packing change before it goes near a 4-node run.
Companion to `RUNBOOK.md`; everything there about the environment (§5), Blackwell
attention (§6) and the failure catalogue (§8) still applies. §13.5 records the design.

**State of the change:** all CPU-testable parts are green. Nothing has run on a GPU.
Three things are unverified by construction and this document is how they get settled:

1. Whether TE accepts the packed `cu_seqlens` / THD route on SM 10.x at `head_dim 256`.
2. Whether the 30 GatedDeltaNet layers stay document-isolated with the real FLA kernels.
3. What a packed step actually costs in memory and wall-clock.

A fourth is diagnostic only, and worth answering while a box is available: whether TE
*raises* on the malformed 6-D mask the old code produced, or silently accepts it. If it
accepts it, the two `neat` recipes shipped upstream
(`examples/vlm_finetune/qwen3_5/qwen3_5_4b_neat_packing.yaml` and
`qwen3_5_35b_neat_packing.yaml`, both `attn: sdpa`) are unaffected, but any user who
switched them to `attn: te` has been training with cross-document attention. That is
worth reporting upstream either way.

---

## 0. The hard gate

Packing now depends on TE's **fused** attention, because the packed route is
`qkv_format="thd"` + `cu_seqlens`. On SM 10.x at `head_dim 256` that needs **TE ≥ 2.18
and cuDNN ≥ 9.23** (RUNBOOK §6). Without them TE falls back to
`UnfusedDotProductAttention`, which materializes `[heads, seq, seq]` scores — measured at
**+259 GiB at 32k tokens**. At `pack_size 40960` that is not a slowdown, it is an
instant OOM.

Confirm before anything else:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 python -c "
import transformer_engine, torch
print('TE', transformer_engine.__version__)
print('cuDNN', torch.backends.cudnn.version())
"
```

Then, during rung P4 and P5, the TE log **must** say `Selected backend = FusedAttention`.
If it says `FusedAttention=False`, stop and fix the environment (RUNBOOK §5 has the
`CUDNN_HOME` / `LD_LIBRARY_PATH` traps that cause exactly this).

---

## 1. Rungs

Run in order. P0–P2 need no GPU.

### P0 — CPU unit tests

```bash
pytest tests/unit_tests/models/qwen3_5_moe/test_qwen3_5_moe_packed_te_attention.py \
       tests/unit_tests/datasets/vlm/test_packed_te_labels_and_mask.py \
       tests/unit_tests/loss/test_mtp_packed_seq_idx.py \
       tests/unit_tests/models/qwen3_5_moe/test_qwen3_5_moe_block_forward.py -q
```

Expect 29 + 5 passed. These cover the unpad/`cu_seqlens`/repad contract, the mask guard,
the block routing per backend, the label hook, and the MTP cross-document target masking.

### P1 — environment probes

```bash
# Kernels present (a failure here means a silent slow path, not a crash).
python -c "import fla, causal_conv1d, deep_ep, transformer_engine"

# FLA accepts the packed arguments the GDN layers pass.
python -c "
import inspect
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f
sig = inspect.signature(f)
print(sig)
assert 'cu_seqlens' in sig.parameters, 'FLA build predates packed GDN support'
print('cu_seqlens_cpu present:', 'cu_seqlens_cpu' in sig.parameters)
"
```

`cu_seqlens_cpu` is passed by four model packages in this repo. If this build lacks it,
the GDN packed path will fail loudly at P4 — do not paper over it.

Diagnostic for the fourth open question (what TE did with the old malformed mask):

```bash
python - <<'PY'
import torch
from transformer_engine.pytorch import DotProductAttention
# The shape the old code produced: a 4-D block-causal mask logical_not'ed and
# unsqueezed twice. If this returns instead of raising, TE silently ignored it.
dpa = DotProductAttention(num_attention_heads=4, kv_channels=256, attn_mask_type="padding_causal")
q = torch.randn(1, 16, 4, 256, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(1, 1, 1, 1, 16, 16, dtype=torch.bool, device="cuda")
try:
    out = dpa(q, q, q, attention_mask=mask, attn_mask_type="padding_causal")
    print("TE ACCEPTED a 6-D mask -> the old neat+te path was bleeding silently")
except Exception as exc:
    print("TE rejected it:", type(exc).__name__, exc)
PY
```

Record the answer in RUNBOOK §13.5 either way.

### P2 — data and config

```bash
# Pre-filter, if data/v6_137k_filtered does not already exist (RUNBOOK §4 keep rates).
python examples/vlm_finetune/qwen3_5_moe/affine/prefilter.py \
    --max-seq-len 40960 --out data/v6_137k_filtered

# Masking, both paths. Every row must print `collate=ok packed=ok`.
python examples/vlm_finetune/qwen3_5_moe/affine/check_masking.py \
    --dataset vuhaian/v6_137k --n 16

# Config parses and resolves the packing block.
python -c "
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes._typed_config import RecipeConfig
dl = RecipeConfig(load_yaml_config('examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml')).vlm_dataloader
print('packing', dl.packing.packing_format, dl.packing.pack_size)
print('hook', dl.pretokenization.label_post_hook.__name__)
print('inject_fake_images', dl.pretokenization.inject_fake_images)
print('sampler', dl.length_grouped_sampler)
"
```

The last command must print `neat 40960`, `last_turn_label_hook`, `False`, `None`. The
`packed=ok` column of `check_masking.py` is the one that matters here — it exercises
`last_turn_label_hook`, which is what masks a packed run. `collate=ok` alone would pass
even if packing were supervising every assistant turn.

### P3 — cross-document parity (the gate)

This is the test that decides whether the change is correct. It stacks real
`Qwen3_5MoeBlock` layers in the production `[linear ×3, full]` pattern and compares a
packed forward against each document run alone.

```bash
# The configuration the recipe runs.
python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py \
    --attn te --head-dim 256 --dtype bfloat16

# Same, forcing the SDPA reference path. Both must pass; if te leaks and sdpa does not,
# the fault is in the TE routing, not in the packing metadata.
python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py \
    --attn sdpa --head-dim 256 --dtype bfloat16

# Tighter: float32 removes bf16 rounding as an explanation for a small delta.
python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py \
    --attn te --head-dim 256 --dtype float32
```

Pass condition: `OK - all 3 documents are unaffected by being packed together`. The
per-document `max|packed - standalone|` should be at rounding level — on CPU with SDPA and
float32 it measures **2.4e-07**, which is the number to compare against. bf16 will be
nearer 1e-2; that is why the float32 run is worth doing.

A **LEAK** line localises the fault: vary `--layers` to bisect. `--layers full_attention`
isolates the TE/SDPA attention route; `--layers linear_attention` isolates the GDN state
reset.

> The SDPA + `full_attention` case has already been verified on CPU (2.4e-07). The GDN
> layers could not be checked without a real `causal_conv1d` build, so P3 is the first
> time that half runs at all.

### P4 — smoke run, 2 × B300

The packed recipe is a 4-node / 32-GPU config; on two GPUs override the parallelism. With
`world_size 2`, `ep_size 2` keeps `256 % ep_size == 0` and `dp_size × cp_size % ep_size == 0`.

Start small so iteration is fast — correctness first, shape later:

```bash
set -a; . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 \
automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml \
  --nproc-per-node 2 \
  --distributed.ep_size 2 \
  --packed_sequence.pack_size 4096 \
  --packed_sequence.collate_max_length 4096 \
  --step_scheduler.local_batch_size 1 \
  --step_scheduler.global_batch_size 2 \
  --step_scheduler.max_steps 5 \
  --lr_scheduler.lr_warmup_steps 1 \
  --checkpoint.enabled false 2>&1 | tee smoke_packed.log
```

Check **all** of these, not just that it ran:

| Check | Expected |
|---|---|
| TE fused attention | `grep "Selected backend" smoke_packed.log` says `FusedAttention` |
| No mask reshaping | no `ValueError` naming a 2-D mask (that guard firing means the model handed TE a mask instead of `cu_seqlens`) |
| DeepEP engaged | `grep "Falling back to standard GroupedExperts" smoke_packed.log` finds nothing |
| Backends survived | logged `backend.experts == "te"`, `dispatcher == "deepep"` |
| **Weights loaded** | step-0 loss ≈ **0.8–1.8**. ≈ **12.4** is `ln(248320)`, i.e. the state-dict adapter matched nothing |
| **Masking survived** | `num_label_tokens > 0` and of the same order as an unpacked run per token processed. A jump of roughly 30× means the label hook did not run and every assistant turn is supervised |
| No synthetic images | no `pixel_values` in the batch; `dataset.inject_fake_images` is false in the config and must stay so |
| Pack count logged | the packer logs how many packs it built — **write this number down**, P6 needs it |

`num_label_tokens` is the single most informative line. On v6 the supervised fraction is
a few percent of processed tokens (RUNBOOK §4); if packing silently reverted to
all-assistant-turn labels it will be far higher, and the loss curve will still look fine.

### P5 — production shape and memory

Repeat P4 at the real pack size to find the memory envelope:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > memlog.csv &

automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml \
  --nproc-per-node 2 --distributed.ep_size 2 \
  --step_scheduler.local_batch_size 1 \
  --step_scheduler.global_batch_size 2 \
  --step_scheduler.max_steps 8 \
  --lr_scheduler.lr_warmup_steps 1 \
  --checkpoint.enabled false 2>&1 | tee packed_40k.log
```

Judge headroom from `nvidia-smi`, not `torch.cuda.max_memory_allocated` — smi runs
20–30 GiB higher and none of it is reclaimable (RUNBOOK §9).

Expect the fixed cost to dominate at this world size. At EP8 on 8 GPUs it is params 8.8 +
grads 8.7 + AdamW8bit moments ≈ 8.7 GiB; each halving of the world size multiplies that by
2, so at EP2 on 2 GPUs expect roughly **4×**, on the order of 105 GiB before activations.
Activations for one 40,960-token pack should be well under the 163,840-token worst case
that measured ~113 GiB of activation on 8 GPUs. This is arithmetic from RUNBOOK §9/§12,
not a measurement — the point of P5 is to replace it with one.

**Every step is now worst-case-shaped.** Unlike the unpacked recipe there is no
occasional-peak luck: if step 1 fits, the run fits. If it does not, drop
`pack_size`/`collate_max_length` together rather than `local_batch_size`, since lbs is
already 1.

Also worth recording here: whether the bimodal step times of RUNBOOK §8.5 are gone. Fixed
pack shapes should keep FLA's autotune key constant. If they persist, `collate_max_length`
is not taking effect.

### P6 — recompute the step schedule

The committed step counts are derived from an **assumed ~97% pack fill** and are almost
certainly wrong. Take the pack count from P4/P5 and recompute:

```
epoch_len        = floor(packs / dp_world_size / local_batch_size)
total_steps      = epoch_len * num_epochs
wsd_decay_steps  = epoch_len            # anneal across exactly the last epoch
ckpt_every_steps = floor(epoch_len / 4)
lr_warmup_steps  = round(epoch_len * 0.045)
```

Update `qwen3_6_35b_4node_ep8_packed.yaml` with the 32-GPU figures before the real run.
Leaving the estimates in place would put the WSD anneal in the wrong place.

---

## 2. Then, and only then, 4 nodes

Nothing about the 4-node procedure changes: `launch_node.sh` per node with `NNODES=4`, or
`launch_cluster.sh` from rank 0 (RUNBOOK §11). Two packing-specific notes:

- **Start in a fresh `checkpoint_dir`.** The config already points at a new one. The
  sibling recipe's dataloader state was written by the length-grouped sampler, the restore
  path checks only the model config, and resuming that state into this sampler-free
  pipeline would silently mismatch the data order (RUNBOOK §13.3).
- **Keep `cp_size: 1`.** At `cp > 1` the packed path routes through block-diagonal CP,
  which requires `backend.attn: sdpa` and gives up TE fused attention.

Re-run P3 after any change to the model, the collater, or the TE version. It is cheap and
it is the only check that catches a silent leak.

---

## 3. Triage

| Symptom | Likely cause |
|---|---|
| `ValueError: attn_impl='te' expects a 2-D [batch, sequence] padding mask` | the model handed TE a mask instead of `cu_seqlens` — the block's TE routing did not fire. Check `backend.attn` really is `te` and that the batch carries `_packed_seq_ids` |
| OOM on step 0 at `pack_size 40960` | TE fell back to unfused attention (§0), or the fixed cost at EP2 is larger than estimated. Check `Selected backend` first |
| `not enough values to unpack` in `cp_linear_attn.py` | `packing_format: thd` got set somewhere. This model only supports `neat` |
| `num_label_tokens` ~30× expected | `label_post_hook_fn` is not resolving; the packed collater ignores `dataloader.collate_fn`, so nothing else masks labels |
| step-0 loss ≈ 12.4 | state-dict adapter matched nothing; unrelated to packing (RUNBOOK §7 rung 6) |
| P3 leaks on `te` but not `sdpa` | the TE THD routing — `cu_seqlens`/`max_seqlen` ordering against the flattened stream |
| P3 leaks on both | the packing metadata itself, or `_packed_seq_ids` not reaching the block |
| Bimodal step times persist | `collate_max_length` not applied, so pack shapes still vary (§8.5) |
| `pixel_values` in a text-only batch | `dataset.inject_fake_images` reverted to its `true` default |

---

## 4. What to hand back

Enough to update RUNBOOK §13.5 from "implemented, not yet run" to a measured entry:

- TE and cuDNN versions, and the `Selected backend` line.
- P1's FLA signature, and the answer to the 6-D mask diagnostic.
- P3's per-document deltas for `te` and `sdpa`, bf16 and fp32.
- P5's `nvidia-smi` peak, s/step warm, and whether autotune stalls disappeared.
- The pack count, and the recomputed step schedule.
- `num_label_tokens` from a packed step, next to the same figure from an unpacked run.
