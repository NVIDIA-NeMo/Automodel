# Bring-up log — `qwen3_6_35b_v4_88k_ep8.yaml` on 2 × B300

Companion to `RUNBOOK_v4_88k.md`. The runbook says what to run; this file records what
actually happened when the recipe was first brought up on hardware, which of its
assumptions held, and what had to change. Written 2026-09-11.

**Hardware:** one RunPod node, 2 × NVIDIA B300 SXM6 (SM 10.3, 268 GiB each), driver
580.126.09, Ubuntu 24.04, 344 logical CPUs. The runbook targets 8 × H200, so every run
below uses `--nproc-per-node 2 --distributed.ep_size 2` (256 experts % 2 == 0,
`dp_size * cp_size = 2`).

---

## 1. Environment

The container shipped CUDA 12.8; `uv.lock` pins `torch==2.10.0+cu130`. CUDA 13.0 was
installed from the NVIDIA apt repo and selected with
`update-alternatives --set cuda /usr/local/cuda-13.0`, plus `rdma-core libibverbs-dev`
for DeepEP.

`uv sync` was run in two phases, because the MoE extra contains source builds that need
torch already present:

```bash
uv sync --locked --extra vlm --extra vlm-media --extra fla
uv sync --locked --extra vlm --extra vlm-media --extra moe
uv sync --locked --inexact --extra vlm --extra vlm-media --extra moe --group dev
```

Build environment for the source packages (DeepEP, TE-torch, causal-conv1d,
nv-grouped-gemm): `TORCH_CUDA_ARCH_LIST="10.0;10.3"`, `NVTE_CUDA_ARCHS="100;103"`,
`CPATH=$CUDA_HOME/include/cccl` (CUDA 13 moved CCCL, and DeepEP's host sources need it).

### Packages outside `uv.lock`

An exact `uv sync` reverts all of these; reinstall after any sync.

| Package | Why |
|---|---|
| `nvidia-ml-py` | DeepEP built from the pinned rev imports `pynvml`, which is not in the lock |
| `transformer-engine{,-cu13,-torch}==2.18.0` | fused attention at `head_dim=256` on SM 10.x landed in TE 2.18 (TE PR #3056) |
| `nvidia-cudnn-cu13==9.26.0.51` | that path needs cuDNN ≥ 9.23 (BSHD); torch pins 9.15.1.9 |

`transformer-engine-torch` must be built with `NVTE_WITH_NCCL_EP=0`: its optional NCCL-EP
extension includes `torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp`, which
torch 2.10 does not ship. Run `uv pip install` from **outside** the repo, otherwise
`[tool.uv]` build settings in `pyproject.toml` apply and the resolve fails.

### Traps that cost real time

- **`libnvidia-ml-dev` must not be apt-installed.** It pulls `libnvidia-compute-*` and
  `nvidia-persistenced`, which cannot unpack in a container ("Invalid cross-device link")
  and leave dpkg wedged. CUDA 13's `cuda-nvml-dev-13-0` already provides `nvml.h`.
- **The CUDA toolkit's `lib64` must not be on `LD_LIBRARY_PATH` at runtime.** Its cuBLASLt
  13.1 shadows the pip wheel's 13.4 and TE fails to import with
  `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`.
- **TE prefers a "system" cuDNN over the venv one.** It globs `$CUDNN_HOME` first, then
  falls back to `dlopen("libcudnn.so")`, which on this image is an apt cuDNN 9.8. With
  that loaded, TE silently selects `UnfusedDotProductAttention` (see §3), and mixing the
  apt NCCL 2.25 in gives `undefined symbol: ncclCommWindowDeregister`. Fix: export
  `CUDNN_HOME=<venv>/lib/python3.12/site-packages/nvidia/cudnn` and put the venv's
  `nvidia/cudnn/lib` and `nvidia/nccl/lib` first on `LD_LIBRARY_PATH`.
- **Dropping `--all-groups` drops `cut_cross_entropy`.** The recipe's
  `FusedLinearCrossEntropy` imports it, and it lives only in the `dev` group, so the run
  dies with an `ImportError` *after* loading the model. (`--all-groups` itself is best
  avoided: it pulls the `magi` group and builds MagiAttention, unused here.)
- **DeepEP works on B300.** `tests/test_intranode.py --num-processes 2` passes, so the
  `dispatcher: hybridep` workaround from the `*_gb200.yaml` benchmarks is not needed.
- Killing a run: rank processes re-parent to PID 1 in their own process group, so
  `kill -- -PGID` misses them; they keep ~170 GB of GPU memory and the next run OOMs for
  no reason. Kill `recipes/vlm/finetune.py` PIDs explicitly and wait for ~0 MiB.

---

## 2. Correctness: the empty think block (rung 2)

Rung 2 failed 8/8 on first run: `FAIL generation-prompt prefix was not excluded from the
loss`.

`_resolve_markers` derived the generation-prompt suffix with `add_generation_prompt=True`
and got `<think>\n` (ids `[248068, 198]`). But this corpus has no `reasoning_content`, so
the chat template renders every final assistant turn as `<think>\n\n</think>\n\n` +
content — and `\n\n` is a single token (271), so `[248068, 198]` never matches. The trim
was gated on that comparison, so it never fired and the whole empty think block stayed in
the loss.

Fix (in `v4_88k.py`): derive **both** suffixes, `enable_thinking=True` (`<think>\n`) and
`enable_thinking=False` (`<think>\n\n</think>\n\n`), and trim whichever the supervised
span starts with, longest first. `check_masking_v4_88k.py` accepts either.

**Consequence, decided by the corpus owner:** on the current corpus the whole empty block
is masked, so only the action text is supervised and **the model is trained for
`enable_thinking=False` inference — serve it with thinking disabled.** When the
`reasoning_content` revision lands, final turns will render `<think>\n` + reasoning, only
the opening tag will be masked, and the reasoning becomes supervised: that flips the
model to thinking-enabled inference. Re-run rungs 2 and 3 when it does.

After the fix: 8/8 pass, ~4 fewer supervised tokens per row (p50 ≈ 160).

---

## 3. Attention on Blackwell

The recipe's `attn: te` was silently degrading. With TE 2.15 + cuDNN 9.15, TE logs
`Available backends = {FlashAttention=False, FusedAttention=False,
UnfusedDotProductAttention=True}` for this model's full-attention shape (16 heads / 2 KV,
`head_dim=256`; flash-attn is not installed). The unfused path materializes
`[heads, seq, seq]` scores — measured **259 GiB extra at 32k tokens** — and rung 6 died in
`backward()` trying to allocate 45.5 GiB for a ~39k-token row.

Two fixes were measured, single GPU, attention forward + backward:

| path | 32k tokens | extra memory |
|---|---|---|
| TE 2.15 unfused | 591–1216 ms | ~259 GiB |
| PyTorch SDPA flash (`--model.backend.attn sdpa`) | 93.5 ms | 2.6 GiB |
| **TE 2.18 + cuDNN 9.26 fused** | **26.6 ms** | **2.6 GiB** |

The recipe keeps `attn: te`; the fix is the package upgrade plus `CUDNN_HOME` (§1). Verify
with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` — the run log must say
`Running with FusedAttention backend`, and must not say `UnfusedDotProductAttention`.

A code change to `components/attention/utils.py` (skip the explicit mask for right-padded
batches so SDPA keeps its flash kernel) was written, tested and then **reverted** in favour
of the package fix, to keep upstream attention behaviour. TE handles padded batches
natively, so nothing in the model needed to change.

---

## 4. Optimizer: 8-bit AdamW

`torch.optim.AdamW` on bf16 parameters keeps bf16 moments (~17.4 GiB/GPU at EP8). The
recipe now uses `torchao.optim.AdamW8bit` (block-wise quantized moments, ~8.7 GiB).

This needed one framework fix: torchao's optimizers keep `lr` as a tensor and raise
`RuntimeError: lr was changed to a non-Tensor object` when a scheduler assigns a float.
`OptimizerParamScheduler.step` now updates a tensor `lr` in place with `fill_()` and is
unchanged for float `lr` (`components/optim/scheduler.py`, plus a unit test).

Verified before use: steps correctly on plain, DTensor and 1-D parameters; DCP
save/load round-trips the quantized state bit-identically (rung 7's resume depends on it);
warmup/cosine decay produce finite, correctly scheduled values over 6 steps.

---

## 5. Rung results

| Rung | Result |
|---|---|
| 1 kernels | pass — `fla` 0.4.2, `causal_conv1d` 1.6.0, `deep_ep` (+HybridEP), TE 2.18, `grouped_gemm`, `cut_cross_entropy`. The FLA/causal-conv1d imports resolve, so the 30 GatedDeltaNet layers are not on the reference path |
| 2 masking | fail 8/8 → fixed (§2) → pass 8/8 |
| 3 pre-filter | pass — 87,279 / 87,552 kept (99.69%), 273 dropped, +512-row val split |
| 4 config parse | pass |
| 5 tiny proxy | pass — 6 steps, finite decreasing loss |
| 6 smoke (5 steps) | pass — step-0 loss 0.94 (not ≈12.4, so weights loaded), val 0.68, `num_label_tokens` stable, no `Falling back to standard GroupedExperts`, backend stayed `te`/`deepep` |
| 7 200-step probe | **not run** (blocked, §8) |

Two runbook corrections found here:

- Any run shorter than 51 steps also needs `--lr_scheduler.lr_warmup_steps 1`.
  `lr_decay_steps` defaults to the step count and `OptimizerParamScheduler` asserts
  `lr_warmup_steps < lr_decay_steps`, so `--step_scheduler.max_steps 5` alone fails at
  setup, after the model has loaded.
- The routing-health check cannot work as written: `moe_metrics` is only read by
  `recipes/llm/train_ft.py`, not the VLM recipe, and only logs to W&B.

---

## 6. Performance (FSDP2 + EP2, TE fused attention, AC on, AdamW8bit)

Real shuffled data, `global_batch_size: 32`, steady state over steps 1–5:

| `local_batch_size` | s/step | real tokens/s | peak GiB/GPU |
|---|---|---|---|
| 1 | ~79 | ~5,000 | 138.6 |
| 2 | ~64 | 4,800–7,000 | 166.2 |
| 4 | not measured | — | — |

Memory ceiling, probed with a worst-case dataset of the 256 longest rows (38.8k–40.9k
tokens each, so every micro-batch is worst case from step 0):

| `local_batch_size` | worst-case peak | verdict |
|---|---|---|
| 2 | 168 GiB | fits |
| 4 | ~192 GiB | fits |
| 8 | — | **OOM** (needed ~9.5 GiB more) |

Padding cost of plain shuffled batching, computed over all 86,767 rows from the
pre-filter's `n_tokens`:

| `local_batch_size` | tokens processed / real tokens | padding |
|---|---|---|
| 1 | 1.00× | 0% |
| 2 | 1.47× | 32% |
| 4 | 1.95× | 49% |
| 8 | 2.37× | 58% |

`tps` in the logs counts real tokens only (`labels.numel()` minus tail padding), so the
speed column above is directly comparable across batch sizes.

At `local_batch_size: 2` one epoch is ~2,711 optimizer steps ≈ 48 h, so 2 epochs ≈ 4 days
on this node. Before the TE fix the same run was ~165 s/step (≈10 days).

`grad_norm` ran 3.1–7.3 against the 1.0 clip in every short run. Five steps is too early
to judge; the runbook's rule (halve the LR if clipping persists) applies at rung 7.

### Options examined and set aside

- **Sequence packing / padding-free.** `packing_format: thd` emits `seq_lens` but no
  `attention_mask` / `_packed_seq_ids`, and the model's THD branch nulls the mask — the 30
  GatedDeltaNet layers would then treat a whole pack as one sequence and bleed across
  samples silently (consistent with `supports_thd: False`). `neat` + TE passes the
  document-id mask to TE as a *padding* mask, which also bleeds; `neat` + SDPA is correct
  but slower than today. Packing also bypasses the last-turn masking, since labels are
  built inside `PreTokenizedDatasetWrapper` and its only hook runs before labels exist.
  Making THD correct needs a model change (derive GDN boundaries from `cu_seqlens`) plus a
  label hook.
- **Length-grouped batching.** The repo's two `LengthGroupedSampler`s sort globally and
  reuse the same sample pairings every epoch; the VLM one also assumes each rank already
  holds its own shard. Neither is wired into the VLM loader.
- **Not applicable here:** fused RoPE (force-disabled globally, see #3027),
  `compile_attn` (requires `attn: sdpa` + `linear: torch`), CUDA graphs (sequence lengths
  vary per micro-batch), `rms_norm` backend (this model hard-codes `Qwen3NextRMSNorm`).
- **Unmeasured:** activation checkpointing off / `selective`, `defer_fsdp_grad_sync`,
  `experts` gmm/torch_mm, `hybridep`, async dispatch, `dispatcher_num_sms`, and
  FP8/MXFP8 (`te_fp8`, numerics-changing — needs an explicit decision).

---

## 7. Pure DDP variant

`qwen3_6_35b_v4_88k_ddp2.yaml` was added to test data-parallel replication instead of
sharding: `strategy: ddp`, `ep_size: 1`, `dispatcher: torch`, `experts: torch_mm` (DDP
cannot use an EP mesh — `mesh_utils` requires `ep_size=1`), `gradient_as_bucket_view: true`,
`local_batch_size: 1`. FSDP2's HSDP route is not an alternative on 2 GPUs either:
`dp_replicate_size == dp_size` is rejected.

Findings so far: each rank materializes the whole model on CPU before moving it to its
GPU, which takes **~13 minutes** at ~5 GB/min. That is single-threaded Python-side weight
loading, so raising `OMP_NUM_THREADS` does not help (measured: 104% CPU either way; the
setting is still worth having for CPU tensor ops, since torchrun otherwise forces 1).
Estimated ~210 GB/GPU for params + grads + 8-bit moments before activations. **Whether it
fits is still unknown** — the probe hit the hardware fault below.

---

## 8. Why this stops here: GPU 0 hardware fault

At ~09:10 the DDP probe failed during checkpoint load with
`CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error`.
`nvidia-smi` on GPU 0 (`00000000:9A:00.0`) reports `GPU Recovery Action: Reset`,
`Channel Repair Pending` / `TPC Repair Pending: GPU requires reset`, 5 uncorrectable
retired pages pending, and **all NVLinks inactive**. GPU 1 is healthy. NVLink worked
earlier the same day, so the fault appeared mid-session.

The node needs a host-level GPU reset (or replacement) before any further training: every
configuration here — FSDP2 sharding, DeepEP dispatch, DDP's parameter broadcast — depends
on that link, and pending uncorrectable memory errors risk silent corruption in a
multi-day run.

**Health check before trusting the node again:**

```bash
nvidia-smi -q -d ECC,ROW_REMAPPER | grep -iE 'pending|uncorrectable'   # no pending repairs
nvidia-smi nvlink -s                                                   # links active on both GPUs
python <DeepEP>/tests/test_intranode.py --num-processes 2              # ~1 min
```

Then re-run rung 1 and a 5-step smoke before starting rung 7.

---

## 9. Next steps

1. Repair/replace the node, run the health checks above.
2. Optional: finish the `local_batch_size: 4` speed run and the DDP fit probe.
3. Rung 7 (200 steps) at the chosen batch size: watch the loss trend, `grad_norm` against
   the clip, validation at 100/200, then kill at ~150 and resume via
   `checkpoint.restore_from` to confirm the curve rejoins.
4. Rungs 8 and 9 per the runbook. Remember to copy `preprocessor_config.json` and
   `video_preprocessor_config.json` from the base snapshot into the consolidated export.
