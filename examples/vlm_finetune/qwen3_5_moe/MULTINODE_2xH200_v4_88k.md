# Multi-node bring-up — `qwen3_6_35b_v4_88k` on 2 × 8 × H200 (GCP A3 Ultra)

Companion to `RUNBOOK_v4_88k.md`, `BRINGUP_2xB300_v4_88k.md` and `SWEEP_8xH200_v4_88k.md`.
This file records taking the recipe from one node to two: the cluster setup, the container
build, every failure met on the way and its fix, the measured configurations, and the
recommended one. Written 2026-09-14.

**Hardware:** nodes 2 and 3 of the `h200-32c-260914` spot cluster in GCP `us-west1-c`,
`a3-ultragpu-8g` each: 8 × H200 SXM (143,771 MiB), 224 vCPUs, 2.9 TB RAM, 32 local
NVMe SSDs, 2 gVNICs (`enp0s19` control, `enp192s20`) + 8 RoCE NICs (`gpu{0..7}rdma0`,
one per GPU, MTU 8896). Driver 570.211.01 (open kernel module), Ubuntu 22.04. Nodes 0
and 1 were left alone (node-0 was running someone else's vLLM throughout).

| Node | Public IP | Private IP (`enp0s19`) | Role |
|---|---|---|---|
| node-2 | 34.19.18.95 | 10.30.0.2 | torchrun rank 0, NFS server |
| node-3 | 34.83.63.18 | 10.30.0.4 | rank 1, NFS client |

**Result:** the recipe runs across both nodes at **7.1 s/step for 64 samples (9.0
samples/s), 2.3× the single-node sweep's best**, with `ep_size: 8`, `local_batch_size: 4`,
`global_batch_size: 64`, length-grouped batching, at 88% of GPU memory. The shipped config
for it is `qwen3_6_35b_v4_88k_2node_ep8.yaml`.

---

## 1. Node setup

Everything below is per node unless stated. `node_setup.sh` (not committed; the commands
are inline here) did it with `sudo`.

> **The `chilaingoc` account is shared.** During the run in §6, other SSH sessions on
> node-2 (from two other IPs, working in `/mnt/fast/shared/120_Affine`) installed packages
> and created directories under `/mnt/fast/shared`, and `/mnt/fast/hf/hub` (the 67 GB of
> base weights) was deleted by something outside this work; it needed no `sudo`, so it
> left no log. The weights were restored from node-3's copy. Coordinate GPU use with
> whoever else has the account, and keep a second copy of anything you cannot re-download.

### 1.1 Storage

The boot disk is 194 GB. The 32 local SSDs (375 GB each, `nvme_card*`, no partitions)
became one RAID0:

```bash
apt-get install -y docker.io nvidia-container-toolkit mdadm nfs-common ibverbs-utils
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
DEVS=$(lsblk -dn -b -o NAME,SIZE | awk '$2 == 402653184000 {print "/dev/"$1}')
mdadm --create /dev/md0 --level=0 --raid-devices=32 $DEVS --run
mkfs.ext4 -F -m 0 -E lazy_itable_init=1,lazy_journal_init=1 /dev/md0
mkdir -p /mnt/fast && mount -o noatime /dev/md0 /mnt/fast
```

Docker's data root moved to `/mnt/fast/docker` (`/etc/docker/daemon.json`,
`"data-root"`); the image is 46 GB on disk. **The array is not in `/etc/fstab`**: after a
reboot it must be re-assembled (`mdadm --assemble --scan`) and re-mounted. Spot preemption
wipes it, and with it the image, the data, the Triton cache and any checkpoints.

Layout on `/mnt/fast`:

| Path | What |
|---|---|
| `/mnt/fast/hf` | `HF_HOME`; the 26 safetensors shards (67 GB) of `Qwen/Qwen3.6-35B-A3B`, local on each node |
| `/mnt/fast/shared` | node-2 exports it over NFS (`rw,async,no_subtree_check`, `nconnect=16` on the client); node-3 mounts it at the same path |
| `/mnt/fast/shared/Automodel` | the repo checkout, `.env` (`HF_TOKEN=...`), `data/v4_88k_filtered/`, `logs/`, `checkpoints/` |
| `/mnt/fast/shared/build/deepep564` | container build context (§2) |
| `/mnt/fast/triton_cache` | persistent Triton kernel/autotune cache, bind-mounted into every container (§4.5) |

The recipe's relative `data/` and `checkpoints/` paths therefore resolve identically on
both nodes, which DCP needs: each rank writes its own shards and rank 0 consolidates.

### 1.2 Data and weights

- `hf download Qwen/Qwen3.6-35B-A3B` into `/mnt/fast/hf` (30 s on node-3; copied to node-2
  over NFS). **Hugging Face rate-limits node-2's public IP (HTTP 429) even with a token**;
  do all Hub traffic from node-3. The launcher sets `HF_HUB_OFFLINE=1`.
- Pre-filter (rung 3 of the runbook), run in the container on node-3:
  `python scripts/prefilter_v4_88k.py --max-seq-len 40960 --out data/v4_88k_filtered`
  → 87,279 of 87,552 rows kept (99.69%), **86,767 train / 512 val**.
- Masking check (rung 2): `check_masking_v4_88k.py --dataset data/v4_88k_filtered
  --split train --n 64` → 64/64 pass, after the dual-suffix fix that is now upstream
  (`v4_88k.py` masks both `<think>\n` and `<think>\n\n</think>\n\n`; the first version
  masked only the former and failed 15/16).
- `data/v4_88k_longest/{train,val}.parquet`: the 192 + 32 longest rows (38,976–40,921
  tokens), for worst-case memory probes.

### 1.3 The checkout on the nodes

`/mnt/fast/shared/Automodel` is a git clone of
`https://github.com/trungvd-zenai/Automodel.git`, branch
`trungvd-zenai/feat/qwen3-6-v4-88k-sft`, on the NFS share, so both nodes see one tree. To
update it: `cd /mnt/fast/shared/Automodel && git pull`. `git status` should show only
`logs/` untracked (`.env`, `data/` and `checkpoints/` are ignored). Files written by the
container (`logs/`, `data/`, `checkpoints/`) are owned by root.

During the bring-up the tree was synced from a Windows checkout instead; if that is ever
done again, note that `git archive` from Windows emits CRLF unless run with
`-c core.autocrlf=false`, and a CR in `launch_2node_v4_88k.sh` or a YAML breaks the run
in confusing ways (`sed -i 's/\r$//'` fixes a file).

### 1.4 Driver option (applied, not needed)

`/etc/modprobe.d/nvidia-ibgda.conf` with
`options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`
was applied on nodes 2 and 3 (module reload, no reboot) while chasing the DeepEP internode
failure, because DeepEP's NVSHMEM guide prescribes it. It was **not** the fix (§4.1) and
is harmless; it is still in place.

---

## 2. Container

`nvcr.io/nvidia/nemo-automodel:26.08.00` (torch 2.13.0a0+nv26.06, CUDA 13.3, TE 2.14.1,
FLA 0.4.2, DeepEP 1.2.1+4214430, torchao 0.17.0, transformers 5.12.1) runs on the host's
570 driver through CUDA forward compatibility. It needed two changes, both in
`deepep564/Dockerfile`; build it on every node:

```bash
docker build -t nemo-automodel:26.08.00-deepep564 examples/vlm_finetune/qwen3_5_moe/deepep564
```

1. **DeepEP PR #564 backported** onto the image's DeepEP commit (`4214430`, hybrid-ep
   branch). Required for any `ep_size` that spans nodes (§4.1). The patch is
   `deepep564/deepep_4214430_backport564.patch`: #564's two files (`configs.cuh`,
   `ibgda_device.cuh`) resolved onto the hybrid-ep tree, plus the repo's
   `docker/common/deepep.patch` (CCCL include path), without which the build fails with
   `fatal error: cuda/std/tuple`. Built with `HYBRID_EP_MULTINODE=0` and
   `TORCH_CUDA_ARCH_LIST=9.0` (160 s).
2. **torchao pinned to 0.14.0** (§4.3).

Environment inside the container, set by the launcher: `source
/usr/local/gib/scripts/set_nccl_env.sh` (Google's gIB NCCL plugin for RoCE, bind-mounted
from the host), `NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enp0s19`,
`NVSHMEM_IB_GID_INDEX=3` (RoCE v2), per-rank `NVSHMEM_HCA_LIST=gpu{N}rdma0` so each GPU
uses its own rail, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`--privileged --network host --ipc host --ulimit memlock=-1`.

---

## 3. How to run

From `/mnt/fast/shared/Automodel` on **each** node, rank 0 (node-2) first:

```bash
# node-2
bash examples/vlm_finetune/qwen3_5_moe/launch_2node_v4_88k.sh 0 [--key.sub value ...]
# node-3
bash examples/vlm_finetune/qwen3_5_moe/launch_2node_v4_88k.sh 1 [--key.sub value ...]
```

The launcher runs `torchrun --nnodes 2 --nproc-per-node 8` inside the container with the
config `qwen3_6_35b_v4_88k_2node_ep8.yaml`, tees rank 0's output to
`logs/<timestamp>_node<rank>.log`, and accepts recipe overrides after the node rank.
`IMAGE`, `CONFIG`, `MASTER_ADDR`, `MASTER_PORT`, `TORCHRUN_ARGS`, `NCCL_DEBUG`,
`NVSHMEM_DEBUG` and `TRITON_PRINT_AUTOTUNING` are environment knobs.

Useful variants:

```bash
# NCCL + DeepEP comms smoke test (16 ranks, pass = COMM_CHECK_PASS on node-2)
ENTRY=examples/vlm_finetune/qwen3_5_moe/comm_check_2node.py bash examples/vlm_finetune/qwen3_5_moe/launch_2node_v4_88k.sh 0

# Short probe: any run shorter than 50 steps needs the warmup override
bash ... 0 --step_scheduler.max_steps 5 --lr_scheduler.lr_warmup_steps 1 --checkpoint.enabled false

# Worst-case memory probe on the longest rows
bash ... 0 --step_scheduler.max_steps 5 --lr_scheduler.lr_warmup_steps 1 --checkpoint.enabled false \
  --dataset.path_or_dataset data/v4_88k_longest/train.parquet \
  --validation_dataset.path_or_dataset data/v4_88k_longest/val.parquet
```

Operational rules learned the hard way:

- **Sample `nvidia-smi` alongside the run** (`nvidia-smi --query-gpu=index,memory.used
  --format=csv,noheader,nounits -l 2 > memlog.csv`); the recipe's `mem` field is
  `torch.cuda.max_memory_allocated` and sits 20–30 GiB below the real footprint.
- **A failed rank does not end the run.** When some ranks OOM, the survivors wait in a
  collective for the 120-minute NCCL timeout while holding all their memory. Watch the
  logs for `OutOfMemoryError`/`Traceback` and tear both nodes down
  (`docker rm -f v4_88k_node0` / `node1`); wait for GPU memory to drain to < 2 GB and port
  29500 to free before relaunching.
- **Judge speed only from a warm run** (§4.5). The first run after any change to batch
  shape or parallelism carries ~25 autotune stalls of ~30 s.

---

## 4. Failures and fixes, in the order they were hit

### 4.1 DeepEP internode dispatch: Xid 31, then `timeout (dispatch CPU)`

Symptom (EP16, first MoE layer): every GPU on both nodes logs `Xid 31 ... MMU Fault ...
faulted @ 0x0`, then `RuntimeError: DeepEP error: timeout (dispatch CPU)` after 100 s.
NCCL across nodes was fine (16-rank 1 GiB all-reduce: 4.6 ms, ~3.3 Tbps bus bandwidth
over gIB), `ibv_rc_pingpong` passed on all 8 rails, and DeepEP within one node passed
(8-rank dispatch/combine round trip 1.9 ms). NVSHMEM's IBGDA transport initialized
cleanly; the `PeerMappingOverride` driver option changed nothing.

Cause: the image's DeepEP (`4214430`, hybrid-ep branch) predates
[DeepEP PR #564](https://github.com/deepseek-ai/DeepEP/pull/564) (Jan 2026), which
fixes the RC-QP layout and device-state RDC for NVSHMEM ≥ 3.5; the image ships NVSHMEM
3.6.5. Same defect as [NVIDIA-NeMo/RL #4027](https://github.com/NVIDIA-NeMo/RL/issues/4027).
Even hybrid-ep HEAD lacks it.

Fix: the backport in §2. Afterwards the 16-rank comms check passes (dispatch + combine of
8192 tokens: 3.2 ms per round trip).

### 4.2 `torch.optim.AdamW`: mixed Tensor/DTensor in `_foreach_mul_`

With the plain AdamW the first optimizer step raised `aten._foreach_mul_.Scalar got mixed
torch.Tensor and DTensor`: at `ep_size == world_size` the `experts: te` weights are plain
tensors while everything else is FSDP-sharded. Fix is `optimizer.foreach: false`, as in
`examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep_gb200.yaml`. Moot once the recipe
moved to `torchao.optim.AdamW8bit`, which does not batch across params.

### 4.3 `torchao.optim.AdamW8bit` on torchao 0.17.0

The image's torchao 0.17.0 fails on FSDP2 DTensor params inside AdamW8bit's compiled
step: `OptimState8bit dispatch: attempting to run unimplemented operator/function:
func=aten.view.dtype`; with `torch.compiler.set_stance("force_eager")` it fails
differently (`expected dtype torch.bfloat16 for 'end', but got torch.float32`). Both
reproduce on a 2-GPU `fully_shard` toy model. torchao 0.14.0 (the version the 8×H200
sweep validated) works on this torch: 3 steps match `torch.optim.AdamW` to 3.4e-3
relative. Pinned in the image with `pip install --no-deps torchao==0.14.0`; the
"Skipping import of cpp extensions" warning it prints is harmless (the optimizer is pure
Python + `torch.compile`).

### 4.4 `CUDA driver error: invalid argument` from the static Triton launcher

After the pin, some ranks died in AdamW8bit's compiled kernel with `CUDA driver error:
invalid argument` (`static_triton_launcher.py`) while the other ranks of the same node,
holding identical shard shapes, were fine. Cause: the 8 ranks of a node share one
container and one `/tmp/torchinductor_root`; concurrent compiles corrupt kernels that
another rank then loads. Reproduced 2/2 with a shared inductor cache and 0/2 with a
per-rank one, on an 8-GPU probe that sharded tiny params (`(5,)`, `(8,)`, `(16,)`) so
every rank held sub-block and empty shards.

Fix: the launcher sets `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_rank${LOCAL_RANK}`.

### 4.5 Length-grouped batching is slow until the Triton cache is warm

The first 100-step EP16 run averaged 21 s/step against the sweep's 8 s for the same
per-GPU shape, and the timeline was bimodal: same-sized batches took 4 s on one step and
31 s on another. Ruled out in turn, with measurements: cross-rank batch misalignment
(simulating the sampler on the real data: every rank's padded length within 0.5% at every
step), the dataloader (collating 4 of the longest rows takes 1.3 s; workers idle), the
optimizer (2.6% of main-thread time).

`py-spy` on rank 0 showed 12–13% of wall time inside **Triton autotune benchmarks**
(`autotuner.run → check_disk_cache → benchmark → do_bench`) for FLA's `l2norm_fwd` and
`layer_norm_gated_fwd`, at step 50 and beyond. Their autotune keys include a batch-length
term, and with length grouping every step has a new length; each new key costs ~30 s of
benchmarking on the rank that hits it, and that rank stalls all 16 at the next collective.
With a warm cache the same shape ran at 6.5 s/step (1 stall in 28). Two things made this
worse here than in the sweep: the fix in §4.4 had put the Triton cache per rank *inside*
the container, so nothing was shared across ranks or across runs, whereas the sweep box
kept `~/.triton/cache` across all its runs.

Fix: the Triton cache is shared and persistent, `/mnt/fast/triton_cache` on the host
bind-mounted as `/triton_cache`, `TRITON_CACHE_DIR=/triton_cache`; the inductor cache
stays per rank. The same 8-GPU probe passed 3/3 with this split (the race is in the
inductor cache, not Triton's). `TRITON_PRINT_AUTOTUNING=1` on a warm run showed only the
MoE permute kernels benchmarking (32 events × 0.9 s at startup). The first epoch after a
change in batch shape still pays ~25 stalls (~12 min); after that they are gone.

### 4.6 Smaller items

- Runs shorter than 50 steps need `--lr_scheduler.lr_warmup_steps 1`
  (`assert lr_warmup_steps < lr_decay_steps`), or `--lr_scheduler.lr_decay_steps N` for
  a probe that should keep the real warmup.
- `global_batch_size` must be a multiple of `local_batch_size × 16` on 16 GPUs
  (`step_scheduler.py:117`): lbs 8 needs gbs 128, lbs 4 → 64, lbs 2 → 32.
- `parse_run.py`-style log parsing must strip the tqdm prefix that precedes each
  `step N |` line on the same line.

---

## 5. Measurements

All runs: 16 GPUs, `torchao.optim.AdamW8bit`, full activation checkpointing, `attn/linear/
experts: te`, `dispatcher: deepep`, length-grouped batching, checkpointing off, LR 1e-5
with 50-step warmup and cosine decay over the epoch. "s/step" is wall-clock between
consecutive step log lines on rank 0; "warm" excludes autotune stalls (steps > 12 s).

### 5.1 EP16 (experts across both nodes)

| lbs / gbs | steps | s/step (all) | warm s/step | samples/s | val @99 | torch peak | smi peak | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 / 32, longest rows | 5 | 17 (steps 1–4) | — | 1.9 | 0.641 | 47 GiB | 64 GiB | smoke test, loss 0.89→0.56 |
| 8 / 128 | 0 | — | — | — | — | — | 137–140 GiB | **OOM on step 0** on 3 ranks |
| 4 / 64 | 100 | 21.2 | ~14–15 | 4.3 | **0.5132** | 103 GiB | 132.6 GiB (95%) | stalls on 75/98 steps (cold cache) |

`py-spy` on the lbs 4 run: 22% of forward time waiting in `internode_dispatch`, i.e. the
RoCE all-to-all is a real cost on top of the stalls.

### 5.2 EP8 (experts within each node; DP, FSDP and expert-grad reduction across both)

| lbs / gbs | steps | s/step (all) | warm s/step | samples/s | val @99 | torch peak | smi peak | notes |
|---|---|---|---|---|---|---|---|---|
| 2 / 32 | 100 | 13.0 | 9.0–9.9 (2nd half) | 3.3–3.6 | 0.5267 | 61.5 GiB | 77 GiB (55%) | first run at this shape: quarters 17.4 → 15.2 → 9.9 → 9.0 |
| 2 / 32 (warm cache) | 30 | **6.5** | 6.5 | **4.9** | — | 53 GiB | — | 1 stall in 28 steps |
| **4 / 64** | 100 | 12.7 | **7.1** (72 steps) | **9.0** | **0.5135** | 106 GiB | **123.7 GiB (88%)** | 25 stalls (534 s of 1,220 s); loss 0.895→0.581, last-10 mean 0.574 |

Reference from `SWEEP_8xH200_v4_88k.md` (8 GPUs, one node): lbs 2 grouped 9.79 s/step
(3.3 samples/s), lbs 4 grouped 8.01 s/step (4.0 samples/s).

### 5.3 Reading the table

- **Two nodes scale better than linearly with EP8:** 9.0 vs 4.0 samples/s (2.3×). Sharding
  the dense params, gradients and 8-bit optimizer state 16 ways instead of 8 frees the
  memory that lbs 4 needs (124 GiB here vs 131 GiB on 8 GPUs at 99% for the same shape),
  and each node's expert traffic stays on NVLink. What crosses the fabric per step is the
  dense-param all-gathers and the expert-gradient all-reduce, each well under 0.2 s at
  the measured 3.3 Tbps.
- **EP16 buys nothing here.** Expert parallelism across nodes only pays when the experts
  don't fit within one; they do (77 GiB at lbs 2). EP16 just moves half of every
  dispatch/combine onto RoCE: 4.3 samples/s at 95% memory vs 9.0 at 88%.
- **Validation loss is the same** across EP8 and EP16 at matched batch (0.5135 vs 0.5132),
  as it should be: the global batch and gradient synchronization are identical.
- **Gradient clipping fired on every step of every run** (grad_norm 1.6–12.5 against
  `max_norm: 1.0`), so at LR 1e-5 the clip sets the effective step size. The shipped
  2-node config halves the peak LR to 5e-6; this is a judgement, not an A/B.

---

## 6. End-to-end run with checkpointing (the recommended config)

`qwen3_6_35b_v4_88k_2node_ep8.yaml` (`ep_size: 8`, `local_batch_size: 4`,
`global_batch_size: 64`, `length_grouped_sampler` on, `AdamW8bit`, `max_lr: 5e-6`,
`moe.reshard_after_forward: true`) was run for 266 of a planned 350 steps with checkpointing
and validation live, then stopped to exercise the export.

### 6.1 The checkpoint bug

The first attempt (with `moe.reshard_after_forward: false`, the single-node recipe's value)
died at the step-200 save with `KeyError: 11` inside DCP's `get_optimizer_state_dict`:
the model shards were written, the optimizer was not. Mechanism: at `ep_size 8` on 16 GPUs
the experts are FSDP-sharded over the 2-way `ep_shard` mesh; FSDP2 reshards only in the
post-backward hook, so the forward-only validation pass that precedes each save
(`validate_on_checkpoint`) leaves the expert parameters unsharded, and DCP's identity
mapping between optimizer params and `model.named_parameters()` fails on the first expert
weight. The single-node recipe never sees this because at `ep_size == world_size` the
experts are not FSDP-wrapped. With the flag on, saves after validation complete (a 4-step
probe with saves at steps 1 and 3, then the real run). The flag costs no throughput
(8.6 s/step either way) and lowers peak memory from 124 to 113 GiB (80%).

### 6.2 Results (266 steps)

| block | mean loss | grad_norm mean | s/step | torch peak |
|---|---|---|---|---|
| 0–49 | 0.830 | 5.2 | 8.6 | 94 GiB |
| 50–99 | 0.655 | 2.4 | 9.0 | 83 GiB |
| 100–149 | 0.633 | 2.8 (one spike to 18.3) | 9.9 | 90 GiB |
| 150–199 | 0.604 | 2.4 | 9.6 | 88 GiB |
| 200–266 | ~0.58 | 2.3 | 9.3 | 50 GiB |

Validation 0.5567 @ 99 → 0.5318 @ 199. nvidia-smi peak 112.7 / 114.4 GiB. Stalls
(§4.5) still account for ~25% of wall time this run, ~14 per 50 steps; they are the
same autotune events and fade with the shared cache, but slower than a single-run
measurement suggests. Speed excluding them: 6.4–7 s/step (9–10 samples/s).

**Checkpoint at step 199:** ~30 s save, ~75 s pause including validation. Contents: 16
sharded model safetensors (67 GiB, all 1,045 tensors, byte count equal to the base),
16 optimizer DCP shards (13 GB; 8-bit state), dataloader, rng, scheduler, `config.yaml`,
`losses.json`; `LATEST` and `LOWEST_VAL` symlinks. No `.incomplete` marker.

### 6.3 Export

The stock consolidation (`checkpoint.save_consolidated: final`, or the `consolidate.sh`
the checkpoint carries) merges the weights correctly (26 shards in the base's layout,
1,045 tensors, every shape right, 37 s with 16 CPU workers) but the result is **not** a
drop-in for the base, which the runbook requires:

- 60 tensors are fp32 (`A_log`, `dt_bias` in the 30 GDN layers; the model keeps them fp32
  and `.hf_metadata/fqn_to_dtype_mapping.json` marks them "intrinsically fp32", so even
  `CAST_DTYPE=bf16` leaves them). The base ships them bf16.
- `config.json`, `generation_config.json` and the tokenizer files are re-serialized from
  the runtime config: `use_cache: false`, `output_hidden_states: true`, the vision
  `model_type` renamed, `transformers_version` bumped, and the sampling defaults
  (`do_sample`, `temperature`, `top_k`, `top_p`, the two-entry `eos_token_id`) dropped.
- `preprocessor_config.json`, `video_preprocessor_config.json`, `vocab.json`, `merges.txt`
  are not written.

`export_hf_v4_88k.sh <ckpt>/model <out>` produces the base-identical layout: it remaps
those 60 entries to bf16 in the dtype mapping, runs the offline consolidation with
`--cast-dtype bf16`, overlays the base snapshot's metadata files, fixes the index's
`total_size`, and asserts name/shape/dtype parity for all 1,045 tensors plus byte
equality of `config.json`, `generation_config.json` and `tokenizer_config.json`. ~40 s.

`check_export_v4_88k.py <out>` is rung 9: loads the export with plain `transformers`
(`Qwen3_5MoeForConditionalGeneration`, 35.107 B params), checks the 333 frozen vision
tensors are bit-identical to the base, checks trained tensors differ (after 200 steps:
max |Δ| ≈ 5e-4), and generates. On the step-199 export: **all pass**, greedy output to
"what does `git rebase --onto` do?" was a correct one-sentence answer. (`A_log` shows no
change: its fp32 updates over 200 steps are below bf16 resolution.)

Note the HF model class has no MTP module, so `state_dict()` reports 1,026 tensors; the
19 `mtp.*` tensors are in the files and are ignored on load, exactly as with the base.

### 6.4 Still open

1. **Resume.** `checkpoint.restore_from` on `epoch_0_step_199` has not been exercised
   (loss/LR/step should continue seamlessly). Needed before trusting spot recovery.
2. **The lower LR** is a judgement, not an A/B.

To go faster: add nodes. With EP8 the expert all-to-all never leaves a node, so nodes 0
and 1 would add near-linearly (`--nnodes 4`, gbs 128): ~18 samples/s, ~1.4 h/epoch.

What will not help on two nodes, measured or derived: lbs 8 (OOM), selective activation
checkpointing (+40 GiB for 2.4% in the sweep), sequence packing (incompatible with the TE
attention masking the recipe depends on, runbook §5), larger gbs via gradient
accumulation (same throughput, fewer optimizer steps).

---

## 7. Files

| File | Purpose |
|---|---|
| `qwen3_6_35b_v4_88k_2node_ep8.yaml` | recommended 2-node config |
| `qwen3_6_35b_v4_88k_ep16_2node.yaml` | EP16 variant, kept as the measured reference |
| `launch_2node_v4_88k.sh` | per-node launcher (container, gIB NCCL, NVSHMEM rails, caches) |
| `comm_check_2node.py` | 16-rank NCCL all-reduce + DeepEP dispatch/combine smoke test |
| `export_hf_v4_88k.sh`, `export_hf_v4_88k_helper.py` | base-identical HF export of a sharded checkpoint (§6.3) |
| `check_export_v4_88k.py` | rung 9: load the export with `transformers`, compare to base, generate |
| `deepep564/Dockerfile`, `deepep564/deepep_4214430_backport564.patch` | the patched image |
| `logs/prof_rank0_*.txt` on node-2 | the `py-spy` profiles behind §4.5 (not committed) |
