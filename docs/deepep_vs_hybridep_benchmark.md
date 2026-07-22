# DeepEP vs HybridEP: Performance Comparison and Analysis

**Date:** 2026-07-22
**Author:** benchmark series run on cw-dfw-cs-001 (H100 80GB SXM, CX7 400Gb/s IB)
**Code:** `deepseek-ai/DeepEP` — `v1.2.1` tag (DeepEP), `hybrid-ep` branch @ 17cfb81 (HybridEP, NVIDIA), `epv2-release` branch (DeepEP v2, included as secondary reference)
**Logs:** `deepep/logs/`, reusable sbatch scripts in `deepep/slurm_*.sub`

---

## 1. Executive summary

We compared DeepEP (v1) and HybridEP at three levels: (a) synthetic kernel microbenchmarks
(mock routing), (b) real Qwen3-30B-A3B training on a single node (EP8), and (c) real training
across 2 nodes (EP16), all on the same H100 cluster with identical configs per A/B pair.

| Level | Config | DeepEP | HybridEP | HybridEP gain |
|---|---|---|---|---|
| Kernel, intranode EP8, DSv3 dims | 8 SMs, BF16 dispatch | 168 GB/s | **360 GB/s** | 2.1× at equal SMs |
| Kernel, intranode EP8, DSv3 dims | saturated | 335 GB/s @ 24 SMs | **367 GB/s @ 8 SMs** | ~10% peak, **3× fewer SMs** |
| Kernel, internode EP16, DSv3 dims | BF16 dispatch (RDMA) | 84.9 GB/s @ 24 SMs | **85.7 GB/s @ 8 SMs** | parity at 1/3 SMs |
| Kernel, internode EP16, DSv3 dims | combine (RDMA) | 68.2 GB/s @ 24 SMs | **76.5 GB/s @ 8 SMs** | +12%, 1/3 SMs |
| **Real training, 1 node EP8** | Qwen3-30B, GBS128 | 5.366 s/step | **5.085 s/step** | **+5.5% throughput** |
| **Real training, 2 nodes EP16** | Qwen3-30B, GBS256 | 5.906 s/step | **5.472 s/step** | **+7.9% throughput** |

Bottom line: peak bandwidth of both libraries sits near the hardware limit; HybridEP's real
advantages are (1) reaching that limit with ~1/3 of the SMs, (2) collapsing the per-layer
kernel pipeline from 6 kernels + 1 CPU sync down to 4 kernels with no CPU sync, and
(3) absorbing the expert-permutation into the communication path. In end-to-end training this
compounds to +5.5% (single node) and +7.9% (cross-node), with identical loss trajectories.
Switching in NeMo Automodel is one config line: `model.backend.dispatcher: hybridep`.

---

## 2. Test environments and methodology

- **Hardware:** 8×H100 80GB (NVLink, ~160 GB/s/dir per pair via NVSwitch) per node; CX7
  400 Gb/s InfiniBand per GPU pair for internode; nodes from the `batch` partition.
- **Software:** NGC 26.04 (torch 2.12) pod for intranode runs; `mb2604.sqsh`
  (Megatron-Bridge, torch 2.11, deep_ep 1.2.1+17cfb81 preinstalled with NVSHMEM internode +
  DOCA GPUNetIO multinode) for internode SLURM jobs. The same `deep_ep` wheel provides both the
  classic `Buffer` (DeepEP) and `HybridEPBuffer`, removing install-skew from the comparison.
- **Microbenchmark harness:** the repo's own tests (`tests/test_intranode.py`,
  `tests/test_internode.py`, `tests/test_hybrid_ep.py`), modified only to sweep SM counts and
  skip redundant correctness passes. Two dimension sets:
  - **DSv3-like:** 4096 tokens/rank, hidden 7168, top-8, 8 experts/rank.
  - **Qwen3-30B-like:** 4096 tokens/rank, hidden 2048, top-8, 128 experts total.
- **Apples-to-apples accounting:** DeepEP numbers use the *cached-handle* dispatch (no layout
  or notify cost — closest to pure kernel time); HybridEP numbers use its *kernel-only* timing
  (bench_kineto). All three implementations count traffic the same way (per-token deduplicated
  destination bytes, local rank included), so relative comparisons are valid.
- **Real training:** NeMo Automodel `qwen3_moe_30b_te_deepep` benchmark recipe
  (Qwen3-30B-A3B, seq 4096, TE attention/linear, balanced gate, mock dataset, activation
  checkpointing). Only `model.backend.dispatcher` differs between runs; 15 steps, mean of the
  10 post-warmup steps. Loss/grad-norm trajectories matched step-for-step between backends.

---

## 3. Microbenchmark results (mock routing)

### 3.1 Intranode EP8, DSv3 dims (hidden 7168) — bandwidth vs #SMs

Kernel-level GB/s (dispatch FP8 / dispatch BF16 / combine BF16):

| SMs | DeepEP | HybridEP | DeepEP comb | HybridEP comb |
|----:|-------------|--------------|----:|----:|
| 4 | – | 189 / 197 | – | 222 |
| 8 | 148 / 168 | **352 / 360** | 151 | **344** |
| 16 | 282 / 322 | 358 / 362 | 284 | 357 |
| 24 | 323 / 335 | 358 / 367 | 326 | 364 |
| 32 | 315 / 308 | – | 322 | – |

Reading: DeepEP's bandwidth is proportional to SM count until it saturates NVLink at ~24 SMs.
HybridEP saturates at **8 SMs** and is flat beyond that; at 4 SMs it already matches DeepEP's
8-SM figure. Saturated peak is also ~10% higher (combine +12%).

API-level (what a framework actually sees), 24 SMs BF16 dispatch: HybridEP non-fused 274 GB/s,
**fused-permute 339 GB/s** — the fused mode recovers most of the gap between API and raw kernel
by eliminating the separate permute pass.

DeepEP's non-cached full path additionally pays: layout kernel 39 µs + notify_dispatch with a
**CPU busy-wait** ~35 µs per MoE layer (measured), and it breaks CUDA-graph capture.

### 3.2 Internode EP16 (2×8 H100), DSv3 dims — RDMA bandwidth

| | SMs | dispatch FP8 | dispatch BF16 | combine |
|---|---:|---:|---:|---:|
| DeepEP (NVSHMEM/IBRC) | 24 | 79.5 | 84.9 | 68.2 |
| HybridEP (DOCA GPUNetIO) | **8** | 81.6 | **85.7** | **76.5** |
| DeepEP v2 (NCCL Gin), reference | 24 | 84.0 | – | 72.0 |

Dispatch saturates the NIC for everyone; the separations are combine (+12% for HybridEP) and,
again, the SM budget (8 vs 24).

### 3.3 Qwen3-30B dims (hidden 2048) — the small-token regime

With 4 KB tokens (BF16) instead of 14 KB, per-token fixed costs dominate and all
implementations drop, but not equally.

Intranode EP8 kernel GB/s (FP8 disp / BF16 disp / combine):

| SMs | DeepEP | HybridEP |
|----:|---|---|
| 8 | 80 / 139 / 134 | 205 / **304** / 188 |
| 24 | 210 / 302 / 287 | 272 / **325** / **304** |

Internode EP16 RDMA GB/s:

| | SMs | FP8 disp | BF16 disp | combine |
|---|---:|---:|---:|---:|
| DeepEP | 24 | 36.2 | 58.9 | 59.1 |
| HybridEP | 8 | 42.0 | 65.3 | 33.5 |
| HybridEP | 24 | 61.4 | **72.1** | 49.3 |

Observations specific to small hidden:
- DeepEP's FP8 dispatch collapses hardest (36 GB/s cross-node): 2 KB messages cannot amortize
  the warp-copy launch pattern. HybridEP's chunked TMA pipeline degrades far more gracefully
  (+70% over DeepEP on FP8 dispatch).
- HybridEP **combine** becomes its weak spot at low SM counts in this bandwidth-bound
  microbenchmark (33.5 @ 8 SMs vs DeepEP's 59.1 @ 24 SMs); raising to 24 SMs recovers to 49.
  (In real training this did not materialize — see §4.2 — because per-dispatch token counts
  are smaller and the combine fits in the compute window.)
- HybridEP's **fused-permute mode inverts** at this size (325 → 265 GB/s): the per-chunk flag
  synchronization and the SMs given to permute blocks are no longer paid for by the saved full
  data pass. Fusion should be enabled for large-hidden models only.

---

## 4. Real training results (Qwen3-30B-A3B, NeMo Automodel)

### 4.1 Single node, EP8 (GBS 128, LBS 4, `experts: gmm`)

| dispatcher | step time | tokens/s (total / per GPU) | est. MFU* |
|---|---:|---:|---:|
| deepep | 5.366 s | 97,689 / 12,211 | ~31% |
| hybridep | **5.085 s** | **103,081 / 12,885** | ~33% |

**+5.5% throughput / −5.2% step time.** Memory +0.5 GiB for hybridep (worst-case
pre-registered buffers). Loss and grad-norm identical per step.
*MFU estimated at ~25 GFLOPs/token (fwd+bwd, 3.3B active params), same accounting both rows.

### 4.2 Two nodes, EP16 (GBS 256, LBS 2, `experts: torch_mm`)

| dispatcher | step time | tokens/s (total / per GPU) | vs deepep |
|---|---:|---:|---:|
| deepep | 5.906 s | 177,531 / 11,096 | – |
| hybridep (default, 8 SMs multinode) | **5.472 s** | **191,621 / 11,976** | **+7.9%** |
| hybridep (24 SMs) | 5.519 s | 190,002 / 11,875 | +7.0% |

Two notable outcomes:
- The cross-node gain (+7.9%) exceeds the single-node gain (+5.5%): communication is a larger
  fraction of step time at EP16, so the dispatcher delta is amplified. Expect the gap to widen
  further at larger EP.
- The multinode default of 8 SMs **beats** 24 SMs in real training, despite the §3.3
  microbenchmark suggesting combine wants more SMs. With LBS 2 each dispatch moves only 8K
  tokens, the combine hides inside the compute window, and the 16 SMs handed back to expert
  GEMMs win. Recommendation: keep HybridEP's defaults in Automodel.

---

## 5. Why is HybridEP faster? (root-cause analysis)

### 5.1 TMA bulk copies instead of SM warp copies — the SM-efficiency factor

This is the dominant effect and explains the entire shape of the bandwidth-vs-SM curves.

- **DeepEP** moves every byte through SM registers. Its inner loop is the
  `UNROLLED_WARP_COPY` macro (`csrc/kernels/utils.cuh`, used from `intranode.cu:304` etc.):
  each of 32 lanes issues `__ldg` int4 loads into registers, then `st_na_global` stores to the
  peer's NVLink buffer. Bandwidth is therefore limited by SM instruction issue —
  which is why DeepEP scales almost linearly in SMs (148→282→323 GB/s at 8/16/24) and needs
  ~24 SMs to saturate NVLink. Those 24 SMs (of 132 on H100) are pure copy engines.
- **HybridEP** uses TMA (`cuda::ptx::cp_async_bulk`, 46 call sites in
  `csrc/hybrid_ep/backend/hybrid_ep_backend.cuh`): one thread submits a bulk-copy descriptor
  and the per-SM DMA hardware moves the whole block asynchronously, never touching registers.
  The kernels are *warp-specialized persistent* pipelines — one block per SM, with dedicated
  warp groups: G2S (load into a shared-memory cyclic FIFO via TMA), S2G (TMA store into remote
  NVLink buffers), and for multinode an RDMA warp group that rings the NIC doorbells directly
  (DOCA GPUNetIO / GPU-initiated verbs). The SM only orchestrates; the copy engines and NIC do
  the moving. Result: **NVLink saturation with 8 SMs, 2.1–2.4× DeepEP's bandwidth at equal SM
  count, and 16 SMs returned to the model's GEMMs.**

The same asymmetry holds cross-node: DeepEP's 24 SMs are spent driving NVSHMEM sends plus the
NVLink forwarding hop; HybridEP does the identical two-level (RDMA-to-peer, then NVLink
fan-out) routing with 8 SMs at equal-or-better bandwidth.

### 5.2 Pipeline consolidation — fewer kernels, no CPU sync

Per MoE layer (forward), the kernel sequences are:

**DeepEP: 6 kernels + 1 CPU sync**
```
① layout kernel            (token counts per rank/expert, ~39 µs)
② notify_dispatch + CPU busy-wait  ← host spins on pinned memory to learn output size
                                     before it can allocate tensors (~35 µs, breaks CUDA graphs)
③ dispatch kernel          ← send-side gather (permute-1) + 2-level NVLink/RDMA transport
                             + FP8 scales — all fused, BUT output lands in arrival order
④ permute-2 kernel (framework)  ← full read+write to regroup by expert + pad
⑤ expert grouped GEMM      (framework)
⑥ unpermute-2 kernel (framework) ← full read+write back to transport order
⑦ combine kernel           ← transport + weighted write-back to original slots (unpermute-1)
```

**HybridEP: 4 kernels, 0 CPU syncs**
```
① metadata preprocessing   (allgather + metadata kernel — entirely on GPU)
② dispatch kernel          ← permute-1 + transport + permute-2 + padding
                             (permute-2 either merged with the copy-out, or as dedicated
                              permute blocks appended to the same kernel grid, streaming
                              per 64-token chunk on ready-flags)
③ expert grouped GEMM      (framework)
④ combine kernel           ← unpermute-2 + intra-node pre-reduce + transport
                             + weighted write-back (unpermute-1)
```

To be fair, DeepEP's transport kernels are themselves well-fused (implicit permute-1,
hierarchical forwarding, weighted combine-reduce). What it leaves exposed are the two ends of
the pipe: a per-layer CPU round-trip at the head, and the expert-regrouping passes at the tail
(two extra full-volume reads+writes per layer). HybridEP internalizes both. At 48 layers, the
CPU-sync elimination alone is worth ~3.6 ms/step of latency plus CUDA-graph compatibility;
the absorbed permute passes are the larger win and are what closes the API-vs-kernel gap
(274 → 339 GB/s at hidden 7168 with fusion on).

### 5.3 Combine-specific gains

Combine must reduce (top-k weighted sum), not just move. DeepEP folds the reduction into the
same warp-copy loops, further stressing SM issue slots. HybridEP dedicates a Reduce warp group
inside the pipeline and, cross-node, pre-reduces within the node before the RDMA hop. This is
where HybridEP's largest steady-state kernel margin shows (+12% at DSv3 dims internode).

### 5.4 Transport tiers and the "hybrid" routing

Both libraries split every all-to-all across two interconnect tiers, and the benchmarks above
exercise both tiers concurrently (the internode logs report NVL and RDMA bandwidth for the
*same* kernel — e.g. HybridEP EP16 dispatch sustains 285.8 GB/s NVLink and 85.7 GB/s RDMA
simultaneously within one 1366 µs kernel; DeepEP 263.0 / 79.5 respectively).

| | intra-node | inter-node |
|---|---|---|
| DeepEP | NVLink, peer buffers mapped via CUDA IPC; **SM warps store directly into the peer's memory** (`st_na_global`, data through registers) | NVSHMEM with **GPU-initiated IBGDA** device verbs (`csrc/kernels/ibgda_device.cuh`): warps build WQEs and ring the NIC doorbell from the kernel |
| HybridEP | NVLink, IPC (or cuMem fabric handles for MNNVL/NVL72); **TMA `cp_async_bulk` writes into the peer's buffer** (no registers) | **DOCA GPUNetIO GPU verbs**: a dedicated RDMA warp group builds WQEs and rings doorbells on-GPU; experimental NIXL/UCX alternative |
| DeepEP v2 (ref) | NVLink, IPC/fabric | **NCCL Gin (GDAKI)** — also kernel-initiated RDMA |

Note the common ground: in all three, the CPU never touches the data path — RDMA is driven
directly from the GPU. The differentiation is *which* GPU-side verbs stack drives the NIC and
whether the NVLink hop is executed by SM warps or by TMA.

The routing topology — the "hybrid" in both names — is identical in DeepEP and HybridEP:
a token crossing nodes is sent over RDMA **once per destination node**, to the peer GPU with
the same local rank (rail-optimized: each GPU only talks to its same-index peers, matching
one-NIC-per-GPU topologies and minimizing QPs), and that peer fans it out over NVLink to the
target experts' ranks. This buys (1) deduplication — IB traffic scales with destination
*nodes*, not destination *experts*; (2) chunk-level pipelining of the RDMA and NVLink legs;
(3) symmetric combine, where partial expert outputs are pre-reduced within the node before the
single RDMA hop back. Since the topology is the same, the measured differences come entirely
from the execution engines on each leg (TMA vs warp-copy; DOCA vs NVSHMEM) and the SM budget.
DeepEP v2 additionally offers a *direct* (point-to-point, no forwarding) mode alongside hybrid;
v1 and HybridEP's high-throughput path is hybrid-only.

### 5.5 Where the wins actually convert to step time / MFU

Two independent channels:
- **Exposed-time channel** (no comm/compute overlap — today's Automodel path): step-time gain
  ≈ comm fraction × kernel speedup + removed CPU syncs/permutes. Measured: +5.5% (EP8) and
  +7.9% (EP16).
- **SM-occupancy channel** (with comm/compute overlap, e.g. DeepSeek-style dual-batch): the
  cost of hidden communication is the SMs it occupies. HybridEP at 8 SMs costs 6% of compute
  throughput vs 18% for a 24-SM dispatcher — worth ~14% relative MFU, *on top of* being able
  to hide the same traffic in a smaller window. This is where HybridEP's headroom is for
  future overlap-enabled recipes; the measured +5.5/+7.9% only exercises the first channel.

---

## 6. Operational caveats (learned the hard way)

1. **Per-dispatch token cap (multinode):** HybridEP sizes its IB QP send queue as
   `3 × max_tokens_per_rank + 1`, and CX7 caps `max_qp_wr` at 32768 → **≤ ~10.9K tokens per
   rank per dispatch**. Automodel with LBS 4 × seq 4096 = 16384 tokens fails with
   `Failed to create 0th QP with status 6`; LBS 2 works. Newer `hybrid-ep` HEAD adds an
   explicit validity check; 17cfb81 dies on a bare assert.
2. **Permute fusion is size-dependent:** enable `fuse_permute_dispatch` for large hidden
   (+24% at 7168); it *loses* at hidden 2048 (−18%). Keep it off for Qwen3-30B.
3. **FP8 dispatch not yet wired for HybridEP in Automodel** (`fused_a2a.py` asserts
   `not fp8_dispatch`); comparisons above are BF16 transport in real training.
4. **Memory:** +0.5 GiB/GPU for worst-case pre-registered buffers (shared dispatch/combine
   buffer by default).
5. **Container requirements:** HybridEP multinode needs the DOCA/rdma-core runtime that the
   NeMo-FW/Megatron-Bridge images ship (`mb2604`); the Automodel image we tried lacked it
   (DOCA QP failures, NVSHMEM timeouts). First JIT compile costs ~40 s once per config.
6. **DeepEP tuning burden:** DeepEP's best numbers required sweeping ~15 NVL-chunk × 8
   RDMA-chunk configs per (SM count, dtype). HybridEP and v2 hit their numbers with defaults.

## 7. Reference: DeepEP v2 (epv2-release)

Included for context: v2 is DeepSeek's own answer to the same problems, via a different route
(NCCL Gin backend, full JIT, analytic SM sizing, unified `ElasticBuffer`). Measured: 303 GB/s
FP8 dispatch @ 8 SMs intranode (nearly saturated — big SM-efficiency gain over v1), best-in-test
84 GB/s FP8 dispatch cross-node, combine between v1 and HybridEP. Its headline features
(EP2048 scale, 0-SM PP/CP/Engram) are outside this test's scope. Note its analytic SM formula
divides by an auto-detected NIC bandwidth that returns 0 on this cluster (`EP_NIC_NAME`
mismatch) — pass `--num-sms` explicitly.

---

## Appendix: raw data locations

| Dataset | Log |
|---|---|
| DeepEP intranode SM sweep (DSv3 dims) | `logs/deepep_v1_sweep.log`, `logs/deepep_v1_intranode.log` |
| HybridEP intranode sweeps | `logs/hybridep_sm{4,8,16,24}.log` |
| DeepEP v2 intranode | `logs/deepep_v2_ep8*.log`, `logs/deepep_v2_sm*.log` |
| DeepEP v1 full-path (notify/CPU-sync cost) | `logs/v1_fullpath_sm24.log` |
| Internode DSv3 dims (v1 + HybridEP) | `logs/internode_14246612_node0.out` |
| Internode DSv3 dims (v2) | `logs/v2_internode_14247570_node0.out` |
| Qwen dims intranode | `logs/qwen_v1_sweep.log`, `logs/qwen_hybrid_sm{8,24}.log`, `logs/qwen_v2_sm{8,24}.log` |
| Qwen dims internode | `logs/qwen_internode_14247894_*.out`, `logs/qwen_hyb24_14248118_node0.out` |
| Real training single node | `logs/am_step_{deepep,hybridep}.log` |
| Real training 2 nodes | `logs/am_ep16_14250238_node0.out` |
