# Step5 full92 mock benchmark on 128 H100 GPUs

> **Re-measurement pending.** The result below was measured at source
> `77e4eddd`, before this branch adopted upstream's benchmark FLOPs and timing
> accounting, HybridEP dispatch, and unseeded mock dataset. The configuration
> is otherwise unchanged; rerun it on the current source before quoting MFU.

This is the native, unprofiled benchmark configuration used by DFW job
**20089678**, preserved from source
`77e4eddd3a76d91b80ab614ee0427d1d6d90241a`. It constructs the full 92-layer
model with `NeMoAutoModelForCausalLM.from_config`; it does not load a checkpoint.
The checked-in model JSON is an exact copy of the measured run's configuration.

The model has hidden size 4096, 88 MoE layers with 352 experts/top-8, four dense
layers, Q64/KV4/D192, and the original SWA/CSA layer layout. Inputs are seeded
mock single documents of 8192 tokens. MoE uses fake balanced, static routing;
the attention indexer's learned selection remains active. SWA varlen and CSA
forward and backward full blocks are enabled; compact HybridEP routing is
disabled.

The topology is 16 nodes × 8 H100, EP32/PP4/VPP1/DP32/CP1/TP1. The public
`1f1b` schedule uses one microbatch per example, M24/LB24/GA1 and global batch
768, with contiguous layer partitions [0,24), [24,46), [46,70), [70,92).
All 17 steps ran; steps 0–6 are warmup and **all ten steps 7–16** form the result.

| Measured quantity | Value |
| --- | ---: |
| Mean max-rank step time | 41.85826148986816 s |
| Native reference MFU | 20.57865201031263% |
| MFU excluding absent fake-router GEMMs | **20.488266201281494%** |
| Conservative MFU also excluding all indexer FLOPs | 20.37338096311403% |
| Raw / supervised tokens per step | 6,291,456 / 6,290,688 |
| Corrected useful FLOPs per step | 1,085,657,048,168,792,064 |

The denominator is 128 × 989 TFLOP/s × mean step time. Rematerialization and
physical attention tile padding are excluded from useful FLOPs. The native
reference counts router GEMMs that the fake gate does not execute: subtract
4,789,472,650,592,256 FLOPs/step for the corrected result above. It is a
mock/fake-balanced throughput measurement, not real-data training performance
or checkpoint/gradient parity. Native audits verified 13,056 logical inputs
(52,224 PP copies) against the frozen expected hashes, finite saved
initialization samples and FP32 norms, and all 128 ranks reported zero
allocator retries/OOMs. The peak allocated/reserved memory on any rank was
65.50/69.08 GiB. Saved initialization samples do not establish complete-state
equality.

Relative to the previous M16/GB512 result from job 20038646 (33.9784 s/step,
16.8264% corrected MFU at source `325ae1eb`), this result adds three source
changes: native FA4 head layout for bounded sparse metadata (`f3293809`), CSA
backward full-block classification (`29ce0bf5`, enabled by
`csa_backward_full_blocks`), and canonical-document CSA masks (`77e4eddd`).
It also raises the microbatch count from 16 to 24 and rebalances the pipeline
partitions. The global batch therefore changes from 512 to 768; compare
throughput per token rather than seconds per step.

## Running the native recipe

Run from the repository root in the same prepared rc5 H100 environment.
The preserved image is `auto2610rc5.sqsh`, with the existing FA4 SM90 overlay
at `/workspace/deps/rc5-fa4-sm90` and compatibility scripts at
`/workspace/scripts`. The tokenizer remains the concrete local path
`/workspace/checkpoints/first4-native-view`: only its tokenizer is used.
The model architecture comes from this directory's full92 JSON.

Within an already allocated 16-node/128-H100 Slurm job, run the following on
each node through `srun` (one task per node). Set `MASTER_ADDR` to the first
allocated node. This is the same native module executed by the original
benchmark-phase wrapper; no profiler or training wrapper is needed.

```bash
export PYTHONPATH="/workspace/deps/rc5-fa4-sm90:/workspace/scripts:$PWD"
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/cache/huggingface-h100-pp
export HYBRID_EP_CACHE_DIR=/workspace/cache/hybrid-ep-h100-pp-rc5
export NEMO_HYBRIDEP_JIT_CACHE=/workspace/cache/hybrid-ep-shared-h100-pp-rc5
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN CUDA_MODULE_DATA_LOADING
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NVLS_ENABLE NCCL_DEBUG_SUBSYS
unset NCCL_LAUNCH_ORDER_IMPLICIT CUDA_DEVICE_MAX_CONNECTIONS TORCH_NCCL_BLOCKING_WAIT
# Avoid repeated hardware scans from shell startup files in compiler subprocesses.
unset BASH_ENV ENV SSH_CLIENT SSH2_CLIENT
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cache_dir="/workspace/cache/step5-full-mock-persistent/rc5-1418c374-sm90-325ae/node-$SLURM_NODEID"
export TORCHINDUCTOR_CACHE_DIR="$cache_dir/inductor"
export TRITON_CACHE_DIR="$cache_dir/triton" CUDA_CACHE_PATH="$cache_dir/cuda"

torchrun --nnodes=16 --nproc_per_node=8 --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" --master_port=29630 \
  -m nemo_automodel.recipes.llm.benchmark \
  --config examples/llm_benchmark/step/step5_full92_mock_h100/benchmark.yaml
```

Use the existing W&B authentication; no credentials are included here.
The stable per-node caches were originally seeded from job 20037178. Their
reuse does not guarantee every kernel is warm. Record runtime dependencies and
cache state for comparisons. The original scheduler used the
`coreai_dlalgo_nemorl` account, H100 constraint, 16 nodes and a 40-minute limit.

The YAML differs from the original input configuration in its model JSON
location and output directories, and it omits the measured run's
batch-hash, initialization-audit and allocator-statistics outputs, which
recorded provenance without affecting timing. Outputs default to
`outputs/step5-full92-mock-h100`; select a fresh directory before another run
by overriding `wandb.dir`, `checkpoint.checkpoint_dir` and
`benchmark.json_output_path`. Do not reuse the immutable original run
directory. The tokenizer, seed, inputs, optimizer, model, backends,
partitions, batch sizes and measured window are unchanged.

## Provenance

Remote artifact paths below are relative to
`/lustre/fsw/portfolios/coreai/users/huiyingl/step5pre/artifacts/mfu-optimization-20261011`
on DFW. Large run artifacts stay outside Git.

| Immutable artifact | SHA256 |
| --- | --- |
| `full-launchers/77e4eddd-pp4-vpp1-m24-unprofiled-v3/manifest.json` | `74386bfdbcc2e7214e3e27da058957278710cee22f2ac2244f8beac433ff2051` |
| Original template `benchmark.yaml` in that bundle | `439f49855712fd09d83d774c0050dde715c4c456f453c7c53c2a2bf09c83a7d9` |
| Original run `benchmark/effective-config.yaml` | `c0fd761bbff2f2c3a00a2aadefe280aaa8202826bcbe66709159443536696d4a` |
| Original run `benchmark/benchmark.json` | `9dcb16c0709d2baaf4878ffd9b9d26d044c9bdf935ee960d99c7d43d0d91699c` |
| This `model-config/config.json` | `817fa0ba01db709c5c59df4832a9751da56a04a0476833708dbd7e449b9a9490` |
| Bundle `expected-mock-row-hashes.json` | `3ef4a806e7bd7022ed6884f0377097b0a698e49fd712d4bd62e4bc9b879d026c` |
| `flops/full20089678-mfu-audit/report.json` | `07ae98c0565f542b350b6f47e8a687ab9c4d6dce8fadfd176da716e35ed467db` |
| `flops/full20089678-mfu-audit/owned-file-seal.json` | `24c65b96eb81429eb3cbe7a372f0befe0382178fec9ac09e25808537c1c1f3e9` |

The original run root is
`step5-full92-mock-cp1-ep32-pp4-vpp1-m24-unprofiled-v3/job-20089678`.
Only its unprofiled `benchmark` phase contributes the result.
