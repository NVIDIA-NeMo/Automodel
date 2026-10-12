# Step5 full92 mock benchmark on 128 H100 GPUs

This is the native, unprofiled benchmark configuration used by DFW job
**20038646**, preserved from source
`325ae1ebf2dc15893d006e35fa6b69bb08f4ea0e`. It constructs the full 92-layer
model with `NeMoAutoModelForCausalLM.from_config`; it does not load a checkpoint.
The checked-in model JSON is an exact copy of the measured run's configuration.

The model has hidden size 4096, 88 MoE layers with 352 experts/top-8, four dense
layers, Q64/KV4/D192, and the original SWA/CSA layer layout. Inputs are seeded
mock single documents of 8192 tokens. MoE uses fake balanced, static routing;
the attention indexer's learned selection remains active. SWA varlen and CSA
forward full blocks are enabled; compact HybridEP routing is disabled.

The topology is 16 nodes × 8 H100, EP32/PP4/VPP1/DP32/CP1/TP1. The public
`1f1b` schedule uses one microbatch per example, M16/LB16/GA1 and global batch
512, with contiguous layer partitions [0,25), [25,47), [47,69), [69,92).
All 17 steps ran; steps 0–6 are warmup and **all ten steps 7–16** form the result.

| Measured quantity | Value |
| --- | ---: |
| Mean max-rank step time | 33.978401565551756 s |
| Native reference MFU | 16.9006693913066% |
| MFU excluding absent fake-router GEMMs | **16.826438062872867%** |
| Conservative MFU also excluding all indexer FLOPs | 16.732086040824154% |
| Raw / supervised tokens per step | 4,194,304 / 4,193,792 |
| Corrected useful FLOPs per step | 723,771,365,445,861,376 |

The denominator is 128 × 989 TFLOP/s × mean step time. Rematerialization and
physical attention tile padding are excluded from useful FLOPs. The native
reference counts router GEMMs that the fake gate does not execute: subtract
3,192,981,767,061,504 FLOPs/step for the corrected result above. This result is
below 20%. It is a mock/fake-balanced throughput measurement, not real-data
training performance or checkpoint/gradient parity. Native audits verified
8,704 logical inputs (34,816 PP copies), finite saved initialization samples
and FP32 norms, and all 128 ranks reported zero allocator retries/OOMs.
Saved initialization samples do not establish complete-state equality.

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
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cache_dir="/workspace/cache/step5-full-mock-persistent/rc5-1418c374-sm90-325ae/node-$SLURM_NODEID"
export TORCHINDUCTOR_CACHE_DIR="$cache_dir/inductor"
export TRITON_CACHE_DIR="$cache_dir/triton" CUDA_CACHE_PATH="$cache_dir/cuda"

torchrun --nnodes=16 --nproc_per_node=8 --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" --master_port=29624 \
  -m nemo_automodel.recipes.llm.benchmark \
  --config examples/llm_benchmark/step/step5_full92_mock_h100/benchmark.yaml
```

Use the existing W&B authentication; no credentials are included here.
The original stable per-node caches were seeded from job 20037178. Their reuse
does not guarantee every kernel is warm. Record runtime dependencies and cache
state for comparisons. The original scheduler used the `coreai_dlalgo_nemorl`
account, H100 constraint, 16 nodes and a 40-minute limit.

The YAML differs from the original effective configuration only in its model
JSON location and output directories. Outputs default to
`outputs/step5-full92-mock-h100`; select a fresh directory before another run
by overriding `wandb.dir`, `checkpoint.checkpoint_dir`, and all four benchmark
output keys: `allocator_stats_path`, `batch_metadata_path`,
`loaded_model_audit_path` and `json_output_path`. Do not reuse the immutable
original run directory. The tokenizer, seed, inputs, optimizer, model,
backends, partitions, batch sizes and measured window are unchanged.

## Provenance

Remote artifact paths below are relative to
`/lustre/fsw/portfolios/coreai/users/huiyingl/step5pre/artifacts/mfu-optimization-20261011`
on DFW; audit copies also exist on PDX. Large run artifacts stay outside Git.

| Immutable artifact | SHA256 |
| --- | --- |
| `full-launchers/325ae1eb-pp4-vpp1-m16-attention-opt-v1/manifest.json` | `5241ca9617537c835d3b65c5b63ff2d6cbc6d911bfed32955aa0a145402eeea1` |
| Original template `benchmark.yaml` in that bundle | `af62af7cd75695e4932b1d22e0fd1b91f501f840de9a06cc026b8b629c361a2b` |
| Original run `benchmark/effective-config.yaml` | `669685b79d287e01547ede6fea52721f297dfdc68289f52a8feef9578c715114` |
| Original run `benchmark/benchmark.json` | `45bed95c603a5e950acf230e1fb95232ff0b684c710f6939180cb86e5721f23e` |
| This `model-config/config.json` | `817fa0ba01db709c5c59df4832a9751da56a04a0476833708dbd7e449b9a9490` |
| Bundle `expected-mock-row-hashes.json` | `cfe000eb1d42cef2e746b985adfd93bc7d9486ec2032589ba313e5773d495773` |
| `flops/full20038646-mfu-audit/report.json` | `ef68c60ae37167a68d5e530b8f4bb739f6293ee4d4844e672ea1605ebd64083f` |
| `flops/full20038646-mfu-audit/final-seal.json` | `6c9667fa84f66a1b950d404bd53ba9a6291adcf5ea7de713efea2e0632c6dcbc` |

The original run root is
`step5-full92-mock-cp1-ep32-pp4-vpp1-m16-attention-opt/job-20038646`.
Only its unprofiled `benchmark` phase contributes the result; the later
`profile` phase is diagnostic. Comparisons with job 20028380 must state the
global-batch change from 256 to 512.
