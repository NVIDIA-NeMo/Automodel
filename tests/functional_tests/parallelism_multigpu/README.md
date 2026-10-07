# Tiny parallel precision regressions

These suites use random weights and synthetic tokens. They need no model or
dataset downloads. One persistent worker group shares Python, CUDA and NCCL
startup across both folders in a GPU job. MoE jobs also compile HybridEP
dispatch/combine kernels in this shared startup, using synthetic tensors and
no model. Each case still runs production model initialization, training and
validation. Startup is bounded at two minutes;
each individual case has a 30-second deadline and records its own timing.
Timeouts terminate the complete worker process group.

| Runner | Suites | Topologies |
| --- | --- | --- |
| `nemo-ci-aws-gpu-x4` | `parallelism_multigpu,moe_multigpu` | PP2/FSDP2, PP4, PP2/TP2, PP2/EP2 |
| `nemo-ci-aws-gpu-x8` | `parallelism_multigpu,moe_multigpu` | PP2/FSDP4, PP4/FSDP2, PP2/TP2/CP2, PP4/EP2, PP2/EP2/FSDP4 |
| `nemo-ci-gcp-gpu-x4` | `parallelism_multigpu` | PP2/FSDP2, PP4, PP2/TP2 |

EP shares the data-parallel ranks; its size does not multiply the GPU count.
The larger member pools retain the workflow's contributor trust checks. The
original `parallelism/test_parallelism.py::TestParallelismParity::test_pp_dtype_parity`
continues to run on the default two-GPU suite.

The dense Nemotron worker checks every parameter's gradient and SGD update
against an unpartitioned reference, plus loss parity. It crosses BF16 and FP32
residuals, activation checkpointing, MTP depth 0/1, and sequence lengths 16/24.
Each combination is a separate pytest case. The four-layer model has width 64
and vocabulary 64. PP4 exercises interior
pipeline stages. BF16 projections and FP32 residuals coexist in the same run;
the FP32 cases do not mean full-FP32 compute.
Stage boundaries explicitly check FP32/BF16 residual and BF16 embedding
dtypes. Final MTP states stay on the last stage and follow the unpartitioned
model's dtype: CUDA autocast may promote their final Torch RMSNorm to FP32
even with BF16 residuals.

The tiny Llama recipe adds tensor parallelism and context parallelism, using
TE attention when CP is enabled. Sequence parallelism is disabled: the PP2/TP2
control with SP enabled hangs in NCCL RECV with full-sequence receive metadata
on the tested PyTorch 2.13 image. This is a separate coverage gap; enabling SP
in `llama.yaml` retains a small reproduction.
The recipes use 32-token sequences and vocabulary 128. Llama has two layers,
width 64 and global batch two, the smallest batch for its PP2 schedule;
the Nemotron MoE recipe retains width 512 and latent width 256 to exercise the
latent projection, with four experts, expert intermediate width 128 and top-2
routing to exercise HybridEP with both Torch and Transformer Engine experts.
Both compare three optimizer steps and final validation against a run with PP
disabled and the same TP/CP/EP degrees. The reference uses the same workers,
increasing DP size and decreasing the local batch by the PP size to preserve
the global batch and accumulation count. After normal recipe setup, the test
entrypoint assigns random weights seeded by parameter name and checks matching
fingerprints across both runs. A global RNG seed alone cannot ensure matching
initial weights when stages initialize different parameter subsets. Production
setup still runs so stage-specific initialization hangs remain detectable.
Every loss and gradient norm must be
finite, every expected step must be present, and gradients must be nonzero.
Loss tolerance is 0.05 for Llama and 0.10 for MoE; gradient-norm relative
tolerance is 0.05. These account for BF16 reduction and GEMM ordering; the
separate dense test provides per-parameter checks.

Run on one allocated node from the repository root:

```bash
bash tests/run_test.sh --TEST_NAME=parallelism_multigpu,moe_multigpu --GPU_COUNT=4
bash tests/run_test.sh --TEST_NAME=parallelism_multigpu,moe_multigpu --GPU_COUNT=8
```

The GPU count must match the allocation. The suites fail on a two-GPU runner
rather than silently omitting their tests. Logs and per-case `timing.json`
files are retained under pytest's temporary directory. The shared worker
startup is reported separately from case execution.
