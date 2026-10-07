# Tiny parallel precision regressions

These suites use random weights and synthetic tokens. They need no model or
dataset downloads. Each distributed subprocess is limited to three minutes,
with forced cleanup ten seconds later.

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
The four-layer model has width 64 and vocabulary 64. PP4 exercises interior
pipeline stages. BF16 projections and FP32 residuals coexist in the same run;
the FP32 cases do not mean full-FP32 compute.

The tiny Llama recipe adds tensor/sequence parallelism and context parallelism.
The Nemotron MoE recipe uses width 512, latent width 256, eight experts and top-2
routing to exercise HybridEP with both Torch and Transformer Engine experts.
Both compare three optimizer steps and final validation against a run with PP
disabled and the same remaining topology. Every loss and gradient norm must be
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
rather than silently omitting their tests. Failed subprocess logs are included
in the pytest failure and retained under its temporary directory.
