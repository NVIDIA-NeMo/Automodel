# HybridEP ragged-slot experiment

This branch isolates DeepEP #744 integration on the September 6 benchmark source,
`8876e135b91ca141bf7f5dec1d517cb4e07d2ce3`. It is an experimental performance
branch, not a dependency/build update or a replacement for current main.

Run it in the same pinned benchmark container as the August/September controls:

```
gitlab-master.nvidia.com#dl/joc/nemo-ci/akoumpa_perf_moe2606-final-cdd4f1d/automodel:sha256:528bac3ec8ebc18b961aefb34bfbaa6df193ecc025436247b4ed004e5a8034fc
```

The runtime must provide the HybridEP ragged API, including upstream #746's
handle-layout fix (`10d4dd7377d5bce900fbb4b80cce863764892b95`). The historical
source's dependency declarations are intentionally unchanged; installing its
historical dependency set is not a supported way to run this experiment.

The dispatcher passes a common slot extent to upstream, retaining real input
activation and probability shapes. Without a proven capacity it still negotiates
the maximum on every dispatch. The fixed mock-data benchmark automatically
provides `sequence_length * microbatch_size` for TP=CP=1, with the scope covering
forward, backward/recomputation and overlapping pipeline microbatches. Other
input/layout paths retain negotiation. There is no new user config option.

A bound overflow raises; it never initiates a rank-conditional collective.
Upstream handles alignment, empty inputs, metadata padding and combine narrowing.
No receive-capacity, static-routing, expert-GEMM, or rotary optimization is enabled.

Validation uses the original MiniMax-M2.5 LoRA recipe: 8 nodes / 64 H100,
PP2/EP32, sequence 4096, local batch 4, global batch 512, 30 steps and 10 warmup.
All three arms run sequentially on one allocation with the same runtime and
model metadata. Nsight captures steps 25-26 on ranks 0 and 32; throughput uses
only steps 10-24. This is not a recreation of the historical August runtime.
