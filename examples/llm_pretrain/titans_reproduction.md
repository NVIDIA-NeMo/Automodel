# Titans paper reproduction target

This directory targets the NeurIPS 2025 proceedings version of
**Titans: Learning to Memorize at Test Time** (Behrouz, Zhong, and Mirrokni),
while retaining the original arXiv v1 experiments. The proceedings paper is
the authoritative target for the added 1.3B/100B-token and RULER results.

## Pinned language-model setup

- Data: FineWeb-Edu.
- Tokenizer: Llama 2, vocabulary size 32K.
- Training sequence length: 4096 tokens (2048 for sliding-window attention).
- Neural-memory chunk size: 16.
- Deep-memory update: public `titans-pytorch` chunk-aggregated semantics, with
  the full 4096-token training sequence as one gradient-anchor batch.
- Persistent memory: 128 tokens.
- Long-term-memory output: 256 memory tokens for hybrid architectures.
- Memory MLP: two layers by default, expansion factor 4, GELU, residual
  connection, and layer normalization.
- Token mixer: Llama macro architecture with SwiGLU, RoPE, RMSNorm, causal
  depthwise convolution of width 4 after Q/K/V projections, L2-normalized
  queries and keys, and gated normalization before the output projection.
- Optimizer: AdamW, weight decay 0.1, cosine learning-rate schedule.
- Effective global batch: approximately 0.5M tokens.

The proceedings architecture table gives these scale points:

- 170M: 12 blocks, hidden size 768, 16 heads, peak LR 3e-3, 15B tokens.
- 340M: 24 blocks, hidden size 1024, 16 heads, peak LR 1.5e-3, 15B tokens.
- 760M: 24 blocks, hidden size 1536, 16 heads, peak LR 1.25e-3, 30B tokens.
- 1.3B: 18 blocks, hidden size 2048, 8 heads, peak LR 7e-4, 100B tokens.

## Reproduction order

1. Validate the neural-memory module and its Gated DeltaNet reduction.
2. Train the 170M memory-only LMM as the scale and convergence gate.
3. Reproduce the memory-depth and component ablations.
4. Train the 340M and 760M LMM and hybrid variants (MAC, MAG, MAL).
5. Reproduce language-model perplexity and common-sense evaluations.
6. Reproduce S-NIAH and the proceedings RULER evaluation.
7. Attempt the 1.3B/100B-token proceedings extension.

## Unresolved details

No official implementation was released. The paper does not fully specify:

- the 400M model dimensions reported in the language-model table;
- whether the architecture-table peak learning rates supersede the prose
  statement that training uses a learning rate of 4e-4;
- all tensor shapes and parameterizations of the channel-wise forget gate;
- exact chunk-boundary retrieval alignment and streaming-state semantics;
- all implementation details needed to reconstruct MAC, MAG, and MAL.

Do not silently guess these details. Record each operational choice in the
resolved run configuration and validate it against an independent reference
or ablation before labeling a run as a paper reproduction.

## Acceptance gates

A full-scale run may start only after:

- tiny-model forward, backward, and checkpoint round-trip tests pass;
- deep-memory chunk size 1 matches an independent per-token autograd reference;
- momentum disabled matches the Gated DeltaNet recurrence;
- state carried across two calls matches one concatenated causal call;
- the configured parameter count matches the intended scale;
- a one-GPU smoke and one-node distributed smoke produce finite loss.

## Preflight status

Validated on 2026-09-08 with RTX 6000 Ada GPUs:

- The deterministic 760,784-parameter smoke model completed two training and
  validation steps on one GPU, wrote DCP checkpoints, and resumed from
  `LATEST` for a third step.
- The same smoke completed one FSDP2 step on two GPUs, including validation
  and checkpointing. Titans uses a dtype-aware FSDP strategy so its fp32 decay
  parameters are isolated from bf16 parameters.
- The full 173,597,376-parameter model completed one forward/backward optimizer
  step at sequence length 128 on one GPU. It allocated 6.68 GiB and processed
  180.7 tokens/s. This is a short-sequence viability probe, not an estimate
  that establishes 4096-token fit.
- The NanoGPT writer/reader round trip now records BOS locations consistently
  as token offsets. Validation datasets use `repeat: false`; otherwise the
  iterable validation loop never terminates.
- The first cw-dfw 4K attempt exposed an invalid one-step warmup and the second
  exposed uncheckpointed neural-memory activations at 79 GiB/H100. Both are
  fixed. Checkpointing reduced the measured allocation to about 22.4 GiB/H100,
  but the per-chunk Python reference backend then exceeded the 30-minute NCCL
  watchdog. The production recipe now selects the vectorized public backend
  and its Triton associative scan; that path still requires this gate.

The next execution gate is a full-shape, 4096-token, one-node FSDP2 smoke on
the target cluster. Do not start the 15B-token run until that job establishes
memory headroom and step throughput.

### cw-dfw H100 gate

The preferred workflow uses the configured `cw-dfw` profile from `slurm-cli`.
From a clean, pushed AutoModel branch:

```bash
tools/submit_titans_cwdfw.sh
# Or submit, monitor to a terminal state, and print the final log:
tools/submit_titans_cwdfw.sh --wait
```

The wrapper verifies that local `HEAD` equals the pushed branch, clones or
fast-forwards the Lustre checkout through `slurm-cli shell`, submits through
`slurm-cli job submit`, and prints reproducible status/log commands. It refuses
dirty or unpushed source trees.

After the 4K gate, launch the production-shaped pilot with:

```bash
tools/submit_titans_cwdfw.sh --pilot
```

This submits an eight-H100 pilot that first streams
`HuggingFaceFW/fineweb-edu` (`sample-10BT`) through AutoModel's NanoGPT
processor with the public `NousResearch/Llama-2-7b-hf` tokenizer when the
cached 64M-token shard is absent. It then trains on 5,242,880 tokens, validates
distributed control flow, and writes a job-specific checkpoint. Dataset
preparation uses the GPU allocation because `cw-dfw` does not currently grant
this account CPU-partition capacity. The pilot is not a paper result.

Defaults:

- account: `coreai_dlalgo_compeval`;
- partition: `batch`;
- allocation: one node with eight GPUs for one hour;
- image: `nvcr.io#nvidia/nemo-automodel:26.04`;
- work root:
  `/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel`.

Override `AUTOMODEL_CHECKOUT`, `TITANS_WORK_ROOT`, or `TITANS_IMAGE` with
exported environment variables when the checkout, storage root, or approved
container differs. Success requires a finite train and validation loss plus
the logged peak memory and tokens/s from all eight workers.
