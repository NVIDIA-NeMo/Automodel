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
