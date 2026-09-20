# What DeepSeek-V4 adds, and why it matters for throughput

V3-style attention is dense multi-head latent attention. V4 replaces it with machinery that is
cheap in theory and awkward in practice.

- **Shared-key MQA.** One key/value head of dimension 512, where the same entry serves as both
  key and value. Partial rotary embedding on 64 dimensions, inverted again on the output.
- **Hybrid sparse attention.** Every layer has a 128-token sliding window. Layers alternate
  between compressed sparse attention, which adds a compressor and a lightning indexer that
  selects the top 512 compressed entries, and heavily compressed attention at ratio 128.
- **Hyper-connections.** The residual stream is carried in 4 parallel copies. Each layer mixes
  them twice through a doubly-stochastic matrix produced by 20 Sinkhorn iterations in fp32.
- **Hash-routed first layers** and sqrt-softplus routing with an auxiliary-loss-free bias.

The throughput consequence, which drives most of this deck: **the model's useful work is
concentrated in a few large GEMMs, while its distinctive parts are many small fp32 operations.**
Sparse attention only pays off if the kernel actually skips the masked positions.
