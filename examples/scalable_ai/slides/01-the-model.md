# The model

**Moonlight-V4-16B-A3B.** A DeepSeek-V4-architecture model at Moonlight scale: 16.5B parameters
total, 3.0B active per token, 27 layers, 64 experts with top-6 routing plus one shared expert.

Three things about V4 matter for everything that follows.

**1. Its attention is meant to be sparse.** Every layer has a 128-token sliding window. Layers then
alternate between *compressed sparse attention*, which adds a compressor and an indexer that picks
the most relevant compressed entries, and *heavily compressed attention*. In theory a layer looks at
a small fraction of the sequence.

**2. Its experts are the bulk of the work.** 64 experts per layer, each token routed to 6 of them.
More than half of all arithmetic in this model is expert matrix multiplications.

**3. It carries four parallel residual streams.** "Hyper-connections": each layer mixes the four
copies twice, through a matrix produced by 20 normalisation iterations in fp32.

Points 1 and 2 are where the performance is. Point 3 is where it quietly leaks away.
