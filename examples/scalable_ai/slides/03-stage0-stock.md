# Stage 0: stock Transformers, out of the box

```bash
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py \
    --config examples/scalable_ai/configs/moonlight_v4_16b_hf.yaml
```

`transformers` has a native `deepseek_v4` implementation. Pointed at the full 27-layer model it
fails before finishing a single forward pass.

| sequence length | micro-batch | result |
| --- | ---: | --- |
| 2048 | 1 | out of memory in the first forward |
| 1024 | 1 | out of memory in the first forward |
| 512 | 1 | out of memory in the first forward |

**Why.** The implementation is written for inference correctness, not training throughput.

```python
if self.compressor is not None:              # Compressed KV (CSA or HCA)
    compressed_kv = self.compressor(...)
    kv = torch.cat([kv, compressed_kv], dim=2)   # extend the key axis
...
attn_output, attn_weights = attention_interface(self, q, kv, kv, attention_mask, ...)
```

The compressed entries are **concatenated onto the key axis** and one dense attention runs over the
whole extended length, with sparsity applied only as an additive mask. A compressed-sparse layer
therefore costs *more* than a dense one, not less. There is no FlashAttention for 512-dimensional
heads and no SDPA path, so the score matrix is materialised and its softmax runs in fp32.

| per layer, micro-batch 1, sequence 2048 | memory |
| --- | ---: |
| attention probabilities, fp32, kept for backward | 336 MB |
| scores, bf16 | 168 MB |
| the same layer computing only its 128-token window | 17 MB |

Across 27 layers the probabilities alone are 9.1 GB, before scores, activations or parameters. And
the baseline cannot use expert parallelism, because that sharding needs Automodel's own
mixture-of-experts modules, so it runs `ep_size: 1`.

Halving the sequence length halves one factor of a term that is quadratic in it and additive against
everything else, which is why 512 tokens does not rescue it either.

To get a baseline that runs at all, **truncate the model to its first 4 layers**: sliding window,
sliding window, compressed sparse, heavily compressed. One of each attention kind, 3.01B parameters.
Everything else stays fixed. That reduced model is the only setting where all four stages of the
next slide can be compared like for like.
