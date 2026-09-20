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

**Why.** The implementation is written for inference correctness, not training throughput. Its
compressed-sparse layers gather `S x k` keys per query, and its attention materialises the full
score matrix. At 80 GB per GPU there is no sequence length short enough to rescue it.

To get a baseline that runs at all, **truncate the model to its first 4 layers**: sliding window,
sliding window, compressed sparse, heavily compressed. One of each attention kind, 3.01B parameters.
Everything else stays fixed. That reduced model is the only setting where all four stages of the
next slide can be compared like for like.
