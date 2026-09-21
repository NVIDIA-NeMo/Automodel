# Starting point: it only just fits

```bash
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_hf.yaml \
  --step_scheduler.global_batch_size 8
```

`transformers` ships a DeepSeek-V4 implementation. The full 27-layer, 16.5B-parameter model does
train on 8 H100s with FSDP2 alone. The only full-model point that completed in our sweep was the
smallest shape:

| micro-batch / GPU | accumulation | global batch | outcome | MFU / peak memory |
| ---: | ---: | ---: | --- | --- |
| 1 | 1 | 8 | 0.940 s, 17.4k tokens/s | **3.67% / 45.6 GB** |
| 1 | 32 | 256 | out of memory in the first backward | - |
| 2 | 1 | 16 | one iteration at 71.9 GB, then out of memory | - |

The first row overturns an earlier version of this slide, which said the stock implementation could
not train the model. That claim generalized from the second row.

**What actually failed.** With 32 accumulation steps, FSDP2 defers the gradient reduction and tries
to retain an unsharded accumulated-gradient buffer on every rank:

```python
post_backward()
to_accumulated_grad_if_needed()
unsharded_accumulated_grad = unsharded_grad.to(reduce_dtype)
# torch.OutOfMemoryError: tried to allocate 1.38 GiB
```

At sequence 2048 the allocation fails in the first backward. At 1024 and 512, the retained gradient
buffer survives into a later micro-batch and collides with the next FSDP all-gather. Shortening the
sequence cannot remove a parameter-sized buffer. The earlier description, "first forward at every
length," was wrong.

**Why the surviving corner is still narrow.** The stock compressed-attention path appends selected
compressed keys and applies sparsity as a dense mask. It also has no FlashAttention path for these
512-dimensional heads, so the score matrix is materialised and its softmax runs in fp32.

| per layer, one sequence of 2048 | memory |
| --- | ---: |
| attention probabilities, fp32, kept for the backward pass | 336 MB |
| the same layer computing only its 128-token window | 17 MB |

Across 27 layers that is 9.1 GB of probabilities alone, before weights, activations or gradients.
That helps explain why micro-batch 2 cannot sustain a second iteration, but it is not why the
accumulation-32 run failed.

**This is the lesson in miniature.** An out-of-memory result describes one configuration, not an
implementation. Read the failing allocation, and finish the optimizer step before declaring what a
program can or cannot do.
