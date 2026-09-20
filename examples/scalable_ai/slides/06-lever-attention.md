# Lever 1: stop emulating sparse attention densely

The eager path computes the whole score matrix and then adds a mask:

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
attn_weights = attn_weights + attention_mask[..., : attn_weights.shape[-1]]
```

Every sliding-window layer therefore does the work of full attention. With 16 heads and head
dimension 512, one layer at sequence 2048 and micro-batch 2 executes 275 GFLOP of attention, while
the model is credited for the 128-token window it actually needs.

```
dense attention executed, 12 layers, fwd+bwd     9.90 TFLOP per GPU per micro-step
credited sparse attention                        1.39 TFLOP
```

Roughly 8.5 of every 44 TFLOP the GPU executes are scores that get masked away. Head dimension 512
makes this four times worse than a conventional 128, because the score matmul scales with it.

The TileLang path instead builds top-k key indices and gathers only what each query needs, and it
brings fused Sinkhorn and indexer kernels with it.

| 12 layers, micro-batch 3 | step time | MFU |
| --- | ---: | ---: |
| eager attention | 0.660 s | 7.95% |
| TileLang sparse attention | 0.489 s | 10.77% |

**1.35x, from one backend setting.** This is the single largest lever in the deck.
