# Lever 5: precision where it is free

Two fp32 operations showed up in the trace that nothing requires to be fp32.

**Gradient reduction.** FSDP2 defaults to `reduce_dtype: float32` while parameters are bf16, which
made the gradient ReduceScatter the single largest kernel at 90 ms. Switching it to bf16 halves the
traffic. The modules that genuinely need fp32, the mixers, attention sinks and the vocabulary
projection, keep fp32 reduction automatically because their FSDP units carry an fp32 parameter dtype.

```yaml
distributed:
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16
    reduce_dtype: bfloat16
    output_dtype: bfloat16
```

**The vocabulary projection.** The released V4 keeps `lm_head` in fp32. Against a 163,840-row
vocabulary, that fp32 matmul profiled at 50 ms per step, 9.6% of GPU time. A new `lm_head_bf16` flag
runs it in bf16 like the rest of the model. This changes the numerics of the output head, so it is
opt-in and clearly labelled rather than a default.

| 12 layers, micro-batch 4 | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| fused hyper-connections | 0.561 s | 12.47% | 48.2 GB |
| plus bf16 gradient reduction | 0.554 s | 12.64% | 48.1 GB |
| plus bf16 vocabulary projection | 0.514 s | 13.60% | 44.3 GB |

The gradient change is worth little here because at one accumulation step the ReduceScatter had
already shrunk to 17 ms. The same change on the micro-batch-1 shape would be worth far more.
