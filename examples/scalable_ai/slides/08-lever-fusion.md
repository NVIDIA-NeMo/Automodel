# Lever 3: a thousand small kernels

> Row 4: 0.388 s to 0.355 s. **1.09x**, and 3 GB.

Recall the profile: 15,027 elementwise launches, 38% of GPU time, for work that earns almost no
credit. Where do they come from?

The hyper-connections. Twice per layer, the model takes its four residual streams and mixes them.
The code looks harmless:

```python
flat = hidden_streams.flatten(2).float()      # a full fp32 copy, written to memory
mix  = linear(rms_norm(flat), fn)             # then read straight back
```

That fp32 copy is 268 MB, written and re-read 24 times per step, for a module worth **0.3% of the
model's arithmetic**. Then the mixing itself touches the four streams several more times.

The fix is not a new algorithm. It is telling the compiler to treat the whole sequence as one
operation, so the intermediate values stay in registers and are never written to memory at all.

| | elementwise launches | small reductions |
| --- | ---: | ---: |
| before | 15,027 | 2,254 |
| after | 7,219 | 86 |

**Transferable lesson.** Cost follows memory traffic and kernel count, not FLOPs. A module can be a
rounding error in the arithmetic and a tenth of the runtime. Always ask what fraction of *time* a
component takes, never what fraction of *the model* it is.
