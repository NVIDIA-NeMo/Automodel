# Lever 4: fuse the hyper-connections

The trace said elementwise work was 38 to 47% of GPU time across 15,027 launches. Most of it is the
mHC mixer, which runs twice per layer:

```python
flat = hidden_streams.flatten(start_dim=2).float()   # materialises an fp32 copy of the whole stack
mix  = F.linear(_rms_norm_last_dim(flat, eps), fn)   # then reads it back
```

At micro-batch 4 that fp32 copy is 268 MB, written and re-read 24 times per step, for a module worth
0.3% of the model's FLOPs. The collapse and expand steps touch the 4-copy residual stack several more
times each.

**The change.** Move the mixer, collapse and expand into module-level cores and let a new backend
flag replace them with `torch.compile`d versions, once per process. Inductor fuses the cast into the
normalisation, so the fp32 copy is never materialised, and collapses the gate chains.

```yaml
model:
  backend:
    compile_hc: true
```

| 12 layers, micro-batch 4 | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| before | 0.599 s | 11.69% | 54.2 GB |
| after | 0.561 s | 12.47% | 48.2 GB |

Elementwise launches fell from 15,027 to 7,219 and the Sinkhorn reductions from 2,254 to 86.
**1.07x and 6 GB.** The refactor is bit-identical to the original formulas in eager mode.
