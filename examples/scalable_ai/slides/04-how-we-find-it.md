# How to find what to fix

Switching to an implementation written for training gets the model running at **6.1% MFU**. Now the
question is where the other 94% goes. Guessing is expensive; a kernel trace is not.

```yaml
benchmark:
  torch_profile_start: 10      # capture one step, after warm-up
  torch_profile_end: 10
  torch_profile_dir: traces/run
```

GPU time for one step, every kernel sorted into a bucket:

| category | time | share | number of kernel launches |
| --- | ---: | ---: | ---: |
| elementwise | 188 ms | 38% | 15,027 |
| matrix multiplication | 125 ms | 25% | 1,188 |
| communication between GPUs | 101 ms | 20% | 189 |
| attention | 24 ms | 5% | 99 |
| small reductions | 20 ms | 4% | 2,254 |

**Read the table against slide 2.** 76% of the *useful* work is matrix multiplication, but only 25%
of the *time* is. Elementwise work, which produces almost no useful FLOPs, is the largest consumer.
Fifteen thousand kernel launches for twelve layers is roughly 1,250 per layer.

Three specific findings, each of which becomes a slide:

1. Attention is only 5% of time, which is suspicious for a model whose attention is supposedly the
   interesting part. It means the sparse path is not being used.
2. The single biggest kernel is one gradient reduction, in fp32, at 90 ms.
3. The 2,254 small reductions are the hyper-connection normalisation iterations.

**Method, not magic.** Everything after this slide comes from this table.
