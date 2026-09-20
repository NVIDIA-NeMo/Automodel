# The hot kernel was not the one I thought

The custom kernel was correct, fast in isolation, and verified to be running: with it enabled, no
cuBLAS kernel appears for that operation at all. **Yet the 60 ms kernel was still in the trace.**

So it was never the mixer projection. Re-reading the trace with operator shapes attached:

```
CompiledFunctionBackward  [[4, 2048, 4], [4, 2048, 4], [4, 2048, 4, 4]]
```

That is the hyper-connection **expand**, and its stream mix is

```python
torch.matmul(comb.transpose(-1, -2), hidden_streams)
```

a batched GEMM whose M, N and K are all `hc_mult`, which is **4**, batched over every token. cuBLAS
serves a 4x4x4 GEMM with a 32x32 tile. The bandwidth reasoning was right; the target was wrong.

**The fix is smaller than a kernel.** Write the 4-way mix as four fused multiply-adds, accumulating
in fp32, and let `torch.compile` lower it to a single elementwise pass that reads the stream stack
once. It agrees with the matmul to within one bf16 rounding unit.

| 12 layers, micro-batch 4 | step time | MFU | GPU kernel time |
| --- | ---: | ---: | ---: |
| stream mix as a batched matmul | 0.515 s | 13.58% | 484 ms |
| stream mix as fused multiply-adds | 0.483 s | 14.48% | 432 ms |

The 60 ms cuBLAS kernel is gone from the trace, replaced by an 11 ms fused kernel.

**The lesson for the lecture:** a profile tells you which kernel is hot, not which line of code owns
it. Confirm attribution before optimising. A 2.1x kernel aimed at the wrong operation is worth zero.
