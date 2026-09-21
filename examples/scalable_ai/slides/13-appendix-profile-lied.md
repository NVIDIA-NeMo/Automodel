# Appendix A: the profile pointed at the wrong line

After the levers, the hottest single kernel was a 32x32 matrix-multiply tile burning **60 ms, 12% of
GPU time**, moving 6.4 GB at 108 GB/s. That is 3% of the machine's bandwidth.

I identified the culprit from the kernel's shape: the hyper-connection mixer projection, which
multiplies a very wide input by a matrix with only 24 output columns. Twenty-four columns means one
tile of work, so the GPU runs 256 blocks and idles. Reasonable diagnosis. So I wrote a kernel for it
(appendix B).

**The 60 ms kernel did not move.** I verified the new kernel was running and that no library call
remained for that operation, and the 60 ms kernel was still there.

Re-reading the trace with operator shapes attached:

```
CompiledFunctionBackward  [[4, 2048, 4], [4, 2048, 4], [4, 2048, 4, 4]]
```

Those are the *other* part of the hyper-connection: mixing the four residual streams, written as

```python
torch.matmul(comb.transpose(-1, -2), hidden_streams)
```

a matrix multiplication whose three dimensions are all **4**, batched over every token. The library
serves a 4x4x4 multiply with a 32x32 tile. The bandwidth reasoning was right; the target was wrong.

**The fix was not a kernel.** Write the 4-way mix as four multiply-adds and let the compiler fuse it
into one pass.

| | step time | MFU | GPU kernel time |
| --- | ---: | ---: | ---: |
| as a batched matrix multiply | 0.515 s | 13.58% | 484 ms |
| as fused multiply-adds | 0.483 s | 14.48% | 432 ms |

**Lesson.** A profile tells you which kernel is hot. It does not tell you which line of your code
owns it. Confirm attribution before optimising, or you will optimise something real and gain nothing.
