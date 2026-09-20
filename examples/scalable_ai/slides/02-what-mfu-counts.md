# What we measure

**Setup.** 8x H100 80GB, NeMo Automodel 26.08 container, FSDP2 with expert parallelism 8,
random init on mock data, 12 steps with 4 warm-up, Adam. Reported step time is the mean of the
measured steps; peak memory is rank 0's `max_memory_allocated`.

**MFU** is credited model FLOPs divided by elapsed time and 989 TFLOP/s, the H100 dense BF16 peak.
The formula counts attention as *sparse*: a sliding-window layer is credited only for the window it
is supposed to attend to.

That definition matters. An implementation that computes dense attention and masks it gets charged
the time but credited only the sparse work.

**The credited FLOPs budget** for the 12-layer configuration used through this deck:

| component | share of credited FLOPs |
| --- | ---: |
| expert GEMMs | 51.6% |
| vocabulary projection | 23.8% |
| attention linears | 16.1% |
| compressor and indexer linears | 4.1% |
| sparse attention BMMs | 3.3% |
| lightning indexer BMM | 0.7% |
| hyper-connection mixers | 0.3% |

91% of the work is large GEMMs. An H100 runs those at 400 to 600 TFLOP/s. So the headroom is real,
and anything that is not a large GEMM is overhead to be minimised. Remember the last row: the
hyper-connection mixers are 0.3% of the work, and they will cost us 12% of the time.
