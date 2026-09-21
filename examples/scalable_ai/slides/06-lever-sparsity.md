# Lever 1: sparsity has to be real

> Row 3: 0.505 s to 0.388 s. **1.30x**, and 11 GB.

The profile said attention was 5% of time. For a model whose headline feature is sparse attention,
that is too good, and it was. The portable implementation does this:

```python
scores = query @ keys.transpose(-1, -2)     # the whole matrix
scores = scores + attention_mask            # then hide most of it
```

Every sliding-window layer does the work of full attention and then discards it. The model is
credited for the 128 positions it should have looked at, and charged for all 2048.

```
executed, 12 layers, forward and backward     9.90 TFLOP per GPU per step
credited                                      1.39 TFLOP
```

Roughly 8.5 of every 44 TFLOP the GPU runs are thrown away. This model's 512-dimensional heads make
it four times worse than a conventional design, because the score matrix scales with head dimension.

The sparse kernels instead compute which keys each query needs and read only those, so the dense
matrix never exists.

**Transferable lesson.** An architecture's asymptotic advantage is a claim about the kernel, not
about the model. If nobody wrote that kernel, the advantage is a cost.
