# Lever 2: move tokens, not weights

> Row 2: 0.576 s to 0.505 s. **1.14x**.

With 64 experts spread over 8 GPUs, every token must reach the 6 experts it was routed to. Two
independent choices:

**How tokens get there.** The simple approach gathers *every* token onto *every* GPU, then each GPU
computes only its own experts and the results are summed back. Every GPU sees eight times the data
it needs. The alternative, DeepEP, sends each token only to the GPUs that will use it.

**How the experts compute.** The simple approach loops over the local experts, one matrix
multiplication per expert on a slice of the tokens. A grouped matrix multiplication does all of them
in a single kernel with one launch.

| | step time |
| --- | ---: |
| loop over experts, gather all tokens | 0.576 s |
| grouped multiplication, route tokens | 0.505 s |

**Why only 1.14x** when this is 52% of the useful work? Because at this batch size each expert
receives only a few hundred tokens, so its matrix multiplication is small no matter how you launch
it. The same change is worth more at a larger batch, which is slide 10.

**Transferable lesson.** In a distributed model, ask what is being moved. Moving activations scales
with tokens; moving weights scales with model size. Prefer whichever is smaller.
