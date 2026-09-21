# What to take away

**The numbers.** 4.80% to 11.51% MFU on the full model, 22.8k to 54.6k tokens per second, peak
memory from 60.5 GB to 37.1 GB. Starting from an implementation that could not train the model at
any sequence length.

**The five ideas, in the order they paid off:**

1. **Sparsity has to be real.** An architecture's asymptotic advantage is a claim about a kernel. If
   the kernel computes densely and masks, you pay for the dense version and get credited for the
   sparse one.
2. **Ask what is being moved.** Distributed training is mostly a question of whether you move
   activations or weights. Pick the smaller one.
3. **Cost follows memory traffic and kernel count, not FLOPs.** A module worth 0.3% of the model's
   arithmetic cost 12% of the runtime.
4. **Check the shape before writing a kernel.** Batch size and accumulation changed more than any
   single kernel here, and cost nothing to try.
5. **A profile describes one configuration, not the program.** Every time you change the shape, the
   ranking changes. Re-measure.

**And the meta-lesson.** Of the changes attempted, three did not work: one kernel that was 2.1x in
isolation and worthless in context, one memory optimisation that cost time, and one profile reading
that pointed at the wrong line of code entirely. They are in the appendix because they took as long
as the successes and taught more.

**The honest ceiling.** We are at 14.7% in the best configuration. Getting to 20% needs another 1.4x
and the remaining time is now spread thin rather than concentrated in a hot spot, which means graph
capture or whole-block compilation rather than another point fix. There is no obvious next win.
