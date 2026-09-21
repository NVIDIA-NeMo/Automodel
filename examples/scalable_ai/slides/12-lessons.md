# What to take away

**The numbers.** The only full-model stock `transformers` point that completed our FSDP-only sweep
used micro-batch 1 and accumulation 1: 3.67% MFU and 17.4k tokens per second. On the training-ready
path, the full model moves from 4.80% to 11.81% MFU and 22.8k to 56.0k tokens per second. At the
baseline's own shape the code changes alone give 1.67x and free 23 GB; handing the GPU more work
gives the rest.

**The five ideas, in the order they paid off:**

1. **Sparsity has to be real.** An architecture's asymptotic advantage is a claim about a kernel. If
   the kernel computes densely and masks, you pay for the dense version and get credited for the
   sparse one.
2. **Ask what is being moved.** Distributed training is mostly a question of whether you move
   activations or weights. Pick the smaller one.
3. **Cost follows memory traffic and kernel count, not FLOPs.** A module worth 0.3% of the model's
   arithmetic cost 12% of the runtime.
4. **Check the shape before writing a kernel.** Batch size and accumulation were worth more than
   every kernel change combined, and cost nothing to try. Memory optimisations are throughput
   optimisations one step removed: they raise the batch size you can afford, which raises
   utilisation again.
5. **A profile describes one configuration, not the program.** Every time you change the shape, the
   ranking changes. Re-measure.

**And the meta-lesson.** Of the changes attempted, three did not work: one kernel that was 2.1x in
isolation and worthless in context, one memory optimisation that cost time, and one profile reading
that pointed at the wrong line of code entirely. They are in the appendix because they took as long
as the successes and taught more.

**The honest ceiling.** The best configuration measured reaches 16.4%, and the full model 11.8%.
Reaching 20% needs roughly another 1.2x, and the remaining time is now spread thin rather than
concentrated in a hot spot: no single kernel is more than 12% of the step. That calls for graph
capture or whole-block compilation rather than another point fix. There is no obvious next win,
which is usually the sign to stop.
