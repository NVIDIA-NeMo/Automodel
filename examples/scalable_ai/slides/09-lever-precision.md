# Lever 4: precision where it is free, and a lever that did nothing

Two changes, both replacing fp32 with bf16 where nothing needs the extra precision. One worked and
one did not, and the pair is more instructive than either alone.

**Gradient reduction (row 5): no effect.** The profile's single largest kernel was the gradient
reduction across GPUs, running in fp32 while the parameters are bf16. Halving that traffic is
obviously good. Measured: 0.355 s to 0.356 s. **Nothing.**

Why: the profile that identified it was captured at a *different batch shape*, one with 32 gradient
accumulation steps. There the reduction ran often enough to be 18% of the step. At one accumulation
step it happens once, and it is already overlapped with computation. The finding was real; it had
been fixed by an earlier change to the batch shape.

**Vocabulary projection (row 6): 1.08x.** The final projection to 163,840 vocabulary entries was
kept in fp32, as the released model does. In bf16 it is a third faster, and it is 24% of all the
useful work in the model. Measured: 0.356 s to 0.330 s.

This one changes numerics, so it is opt-in and labelled, not a silent default.

**Transferable lesson, and the one to remember.** A profile is a measurement of one configuration,
not a property of the program. When you change the configuration, the profile is stale. Optimising
against a stale profile is how you spend a week making something faster that no longer matters.
