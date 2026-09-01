# `nemo_automodel/components/distributed/` — parallelism

Adds to the repository root `AGENTS.md`.

**Load the `nemo-automodel-distributed-training` skill before changing anything
here.**

Meshes (`mesh.py`, `mesh_utils.py`), FSDP2/HSDP (`fsdp2.py`,
`megatron_fsdp.py`), DDP (`ddp.py`), TP plans (`optimized_tp_plans.py`,
`parallel_styles.py`), pipeline parallelism (`pipelining/`), context
parallelism (`context_parallel/`, `blockdiag_cp/`, `cp_vision_frame_shard.py`),
activation checkpointing (`activation_checkpointing.py`), and gradient
utilities (`grad_utils.py`, `thd_utils.py`).

## Correctness bar

Distributed bugs here are silent: the job runs, the loss looks reasonable, and
the model is wrong. Treat changes to loss normalization, gradient accumulation,
reduction scope, or collective ordering as correctness changes requiring
evidence, not refactors.

- **Rank symmetry.** Every rank must enter compatible collectives on every
  code path, including early exits and error paths. An asymmetric branch
  deadlocks rather than failing fast.
- **Reduction domains.** A gradient must be reduced over exactly the mesh
  dimension its parameter is replicated across — no more, no fewer.
- **Exactly-once.** Reductions must fire once per parameter per step. Both a
  missing and a duplicated reduce scale gradients wrongly.
- **Dtype agreement.** Every contribution entering one reduction must share a
  dtype. Zero-fills for unused parameters are a known source of mismatch.
- **Async lifetime.** Own the `Work`/event for every async collective and wait
  on it before the result is read or its buffer is reused.

## Testing

A single-process loop over simulated ranks proves nothing about collectives.
Use a real multi-process functional test, compare per-parameter gradients
against a reference topology, and respect the 2-GPU PR functional-test cap
(see `tests/AGENTS.md`). Cover the changed topology plus at least one supported
composed topology (e.g. FSDP+TP, PP+EP).

## TP plans

A model without an entry in the TP `PARALLELIZE_FUNCTIONS` mapping falls back
to a generic plan that can leave modules replicated instead of sharded. That
shows up as an out-of-memory error at `tp_size > 1`, not as a wrong-answer bug
— check for a missing plan before adding memory knobs.
