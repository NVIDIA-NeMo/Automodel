# `nemo_automodel/components/moe/` — mixture of experts

Adds to the repository root and `nemo_automodel/components/models/AGENTS.md`.

Expert layers (`experts.py`, `mok_experts.py`), routing and load balancing
(`layers.py`, `load_balance_metrics.py`, `router_replay.py`), parallelism
(`parallelizer.py`, `tp_plan_validation.py`), FSDP2 integration
(`fsdp_mixin.py`), state-dict handling (`state_dict_mixin.py`,
`state_dict_utils.py`), and the DeepEP/HybridEP token dispatchers under
`megatron/` and `uccl_ep/`.

## Gradient synchronization

Expert parameters are sharded over a different mesh dimension than dense
parameters, so the default FSDP2 reduction is wrong for them. `MoEFSDPSyncMixin`
exists to fix this and every MoE model must inherit it. When you touch expert
sharding, prove gradients with a real multi-process test — a single-process loop
over simulated ranks cannot catch a missing collective, because autograd sums
replicated contributions for free.

## Dispatch backends

The expert dispatcher is config-selected (`torch`, `deepep`, `hybridep`), and
the backends are not interchangeable in behavior:

- They differ in supported topologies. Do not switch a recipe's dispatcher to
  fix an unrelated failure without saying why in the PR.
- DeepEP/HybridEP kernels are JIT-built against a pinned commit. Changing the
  pin is a build-and-dependency change; see
  `tests/unit_tests/test_deepep_pin_consistency.py`.
- Dispatch is not guaranteed bitwise-reproducible across calls. Combining a
  non-deterministic dispatcher with activation checkpointing can fail
  recomputation even when routing itself is deterministic.

## Quantization interaction

Expert weights are frequently loaded through in-place views. A cast inside a
state-dict adapter can break that view and leave experts silently randomly
initialized — training proceeds with a plausible-looking but far too high loss.
When you change expert loading under quantization, assert loaded values, not
just that loading did not raise.

## Sequence packing

THD packed-sequence support is per-model, not universal. Check the model's
capability flag rather than assuming; models without THD support must use
padded (BSHD) batches.
