# FSDP2 Resident and Compute Dtype Validation Matrix

## Purpose

This document defines the validation matrix for selecting between:

- bounded replication for a very small set of precision-sensitive FP32
  parameters, with one coalesced FP32 gradient buffer reduced across each DP
  mesh dimension per optimizer step;
- the optimized single-owner FSDP2 path for larger FP32-resident parameter sets
  with per-parameter transient compute dtypes; and
- the existing dtype-split FSDP2 path for models whose resident parameters use
  multiple storage dtypes.

The design must treat the following as independent properties:

1. **Model resident dtype**: `Parameter.dtype` on the model.
2. **FSDP compute dtype**: the transient dtype used during forward and backward.
3. **Optimizer master weights**: an optional, separate optimizer-owned FP32 copy,
   such as Transformer Engine FusedAdam with `master_weights: true`.

Path selection must use the actual dtypes of parameters owned by the candidate
FSDP unit. It must not infer resident dtype from the model configuration,
checkpoint configuration, compute-dtype metadata, or optimizer settings.

## Resident, Compute, and Optimizer Matrix

| ID | Bulk resident dtype | Sensitive resident dtype | Bulk compute dtype | Sensitive compute dtype | Optimizer FP32 master | Expected FSDP path | Required outcome |
|---|---|---|---|---|---|---|---|
| D1 | FP32 | FP32 | BF16 | FP32 | No or irrelevant | Bounded replication when sensitive bytes fit; optimized single-owner fallback otherwise | One layer-level FSDP unit owns the bulk; sensitive weights stay resident and compute FP32; bulk gradients reduce-scatter FP32; sensitive gradients use one coalesced FP32 all-reduce per step |
| D2 | FP32 | FP32 | FP32 | FP32 | No or irrelevant | Ordinary uniform FSDP | No per-parameter dtype extension or transient casts; one FSDP unit |
| D3 | BF16 | FP32 | BF16 | FP32 | Yes, optimizer-owned | Bounded replication when sensitive bytes fit; dtype-split fallback otherwise | Preserve BF16 sharded bulk storage and its optimizer-owned FP32 masters; update replicated sensitive FP32 weights directly without downcasting their communication |
| D4 | BF16 | FP32 | BF16 | FP32 | No | Bounded replication when sensitive bytes fit; dtype-split fallback otherwise | Preserve BF16 sharded bulk storage; the optimizer directly updates both bulk resident weights and replicated FP32 sensitive weights |
| D5 | BF16 | BF16 | BF16 | BF16 | Yes or no | Ordinary uniform FSDP | Technically valid as a generic FSDP configuration, but reject or warn for a model whose sensitive parameters must remain FP32 |
| D6 | FP32 | BF16 | BF16 | BF16 | No | Dtype-split fallback | Support an unusual checkpoint layout without entering the FP32-resident-only optimized path |
| D7 | BF16 | FP32 | BF16 | BF16 | Yes or no | Dtype-split fallback | Split by storage dtype even though both groups compute in BF16; treat this as an invalid numerical policy when the sensitive parameters require FP32 compute |
| D8 | BF16, FP16, and FP32 mixed | Mixed | Mixed | Mixed | Any | Multi-group dtype-split path or clear error | Isolate every storage/compute group, or fail clearly when the parameters cannot be isolated into distinct owning modules |

D1 is the optimized configuration. D3 is the primary compatibility case for
BF16 model storage with optimizer-owned FP32 master weights.

## Path Selection

Path selection is model-wide for replication and unit-local for either sharded
fallback:

```text
Each explicitly managed module is resident FP32
AND that module's logical size is at or below the replication byte limit
    -> ignore that module's parameters in every FSDP unit
All eligible modules in a model part
    -> install one coalesced FP32 grad buffer reduced over every DP mesh dimension
otherwise, if all floating parameters owned by a candidate unit are resident FP32
AND the per-parameter dtype extension supports the active configuration
    -> optimized single-owner sharded path
otherwise
    -> existing dtype-split sharded path
```

Parameters passed through `ignored_params` must not participate in the
selection decision because they are owned by another sharding or replication
policy.

## Replication Limit Rationale

The `max_replicated_fp32_param_bytes_per_module` default is **8 MiB per managed
module**, not a model-wide budget. The repository currently contains 131
example PEFT configurations with the following LoRA rank distribution:

| LoRA rank (`peft.dim`) | Example count |
|---:|---:|
| 4 | 1 |
| 8 | 64 |
| 16 | 39 |
| 32 | 19 |
| 64 | 8 |

For an FP32 LoRA pair on a linear projection, the logical adapter size is
`4 * rank * (in_features + out_features)` bytes. At the largest rank present in
the examples (`64`), 8 MiB covers both adapter matrices for a square projection
up to hidden size 16,384. It also covers rank 128 at hidden size 8,192. The cap
therefore includes the repository's common adapter dimensions while preventing
a genuinely large managed module from being replicated merely because other
managed modules are small. Oversized modules independently retain sharded
ownership; eligible siblings may still replicate.

## Ownership and Feature Compatibility

The following cases should be crossed with at least D1 and D3.

| ID | Condition | Expected decision | Validation |
|---|---|---|---|
| F1 | A different-dtype parameter is in `ignored_params` | Exclude it from path selection | D1 remains single-owner, and the ignored parameter is not captured by an ancestor |
| F2 | Frozen multimodal module with root ownership | Preserve default traversal | The root owns the frozen parameters without duplicate ownership |
| F3 | Frozen multimodal module with per-layer ownership | Evaluate each owned unit independently | Produce the correct FSDP units and matching collective order across ranks |
| F4 | Frozen multimodal module with replicated ownership | Exclude replicated parameters | Replicated parameters remain unsharded and do not force dtype fallback |
| F5 | Previously child-sharded parameters during root wrapping | Treat them as ignored or already owned | The root callback does not inspect or recapture child-owned parameters |
| F6 | Activation checkpointing enabled | Preserve the selected dtype path | Recomputed forward values and gradients use the correct dtypes |
| F7 | Context parallelism enabled | Preserve ownership selection | Configure unused-parameter reduction on every resulting FSDP unit |
| F8 | Tensor parallelism enabled | Inspect post-TP parameter dtypes | Preserve tied weights and avoid duplicate FSDP ownership |
| F9 | CPU offload enabled | Fall back to the dtype-split path | Preserve correctness without entering the unsupported tensor-extension path |
| F10 | Compiled autograd enabled | Fall back to the dtype-split path | Avoid raising from the optimized extension when the compatible path is available |
| F11 | A parameter shape does not satisfy the optimized extension's sharding constraint | Fall back before wrapping | Avoid a late `NotImplementedError` during the first all-gather |
| F12 | The candidate unit owns no floating parameters | Use ordinary root or container wrapping | Avoid an empty dtype-map failure |
| F13 | Embedding and LM-head weights are tied | Preserve the alias before ownership | Keep one logical parameter, one owner, and stable checkpoint keys |
| F14 | A state dict is loaded after FSDP initialization | Reinstall extensions only for the sharded single-owner fallback | Restore compute metadata without duplicate hooks or extensions; replicated parameters retain ordinary module state |
| F15 | Model and optimizer state are saved and resumed | Preserve the selected storage contract | D1 restores FP32 bulk shards plus replicated FP32 sensitive weights; D3 restores BF16 bulk shards, replicated FP32 sensitive weights, and the optimizer's separate FP32 bulk master state |
| F16 | A selected replicated trainable parameter is unused in an optimizer step | Communicate local-use bits in the coalesced FP32 payload | Zero-fill a missing rank-local contribution only when at least one DP rank used the parameter; keep `grad=None` everywhere when it was globally unused so weight decay and optimizer state do not advance |
| F17 | HSDP uses non-trivial replicate and shard mesh dimensions | Reduce the same coalesced FP32 buffer over both dimensions | Match a global four-rank reference and issue one replicated-gradient all-reduce per mesh dimension |

## Casting and Numerical Assertions

| Path | Required assertion |
|---|---|
| D1 bulk FP32 to BF16 | Perform exactly one transient cast per materialization; keep the resident shard FP32 |
| D1 sensitive FP32 to FP32 | Reuse the resident shard as the gather input; do not allocate a redundant `.to(torch.float32)` result |
| D1 bulk backward | Reduce-scatter gradients in the configured FP32 reduction dtype without changing the resident parameter dtype |
| D1 sensitive backward | Coalesce all selected gradients into one FP32 buffer and all-reduce from FSDP's synchronizing post-backward callback; with deferred synchronization this is once per optimizer step, after accumulation and before scaling/clipping |
| D3 BF16 bulk | Do not create an additional FP32 model copy inside FSDP |
| D3 optimizer | The optimizer owns the separate FP32 master copy and synchronizes updates back to the BF16 model parameter |
| D3 sensitive FP32 | Keep the parameter replicated and FP32; isolate it from DTensor foreach groups and avoid a redundant FP32 master copy when the optimizer supports dtype-aware master ownership |
| All paths | Match an independent FP32 reference for forward output, loss, and gradients within dtype-appropriate tolerances |
| All paths | Preserve resident dtypes and compute metadata across a checkpoint round trip |

The replicated-gradient collective is installed by the `fully_shard` wrapper on
FSDP's root post-backward callback. It follows FSDP's own
`set_requires_gradient_sync` lifecycle, so deferred backward passes accumulate
locally and the synchronizing backward reduces the complete FP32 gradient. A
training loop does not need to call an AutoModel clipping or finalization helper
to make replicated parameters correct.

The same FP32 payload starts with one rank-symmetric validation value and one
local-use value per managed parameter. Those values add only
`4 * (1 + parameter_count)` bytes and do not add a collective. After reduction,
a parameter used on only some ranks receives the missing ranks' zero
contributions and is divided by the full DP world size. A parameter unused on
all ranks retains `grad=None`.

## Executable Coverage

| Matrix cases | Test owner | Observable contract |
|---|---|---|
| D1 replicated and optimized fallback; D2; D4 replicated and dtype-split | `run_fsdp_casting_ownership.py` | Resident dtype, FSDP-unit count, forward/backward and optimizer-step parity against an independent reference |
| D3 optimizer ownership | `run_te_fused_adam_master_ownership.py` | BF16 weights retain optimizer FP32 masters; resident FP32 weights do not allocate a redundant master; the same ownership survives resume |
| D5-D8 and F9-F12 | `test_parallelizer_utils.py` | Unit-local selection chooses ordinary, optimized single-owner, or dtype-split fallback before wrapping |
| F1, F4, and F5 | `test_parallelization_strategies.py` plus the functional root-after-child case | Ignored, replicated, frozen, and already child-owned parameters are not recaptured |
| F6 and gradient accumulation | Functional activation-checkpoint and two-microbatch cases | Recompute and deferred synchronization preserve numerical parity |
| F14-F15 | Functional model and optimizer state reloads | Extension metadata and optimizer master ownership are restored without duplicate state |
| F16 | Unit tests with globally and rank-locally unused parameters | Globally unused parameters retain `grad=None`; rank-local gaps receive the peer contribution without an extra collective |
| F17 | Four-rank invocation of `run_fsdp_casting_ownership.py` | HSDP matches a global reference and reduces one coalesced FP32 payload over each mesh dimension |

## PEFT Matrix

PEFT validation is separate from the full-parameter training requirement.

| ID | Base weights | Adapter weights | Expected status |
|---|---|---|---|
| P1 | Frozen BF16 | BF16 LoRA | Existing supported baseline |
| P2 | Frozen BF16 | FP32 LoRA | Exploratory; verify ignored/frozen ownership and mixed-dtype optimizer support before claiming support |
| P3 | FP32 resident with BF16 compute | FP32 LoRA | Potential optimized-path case; validate adapter ownership and checkpoint behavior separately |
| P4 | Quantized base | BF16 or FP32 adapter | Out of scope unless the same FSDP path already claims quantized-base support |

P2 should not block D3 unless mixed-dtype PEFT support is explicitly included in
the change's scope.

## Profiler Validation Matrix

Resident dtype, compute dtype, sensitive-parameter precision, and optimizer
master-weight behavior must match between frameworks for a performance result to
be considered equivalent.

| Benchmark | AutoModel configuration | Comparison configuration | Purpose |
|---|---|---|---|
| B1 | D1: FP32 resident, BF16 bulk compute, FP32 sensitive compute | The same resident and compute policy | Measure the single-owner optimization fairly |
| B2 | D3: BF16 resident bulk, FP32 sensitive parameters, optimizer FP32 masters | The same resident dtypes and optimizer-master contract | Measure the compatibility path fairly |
| B3 | D1 AutoModel | BF16-resident comparison without the same FP32 model storage | Diagnostic only; label as non-equivalent and do not use it as the adoption comparison |
| B4 | D3 AutoModel | A comparison that downcasts the sensitive parameters to BF16 | Diagnostic only; a faster result may violate the numerical contract |

### Validated D1 Profiler Signature

The following signature was captured on the four-GB200, 16K neat-packing
workload with three active profiler steps. The profiling snapshot also included
the two dependent Qwen runtime optimizations, but those changes do not touch
distributed code, FSDP ownership, or collective payloads. These counts must not
be treated as invariant across arbitrary batch sizes or profiler schedules.

| Metric | Observed value |
|---|---:|
| FSDP ownership units | 57 |
| BF16 parameter all-gather kernels | 1,332 |
| Int32 preemption-signal all-gather kernels | 3 |
| Total NCCL all-gather kernels | 1,335 |
| NCCL reduce-scatter kernels | 171 |
| Baseline scalar all-reduce kernels | 15 |
| FP32 replicated-gradient all-reduce kernels | 3 |
| Total NCCL all-reduce kernels | 18 |
| `_fp32_params` child gathers | 0 |

Bounded replication preserves the parameter all-gather and reduce-scatter
structure, adds one FP32 replicated-gradient all-reduce per active optimizer
step, and removes the sensitive parameter bytes from FSDP gathers. The three
one-element Int32 gathers come from the training scheduler's once-per-step
distributed preemption-signal poll; they are neither parameter materialization
nor part of the replicated-gradient protocol. The focused two-rank functional
profiler independently asserts two parameter all-gathers, one FP32
reduce-scatter, and one FP32 replicated-gradient all-reduce for one optimizer
step.

## Validation Status and Remaining Performance Gate

The implementation and correctness gates are covered by the functional matrix:

1. D1, D2, D3, and D4 have functional forward, backward, and optimizer parity
   coverage against an independent reference.
2. F1, F4, F5, F6, F14, and F15 cover ownership, accumulation, and resume
   behavior.
3. The D1 communication signature above is trace-confirmed.
4. D3 has a real Transformer Engine optimizer probe with FP32 master weights.
5. Negative coverage prevents a model policy requiring FP32-sensitive
   parameters from silently accepting their downcast storage/compute policy.

A matched framework performance comparison remains separate from these
correctness gates. B1 must compare FP32-resident bulk weights with BF16 transient
compute on both sides; B2 must compare BF16-resident bulk weights with the same
optimizer-owned FP32 master contract on both sides. A control that differs in
resident dtype, sensitive-parameter precision, packed-document semantics, or
optimizer work is diagnostic only and must not be presented as the final
adoption comparison.
