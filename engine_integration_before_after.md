# Molt ↔ AutoModel Engine Integration: Responsibility Changes

Current status of the two branches (2026-08-30):

| Repo | Branch / PR | Diff vs main | Status |
| --- | --- | --- | --- |
| NVIDIA-NeMo/Automodel | `huiyingl/feat/datum-forward-backward` / [#3614](https://github.com/NVIDIA-NeMo/Automodel/pull/3614) | 53 files, +3762/−935 | draft, CI green, full unit suite 13339 passed |
| NVIDIA-NeMo/labs-molt | `huiyingl/feat/sft-automodel-engine-integration` / [#92](https://github.com/NVIDIA-NeMo/labs-molt/pull/92) | 49 files, +2599/−2939 | draft, CI green, 234 unit tests passed, requirements pin the AM branch HEAD |
| verl-project/verl | `huiyingl/automodel-engine-v2` | 2 files, +1505/−336 | branch only, 48 CPU adapter tests passed against the AM branch |

Related PRs: HybridEP token equalization [#3641](https://github.com/NVIDIA-NeMo/Automodel/pull/3641) (merged into AM main); VLM PP validation [#3759](https://github.com/NVIDIA-NeMo/Automodel/pull/3759) and the NemotronParse loss fix [#3760](https://github.com/NVIDIA-NeMo/Automodel/pull/3760) (independent drafts split out of #3614).

---

## Part 1 — RL (molt) side: before → after

Only responsibilities that changed hands are listed. Optimizer CPU offload, the critic value head, RL losses, RL input packing, and checkpoint/refit/Ray orchestration belong to molt both before and after, so they are not in this list.

### 1. Training execution loop

**Before**: `molt/trainer/fsdp/strategy.py` implemented the whole execution path in its `backward()` / `optimizer_step()` — gradient-accumulation bookkeeping, deferred FSDP sync via `no_sync`, distributed gradient finalization across EP/TP (calling AM's low-level `scale_grads_and_clip_grad_norm`), clipping, `optimizer.step()`, `zero_grad`, and scheduler advancement. Trainers had to interact with this state machine on every step.

**After**: the whole execution path belongs to AM's **`nemo_automodel/engine/_engine.py` (Engine)**. Molt trainers write the algorithm in its natural order:

```python
out = actor(...)                       # Engine.forward (ordinary nn.Module call)
loss = policy_loss(...)                # RL loss stays in the molt trainer
actor.model.backward(loss)             # Engine.backward
actor.model.step()                     # Engine: finalize + clip + step + scheduler at the accumulation boundary
```

`FsdpStrategy` no longer implements training execution; it keeps only runtime glue: optimizer CPU offload (`CpuOptimizerOffloader`) and the value head's `sync_replicated_grads`.

### 2. Log-probs / entropy under TP (vocabulary-parallel statistics)

**Before**: `molt/trainer/fsdp/packing.py::log_probs_from_vocab_parallel_logits` — when TP shards logits into a vocab-sharded DTensor, molt computed selected-token log-probs itself without gathering the vocabulary; entropy went through molt's `compute_entropy` + `unshard_dtensor` (which first materializes the vocab axis).

**After**: `molt/trainer/fsdp/packing.py` is deleted. The computation belongs to AM's **`nemo_automodel/components/loss/vocab_parallel.py`** — `token_log_probs` / `token_entropy`, one entry point for both dense tensors and vocab-sharded DTensors, and entropy never gathers the vocabulary either. The molt actor is down to two imports and two calls.

### 3. Routing Replay (R3) lifecycle management

**Before**: `molt/models/base.py` (~lines 452-476) managed AM's per-gate `RouterReplay` handles itself: clearing the global registry, walking the module tree to install a handle on every MoE gate, subclassing a `_SentinelRouterReplay` to implement the `-1` (keep-live-selection) sentinel, and relying on construction order for layer alignment. Molt had to understand the internal structure of AM gates.

**After**: this belongs to AM's **`nemo_automodel/components/moe/router_replay.py::RouterReplayAdapter`** — gate discovery by decoder-layer id, handle binding, `-1`-row fallback to live routing, trailing-padded-token handling, and state restoration on exceptions are all inside AM. Molt only does `adapter = RouterReplayAdapter(model)` and wraps the forward in `with adapter.replay(routes):`, with routes prepared in the model input's token order.

### 4. VLM media merging / on-the-fly packing primitives

**Before**: `molt/utils/vlm_utils.py` carried its own `_pad_to_common_hw` and merging logic to pad variable-resolution image patches to a common shape and concatenate them into a batch tensor; there was no reusable component for packed-VLM sample boxing.

**After**: merging/padding belongs to AM's **`components/datasets/vlm/utils.py::merge_media_values`** (including 4-D variable-resolution padding); sample boxing belongs to AM's **`components/datasets/vlm/neat_packing_vlm.py::pack_vlm_samples`** plus the two collaters. Molt's `pack_vlm_batch` (`molt/models/packing.py`) keeps only the RL-semantic part (valid-span extraction, restore indices) and calls AM for the physical packing.

### 5. HybridEP token equalization for dynamic batches

**Before**: nobody owned this — rollout-produced variable-length/packed batches give each rank a different token count, and the hybridep backend deadlocks or SIGABRTs (a 4-token alignment constraint), so RL dynamic batches could not train on hybridep at all.

**After**: this belongs to AM's **`components/moe/megatron/token_dispatcher.py::_HybridEPManager.dispatch()`** (#3641, already in AM main): all-reduce the EP-group max token count, round up to the 4-token alignment, pad rows route to no expert, and `combine()` slices the padding back off. Callers (molt and AM recipes alike) are completely unaware of the constraint.

---

## Part 2 — veRL side: before → after

Branch `huiyingl/automodel-engine-v2` updates veRL's AutoModel backend (upstream PR #5407 lineage) to the same boundary. The diff vs verl main is deliberately narrow: only `verl/workers/engine/automodel/transformer_impl.py` (±930 lines) plus a new CPU contract-test file (+911 lines).

### 1. Training execution loop

**Before**: verl main's AutoModel adapter hand-rolled the execution loop out of AM internals — manual `prepare_for_grad_accumulation` / `prepare_for_final_backward` calls, a manually set `MoEAuxLossAutoScaler.main_loss_backward_scale` (and only when ep>1), and an `optimizer_step` that invoked `scale_grads_and_clip_grad_norm` itself. The same shape molt main had: the RL framework reimplements the execution layer.

**After**: the loop is the standard calls — `engine(**inputs)` → veRL's own loss → `engine.backward(loss, scale_wrt_gas=False)`, with `set_gradient_accumulation_steps(n)` making the last microbatch the boundary forward (deferred FSDP gradient sync re-enabled there so its backward reduces). The boundary microstep stays pending until veRL's `optimizer_step()`, which closes the window — Engine finalizes, clips, and updates there — preserving the split contract Tinker-style callers rely on (optimizer adjustments between backward and step). Gradients are numerically identical to main's raw `loss.backward()` under FSDP2's averaging reducer, while MoE aux-loss scaling and MegatronFSDP summed-gradient compensation move into the Engine. A failed microstep resets the window (`Engine.reset_accumulation`) so an OOM cannot poison later windows; a second `forward_backward_batch` before the step fails closed. The scheduler stays veRL-owned.

### 2. Log-probs / entropy under TP

**Before**: `prepare_model_outputs` called `full_tensor()` on TP-sharded logits — gathering the whole vocabulary before computing log-probs, which blows up memory on large-vocab models.

**After**: vocab-sharded DTensor logits go through AM's `token_log_probs` / `token_entropy` (the same primitives molt uses), never gathering the vocabulary.

### 3. Optimizer construction

**Before**: imported `build_optimizer` from `nemo_automodel.recipes.llm.train_ft` and went through `ConfigNode` — reaching into AM recipe internals.

**After**: `components.optim.build_optimizer` public API with explicit kwargs, an `override_optimizer_config` escape hatch, and fail-fast rejection of fp16 optimizer states.

### 4. Adapter hardening (from the intermediate commit this migration completes)

- Systematic fail-closed validation: PP, CP, LoRA, activation offload, router replay, and fp16 precisions raise explicitly instead of running wrong.
- Packed inputs distinguish TE THD from FlashAttention indexed-mask layouts.
- vLLM refit's `get_per_tensor_param` converts per tensor via the AM adapter's `convert_single_tensor_to_hf` (main's `convert_weight_keys` mis-maps custom-model weight layouts) and fails fast on non-DTensor EP expert weights.
- Checkpointing goes through AM's `Checkpointer` (DCP/safetensors/consolidated); veRL keeps ownership of the scheduler/RNG/step extra payload.

### 5. Tests

**Before**: no unit coverage for the adapter. **After**: 48 CPU tests — mock-Engine tests pinning the call-sequence contract (one GAS window per mini-batch, `scale_wrt_gas=False`, forward-only never touches the training state machine), plus one end-to-end test driving a real nemo-automodel Engine through two accumulating microbatches with exactly one optimizer update at the boundary.


---

## Part 3 — AM side: the Engine and the modules serving RL

### Class hierarchy — inputs, outputs, ownership

```
RL framework (molt / veRL)                          ── owns: batch prep, losses, window timing
│
├── Engine (nn.Module)                              engine/_engine.py — STATEFUL
│     construct(module, optimizer, lr_scheduler?, mesh_context?, max_grad_norm, gas, defer_fsdp_grad_sync)
│     forward(**model_inputs)      → whatever the model returns (logits may be a vocab-sharded DTensor)
│     backward(loss, scale_wrt_gas) → None          (loss: scalar; scale_wrt_gas=False when caller pre-normalized)
│     step()                        → None          (non-boundary: count; boundary: finalize+clip+update+zero+sched)
│     zero_grad() / get_global_grad_norm() / set_gradient_accumulation_steps(n)
│     is_gradient_accumulation_boundary() / reset_accumulation()
│     owns: accumulation window state machine, deferred-FSDP sync, MoE aux-loss scaling,
│           summed-reducer compensation, gradient finalize (EP/TP factors), clip,
│           non-finite-norm update skip, optimizer + scheduler advance
│     does NOT own: loss math, collation/packing/CP prep, pipeline scheduling
│
├── token_log_probs / token_entropy                 components/loss/vocab_parallel.py — STATELESS fns
│     in : logits Tensor|DTensor [.., vocab], targets int64 [..] (global ids), temperature
│     out: fp32 [..] selected-token log-probs / entropy, replicated on every TP rank
│     owns: the TP vocab-shard layout contract (Shard(-1), even chunks) and the
│           no-gather reduction; caller owns target construction and temperature semantics
│     AM-internal consumers: none (RL-only; SFT losses consume log-probs inside fused CE)
│
├── RouterReplayAdapter                             components/moe/router_replay.py — per-model instance
│     construct(model)              → binds one handle per MoE gate, keyed by decoder layer_idx
│     replay(routes int [tokens, global_layers, topk] | None) → context manager
│           -1 row = keep live routing; context must span forward AND backward (AC recompute)
│     owns: gate discovery, layer-id → route-slice mapping, sentinel/trailing fallback,
│           handle state restore on exit/exception
│     caller owns: recording routes, storing them with old_log_probs, packed token-order alignment
│     AM-internal consumers of the adapter: none; the per-gate RouterReplay handle
│     underneath IS AM-internal (gates call replay_selection in their forward)
│
├── _HybridEPManager.dispatch/combine               components/moe/megatron/token_dispatcher.py (in main)
│     reached only through the MoE layer forward — callers never see it
│     owns: per-rank token-count equalization + 4-token alignment inside dispatch
│
├── pack_vlm_samples / merge_media_values / collaters   components/datasets/vlm/ — STATELESS fns
│     in : per-sample dicts (input_ids, labels, media) + get_rope_index
│     out: one THD-packed physical batch (cu_seqlens, positions, media side channels)
│     owns: physical boxing/media merge; caller owns sample selection and restore semantics
│
└── state_dict_adapter.convert_single_tensor_to_hf     per model family — STATELESS method
      in : (fqn, full_tensor, exclude_key_regex, quantization)
      out: [(hf_name, tensor), ...] for vLLM refit streaming
      owns: custom-layout → HF key/shape mapping; caller owns the gather and the refit protocol
```

The ownership rule behind every node: a component owns exactly the knowledge that is
private to AM (model layout, gate structure, kernel constraints, wrapper conventions);
everything expressible in the RL framework's own terms (losses, advantages, when a window
opens, what a sample means) stays with the caller.

### `nemo_automodel/engine/_engine.py` — Engine (core, new)

Wraps an **already-distributed** eager model. Public surface: `forward` / `backward(loss)` / `step()` / `zero_grad()` / `get_global_grad_norm()` / `set_gradient_accumulation_steps()`.

Responsibilities:
- gradient-accumulation window management; non-boundary microsteps defer FSDP gradient sync through the wrapper's `no_sync`;
- backward scaling (the caller can disable GAS normalization with `scale_wrt_gas=False` when it has already normalized the whole window — the RL case); per-microstep scaling of the MoE auxiliary loss;
- at the accumulation boundary: distributed gradient finalization (EP/TP expert replication factors), clipping, `optimizer.step`, `zero_grad`, MoE gate-bias update, FP8 scale precompute, and scheduler advancement;
- compensation for MegatronFSDP's summed-gradient semantics (detected via `calculate_per_token_loss`).

Explicitly **not** its job: loss computation, RL semantics, collation/packing/CP sharding (the caller's job), and pipeline scheduling (PP runs through the AutoPipeline schedule; the Engine raises explicitly).

### `components/loss/vocab_parallel.py` (new)

`token_log_probs(logits, targets)` / `token_entropy(logits)`: selected-token log-probabilities and categorical entropy, one entry point for dense and vocab-sharded DTensor logits; the sharded path uses distributed reductions and never gathers the vocabulary; the dense path upcasts to fp32 in 256-row chunks to bound peak memory. RL actor/reference/critic scoring and the training forward all depend on it.

### `components/moe/router_replay.py` (extended)

`RouterReplayAdapter`: replays rollout-recorded expert selections through the training forward (forward / activation recomputation / backward), with layer-id mapping, sentinel fallback, trailing-token handling, and state restoration built in. The importance-sampling correctness of GRPO/GSPO on MoE models depends on it.

### `components/moe/megatron/token_dispatcher.py` (in main via #3641)

Per-rank token-count equalization plus 4-token alignment inside HybridEP dispatch, making dynamic batches (RL rollouts) usable on the hybridep backend.

### `components/datasets/vlm/` (extended)

`pack_vlm_samples` (sample → THD physical boxing, including mRoPE positions and media side channels), `neat_packed_vlm_collater` / `packed_sequence_thd_vlm_collater`, and `merge_media_values` (variable-resolution media merge/pad). Molt's on-the-fly VLM packing calls these directly at runtime.

### Supporting model/infrastructure changes (small)

- **state_dict adapters** (llama/qwen2/qwen3 gain `convert_single_tensor_to_hf`): per-tensor HF-name conversion for vLLM refit (called by molt's `policy_actor`; the interface itself predates this PR — this PR fills in the pure-passthrough models).
- **`_transformers/model_init.py`**: an in-memory config passed to `from_pretrained(config=...)` is no longer forwarded twice — molt's `load_automodel` loads with a supplied config.
- **`models/muse_glimmer`**: CP preparation for packed TE THD (molt's Muse path).
- **`utils/model_utils.py::squeeze_input_for_thd`**: skips media keys, fixing the item axis of a single-media batch being squeezed away.
- **`optim/scheduler.py`**: `step(increment=1)` default (the Engine calls the scheduler with no arguments).
- **`quantization/fp8.py`**: capability-assignment ordering so the Engine can read the FP8 precompute flag.
- **`distributed/pipelining/autopipeline.py`**: new `eval()` (forward-only schedule entry sharing `step()`'s implementation) — used by AM's own LLM recipe for PP validation, and the hook for future RL PP support.
- **`context_parallel/magi.py`**: fixes including `pad_value=-100` on the labels dispatch, explicit CP token-index exposure, and cp_group refresh for Engine callers.

### AM's own beneficiaries

The LLM/VLM SFT recipes (`train_ft.py` / `finetune.py` / `benchmark.py`) drive their eager paths through the same Engine and drop their duplicated backward/clip/step implementations — the Engine is not an RL-specific component; it is AM's general training-execution layer.
