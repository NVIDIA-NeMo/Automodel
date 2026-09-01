# `nemo_automodel/components/checkpoint/` — checkpointing

Adds to the repository root `AGENTS.md`.

Save/load orchestration (`checkpointing.py`, `lifecycle.py`), the state-dict
adapter interface (`state_dict_adapter.py`, `conversion_mapping.py`), stateful
wrappers (`stateful_wrappers.py`), and pinned upstream backports
(`_backports/`, `_torch_backports.py`).

## Round-trip is the contract

A state-dict adapter is only correct if HF → NeMo → HF returns the original
tensors. Every adapter change needs a round-trip test over the affected keys;
"the checkpoint loaded without raising" is not evidence. Silent load failures
are the dominant failure mode here — missing keys get randomly initialized
weights and training continues.

Prefer asserting on loaded values and on the set of missing/unexpected keys
being empty, rather than on the absence of an exception.

## Sharding and dtype

Adapters run under sharded loads, so a transform that materializes or casts a
tensor can break the in-place view the loader relies on. Keep transforms
shape-and-dtype preserving where possible, and when a cast is genuinely needed,
verify the value actually lands in the model.

## Backports

`_backports/` and `_torch_backports.py` vendor upstream PyTorch code to pin
behavior across torch versions. Do not edit them to fix a local bug — note the
upstream revision, keep the vendored copy faithful, and put the fix in
Automodel code that calls it.
