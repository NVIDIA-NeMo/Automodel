# Official MSA compatibility

MiniMax M3 uses official [MiniMax-AI/MSA at
80434d7f](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac)
for sparse forward. `msa_patch.py` applies the compatibility fix when
`msa._resolve_msa_forward()` first loads that dependency, before JIT compilation.
Installation remains `uv sync --locked --extra msa`.

The former fork commit `bd37d095345e502e4475bcbbcb8e91fd39f60847` has exactly
that official commit as its parent. Its four-file diff contains:

| Change | Owner in AutoModel |
| --- | --- |
| Pin `nvidia-cutlass-dsl==4.6.2` | `msa` extra in `pyproject.toml` |
| Pin `quack-kernels==0.6.4` | Existing Linux base dependency in `pyproject.toml` |
| Use inferred-result `nvvm.fmax` | `kernels/msa_patch.py` |

The fork repeats the pins in its `pyproject.toml` and two requirements files.
Official MSA's lower bounds accept both versions, so AutoModel can own the
pins without rewriting dependency metadata. QuACK 0.6.4 also requires CUTLASS
DSL 4.6.2; there is no separate QuACK kernel change to carry.

The official helper selects the old explicit-result-type NVVM binding on CUDA
12.9. CUTLASS DSL 4.6.2 instead exposes `fmax(a, b, *, c=None, ...)` for both
CUDA flavors. The patch preserves the optional third operand, fp32 conversion,
`dsl_user_op`, and `loc`/`ip` while removing the explicit result type.

The existing resolver is the **Seam**. Its returned launchers remain the
**Interface**, and the compatibility **Implementation** stays in the owning
model's kernel **Module**. This keeps compatibility knowledge local without
adding a backend flag, registry, installer, or forward-kernel copy. The forward
and backward tensor contracts, including canonical support, stay intact.

| Alternative | Maintenance and installation cost |
| --- | --- |
| Model-private runtime patch (chosen) | One helper at the existing resolver; normal uv installation |
| Unapplied source `.patch` file | Requires a separate source-patching/build step after dependency resolution |
| Vendored forward or maintained fork | Carries the full upstream implementation or another repository |

MSA exposes its CuTe code through top-level `src` imports. The patch checks the
helper's file belongs to the loaded MSA package, then replaces only that module's
`fmax`. Its softmax code looks up this helper through the same module. The
replacement lasts for the process lifetime; installed files and CUTLASS globals
are untouched. A normal model import does not load MSA or CuTe. The resolver
guards the complete sparse import with `safe_import`.

Remove the compatibility helper and its resolver call when the pinned official
revision adopts the inferred-result binding. Before changing that pin, run the
MSA CPU tests and `tests/functional_tests/models/minimax_m3_vl/test_msa_sm100.py`
on SM100 with a clean official installation. The functional suite checks packed
O/LSE/dQ/dK/dV parity, nontrivial top-16 support, and checkpointed parameter
gradients; its two-device cases additionally require two SM100 GPUs.
