# MiniMax M3 MSA kernels

Forward uses [official MSA at 80434d7f](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac); backward lives here.
Install with `uv sync --locked --extra msa` (Linux, SM100, CUTLASS DSL 4.6.2); use `uv run` so torch can find `ninja`.
`msa_patch.py` lazily fixes the official CUDA 12.9 `nvvm.fmax` binding for DSL 4.6.2, preserving fp32 conversion, the third operand and `loc`/`ip`.
It checks MSA module ownership and replaces only its helper before JIT; remove the patch when the pinned upstream uses inferred result types.
`msa_schedule.py` adapts the saved forward CSR to 8-query task rows; `msa_task_build_sm100.py` builds the locality-ordered task tables and the CTA walk on the device without a host sync; `msa_backward_preprocess_sm100.py` computes delta; `msa_backward_sm100.py` is the main kernel and its wrapper; `msa_grad_finalize_sm100.py` casts the gradient pools to BF16.
dK/dV accumulate with FP32 atomics and dQ with packed 16-bit atomics: `MSA_M3_DQ_ACCUM=fp16` (default; saturates at |dQ| >= 65504) or `bf16` (FP32 range, 8 mantissa bits).
`MSA_M3_TASK_BUILD=fused|torch` (default `fused`; `torch` restores the eager chain with one host sync), `MSA_M3_TASK_ORDER_CHUNK` (default 2048, 0 disables locality ordering) and `MSA_M3_ROWS_PER_CTA` (forces the CTA walk length) are sweep switches read at import.
Validate pin changes with the MSA CPU tests and `tests/functional_tests/models/minimax_m3_vl/test_msa_sm100.py` on SM100; PP=2 requires two GPUs.
