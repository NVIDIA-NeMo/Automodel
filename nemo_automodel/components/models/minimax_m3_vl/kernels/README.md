# MiniMax M3 MSA kernels

Forward uses [official MSA at 80434d7f](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac); backward and its delta preprocess live here.
Install with `uv sync --locked --extra msa` (Linux, SM100, CUTLASS DSL 4.6.2); use `uv run` so torch can find `ninja`.
`msa_patch.py` lazily fixes the official CUDA 12.9 `nvvm.fmax` binding for DSL 4.6.2, preserving fp32 conversion, the third operand and `loc`/`ip`.
It checks MSA module ownership and replaces only its helper before JIT; remove the patch when the pinned upstream uses inferred result types.
Validate pin changes with the MSA CPU tests and `tests/functional_tests/models/minimax_m3_vl/test_msa_sm100.py` on SM100; PP=2 requires two GPUs.
