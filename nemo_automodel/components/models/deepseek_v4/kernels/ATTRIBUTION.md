# DeepSeek V4 Kernel Attribution

This directory contains TileLang kernels adapted for AutoModel's DeepSeek V4
implementation.

## Miles DeepSeek V4 Kernels

The vendored sparse attention and indexer kernels were adapted from the Miles
DeepSeek V4 implementation:

- Upstream project: https://github.com/yueming-yuan/miles
- Upstream branch: `deepseek-v4`
- Upstream revision: `e561465d0b9bbf06188b7a5e2020dc7fd691f732`
- Upstream source tree:
  https://github.com/yueming-yuan/miles/tree/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops
- Upstream license: Apache License 2.0
- Upstream copyright notice from the Miles license: Copyright 2025 Zhipu AI
- Upstream NOTICE file: none present at the referenced revision

Source mapping:

| Local file | Upstream file |
| --- | --- |
| `sparse_attention.py` | `miles_plugins/models/deepseek_v4/ops/attention_core.py` |
| `tilelang_indexer.py` | `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer.py` |
| `tilelang_indexer_bwd.py` | `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_bwd.py` |
| `tilelang_indexer_fwd.py` | `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_fwd.py` |
| `tilelang_sparse_mla_bwd.py` | `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py` |
| `tilelang_sparse_mla_fwd.py` | `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py` |

Local modifications include adapting the kernels to AutoModel's DeepSeek V4
tensor layouts, packed-sequence dispatch, optional backend selection, and
forward/backward parity tests against the torch reference implementation.

## DeepSeek TileKernels

The Sinkhorn optimized path imports DeepSeek TileKernels at runtime. AutoModel
does not vendor TileKernels source code.

- Upstream project: https://github.com/deepseek-ai/TileKernels
- Upstream revision used for validation: `36d9e45d38e204ebb87e6f6e833821eee0482fe5`
- Imported symbol: `tile_kernels.modeling.mhc.ops.sinkhorn_normalize`
- Upstream source:
  https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/modeling/mhc/ops/sinkhorn.py
- Upstream license: MIT License
- Upstream copyright notice: Copyright 2026 DeepSeek
