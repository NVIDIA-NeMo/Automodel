# Qwen3.8 FlashNext QSA selection backends

QSA can use the optional [DeepSelect](https://github.com/deepseek-ai/DeepSelect)
CUDA extension to select compressed key blocks. Enable it in the model backend:

```yaml
model:
  backend:
    _target_: nemo_automodel.components.models.qwen3_8_flash_next.backend.Qwen3_8_FlashNextBackendConfig
    attn: flex
    qsa_topk: deepselect
```

The default is `qsa_topk: torch`. This switch changes block selection in the QSA
indexer; the existing FlexAttention forward/backward still computes attention.

## Requirements and behavior

- Install the `deep_select` module built for the target GPU architecture and the
  PyTorch/CUDA environment used by every training rank. Selecting this backend
  without the optional dependency raises an error.
- This integration supports CUDA FP32 selection scores and exactly 512 selected
  blocks (`indexer_budget / indexer_compress_ratio == 512`). For FlashNext this
  corresponds to a token budget of 2048 and compression ratio of 4.
- The indexer's score calculation remains FP32. DeepSelect consumes per-query
  causal lengths directly. The adapter pads row storage when the extension
  requires alignment; it preserves the logical number of key blocks.
- Dense prefixes, padding, incomplete-block tails, packed document boundaries,
  and CP query offsets retain the existing handling.
- Sparse index order is unspecified. Attention uses the selected set, so tests
  compare membership, attention outputs, and gradients rather than sparse
  ordering. Equal-score ties may choose a different valid top-k set than
  PyTorch and consequently change attention outputs; neither backend promises
  identical tie breaking.
- The selector is non-differentiable, as in the existing QSA indexer. This
  backend does not introduce an indexer auxiliary loss or train its parameters.

## Validated extension

The H100 integration was tested with DeepSelect revision
`0f03b68748b304863fdf0181a11458d04ae533a9`, compiled for `sm_90a`.
That revision's default build targets newer architectures, so its build target
must be adjusted for H100. The CUDA selection kernels were unchanged.

GPU coverage is in
`tests/functional_tests/models/test_qwen3_8_flash_next_deepselect.py`.
It includes selected-set equivalence, dense-prefix/tail behavior, padded query
rows, non-aligned key-row storage, packed documents, attention output/gradient
parity, and activation-checkpoint recomputation consistency.

## Installing the validated H100 extension

Use the training environment's Python, PyTorch and CUDA toolkit (NVCC >= 12.9
for this pinned build script). Build on an allocated compute node:

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git
cd DeepSelect
git checkout 0f03b68748b304863fdf0181a11458d04ae533a9
git submodule update --init --recursive
```

At this revision, set the following values in `setup.py` for H100:

```python
cc_flag = ["-gencode", "arch=compute_90a,code=sm_90a"]
datetime_rev = "20260916.000000"
```

The fixed timestamp replaces the generated build time so uv's metadata and
wheel phases report the same version. These are build-only changes; the
selection kernel source is unchanged.

```bash
DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA MAX_JOBS=8 NVCC_THREADS=2 \
  uv pip install --python "$(command -v python)" --no-build-isolation --no-deps .
```

Install the resulting extension in the environment used by every rank.
The extension is optional and is not added to AutoModel's base dependencies.
