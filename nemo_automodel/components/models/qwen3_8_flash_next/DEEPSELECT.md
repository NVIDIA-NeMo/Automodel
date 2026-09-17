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
The build adds SM90a through `NVCC_APPEND_FLAGS` to the upstream SM100a/SM103a
targets. Upstream source files, including `setup.py` and CUDA kernels, are unchanged.

GPU coverage is in
`tests/functional_tests/models/test_qwen3_8_flash_next_deepselect.py`.
It includes selected-set equivalence, dense-prefix/tail behavior, padded query
rows, non-aligned key-row storage, packed documents, attention output/gradient
parity, and activation-checkpoint recomputation consistency.

## Installation

The AutoModel Dockerfile clones the pinned upstream source, builds a wheel, and
installs it by default (`INSTALL_DEEPSELECT=true`). The wheel targets SM90a,
SM100a and SM103a. It installs into system site-packages, inherited by the image's
uv environment. The backend remains opt-in.

For an existing environment, build on a compute node with its PyTorch and CUDA
toolkit already installed. Git, pciutils (for `lspci`), a C++ compiler, Ninja,
setuptools and NVCC >= 12.9 are required by this upstream build:

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git
cd DeepSelect
git checkout 0f03b68748b304863fdf0181a11458d04ae533a9
git submodule update --init --recursive
DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
NVCC_APPEND_FLAGS="-gencode=arch=compute_90a,code=sm_90a" MAX_JOBS=8 NVCC_THREADS=2 \
  uv build --wheel --no-build-isolation --python "$(command -v python)" --out-dir dist .
uv pip install --python "$(command -v python)" --no-deps dist/*.whl
cd ..
```

Building the wheel first avoids separate metadata and wheel phases disagreeing
on upstream's timestamped version. The NVCC flag adds H100 support without
modifying upstream source. No patch or experiment-directory `PYTHONPATH` is needed.

A standalone `uv sync` does not install this optional source-built extension.
Use the image or commands above in every training rank's environment.
Docker builds can opt out with `--build-arg INSTALL_DEEPSELECT=false`.
