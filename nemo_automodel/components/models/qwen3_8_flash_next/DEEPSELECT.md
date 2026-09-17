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

## Installation

The AutoModel Dockerfile builds and installs the pinned extension by default
(`INSTALL_DEEPSELECT=true`), targeting SM90a, SM100a and SM103a. It installs into
the base interpreter's site-packages, inherited by the image's uv environment.
The backend remains opt-in; ordinary Torch selection does not import DeepSelect.

For an existing environment, run from the AutoModel checkout on a compute node:

```bash
DEEPSELECT_PYTHON="$(command -v python)" DEEP_SELECT_CUDA_ARCHS=90a \
  bash docker/common/install_deepselect.sh
```

The script uses uv, builds against that interpreter's installed PyTorch and
CUDA toolkit, and pins the source revision above. Git, a C++ compiler, Ninja,
setuptools and NVCC >= 12.9 are required by this upstream build.
Set `DEEP_SELECT_CUDA_ARCHS` to a semicolon-separated subset of
`90a;100a;103a` when building for different targets.

The checked-in build patch adds H100 to the supported build targets and makes
the package version deterministic across uv's metadata/wheel phases. It also
permits a CUDA build without a visible GPU. Selection kernels are unchanged.
No experiment-directory `PYTHONPATH` is needed.

This source-built extension is installed outside uv's project resolution,
following the image's prebuilt-extension pattern; a standalone `uv sync` does
not install it. Use the image or installer above, and install it in every
training rank's environment. Docker builds can opt out with
`--build-arg INSTALL_DEEPSELECT=false`.
