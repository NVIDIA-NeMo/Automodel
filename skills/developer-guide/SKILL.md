---
name: developer-guide
description: Environment setup and day-to-day development workflow for NeMo AutoModel, including installation, tooling, and code quality commands.
---

# NeMo AutoModel Developer Guide

## Quick Start

Clone and install:

```bash
git clone <repo-url> && cd nemo-automodel
uv sync --locked --extra cuda --extra extra
```

Or use the NeMo-AutoModel container from NVIDIA NGC:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:latest
docker run --gpus all -it nvcr.io/nvidia/nemo-automodel:latest
```

Verify your setup:

```bash
ruff check .
pytest tests/unit_tests/ -v -x --timeout=60
```

## Environment Setup

### Option 1: NeMo-AutoModel Container (NGC)

The container ships with all dependencies pre-installed. Mount your data and code:

```bash
docker run --gpus all -v $(pwd):/workspace -it nvcr.io/nvidia/nemo-automodel:latest
```

### Option 2: uv (Recommended for Local Development)

```bash
uv sync --locked                          # base install
uv sync --locked --extra cuda             # CUDA support
uv sync --locked --extra fa               # flash-attention
uv sync --locked --extra moe              # mixture-of-experts
uv sync --locked --extra vlm              # vision-language models
uv sync --locked --extra diffusion        # diffusion models
uv sync --locked --extra delta-databricks # Delta Lake / Databricks
uv sync --locked --extra all              # everything
```

### Option 3: pip

```bash
pip install -e ".[all]"
```

### Environment Variables

```bash
export HF_TOKEN="hf_..."           # Hugging Face token for gated models
export WANDB_API_KEY="..."         # Weights & Biases logging
export HF_HOME="/path/to/hf_cache" # Hugging Face cache directory
```

## Code Quality

Run formatting and linting before every commit:

```bash
# Step 1: Auto-format all source files (line length, quotes, trailing commas, etc.)
ruff format .

# Step 2: Lint and auto-fix what can be fixed (unused imports, isort, etc.)
ruff check --fix .
```

To check without modifying files (useful in CI or for a dry run):

```bash
ruff format --check .   # exits non-zero if any file would change
ruff check .            # exits non-zero on lint violations
```

To lint a single file or directory:

```bash
ruff format nemo_automodel/components/models/llama/model.py
ruff check --fix nemo_automodel/components/models/llama/
```

### What ruff enforces (from `pyproject.toml`)

| Rule | ID | Description |
|---|---|---|
| Line length | — | 120 characters (formatter) |
| Quote style | — | Double quotes |
| Unused imports | F401 | Auto-removed by `--fix` (ignored in `__init__.py`) |
| Unused variables | F841 | Auto-removed by `--fix` |
| Undefined names | F821 | Error |
| f-string without placeholders | F541 | Error |
| Import sorting | I | isort-compatible ordering, auto-fixed |
| Docstring convention | D101/D103 | Google style (currently ignored — selected then suppressed) |
| No pickle | S301/S403 | Security: forbids `pickle.load` |
| Ambiguous variable names | E741 | Error (e.g., `l`, `O`, `I`) |

Tests (`tests/`) are excluded from lint checks. Docstring rules (`D`) are also relaxed in test files.

### Additional style conventions

- Type hints required on all public API functions and methods
- Google-style docstrings where docstrings are added
- Every Python file must include the NVIDIA copyright header at the top
- Components must not import each other (enforced by `import-linter` — see `pyproject.toml`)

## Testing

### Unit Tests (CPU)

```bash
pytest tests/unit_tests/ -v
```

Unit tests must be CPU-compatible. Use tiny model configs with small hidden dimensions and short sequence lengths.

### Functional Tests (GPU Required)

```bash
pytest tests/functional_tests/ -v
```

These require at least one GPU. Mark GPU tests with:

```python
@pytest.mark.gpu
```

or:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
```

### CI Tests

CI test scripts live in `tests/ci_tests/`. These are executed in the CI pipeline and should not be run locally unless reproducing a CI failure.

### Tips

- Keep unit test configs tiny: small hidden dims, 1-2 layers, short sequences.
- Set `CUDA_VISIBLE_DEVICES` explicitly when running multi-GPU tests locally.
- Watch for port conflicts when running multiple `torchrun` processes. Use `--master_port` to avoid collisions.

## CLI Usage

The entry point is `automodel` (defined at `nemo_automodel._cli.app:main`).

Pattern: `automodel <command> <domain> -c <config.yaml>`

### LLM

```bash
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
automodel pretrain llm -c config.yaml
automodel kd llm -c config.yaml
automodel benchmark llm -c config.yaml
```

### VLM

```bash
automodel finetune vlm -c config.yaml
```

### Diffusion

```bash
automodel finetune diffusion -c config.yaml
```

### Retrieval

```bash
automodel finetune retrieval -c config.yaml
```

Override any config value from the CLI:

```bash
automodel finetune llm -c config.yaml --model.name_or_path meta-llama/Llama-3.2-1B
```

## Commit & PR Workflow

### Commits

All commits require DCO sign-off:

```bash
git commit -s -m "feat: add new recipe for Qwen2"
```

### Pull Requests

Follow the PR template in `.github/PULL_REQUEST_TEMPLATE.md`. Every PR should include:

1. **What**: concise description of the change.
2. **Changelog**: bullet list of user-visible changes.
3. **Pre-checks**: confirm linting, tests, and sign-off.

### Branch Naming

Use descriptive branch names prefixed with your username or a category:

```
username/feat_add_qwen2_recipe
fix/gradient_clip_nan
```

## Common Pitfalls

| Problem | Fix |
|---|---|
| Stale `.venv` after switching branches | Delete `.venv` and re-run `uv sync --locked` |
| Import errors for optional features (TE, flash-attn, MoE) | Install the matching `uv` extra (`fa`, `moe`, etc.) |
| TransformerEngine version mismatch | Pin TE version to what the container ships, or rebuild from source |
| Multi-GPU tests fail silently | Set `CUDA_VISIBLE_DEVICES` explicitly |
| `torchrun` port conflict | Pass `--master_port=<unused_port>` or set `MASTER_PORT` |
| DCO sign-off missing | Amend with `git commit --amend -s` |
