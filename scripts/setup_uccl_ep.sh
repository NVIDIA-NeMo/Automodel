#!/usr/bin/env bash
# setup_uccl_ep.sh — one-click UCCL-EP installation for NeMo AutoModel
#
# UCCL-EP (https://github.com/uccl-project/uccl/tree/main/ep) is a
# community fork of DeepEP that replaces NVSHMEM/IBGDA with libibverbs
# RC QPs.  This makes expert-parallel MoE training work on Azure HPC
# VMs and other environments where GPU-to-NIC PCIe P2P is unavailable.
#
# -- uv install (recommended) ------------------------------------------------
# Step 1: Install system libraries (uv cannot install apt packages)
#   sudo apt install -y libibverbs-dev librdmacm-dev \
#       libnl-3-dev libnl-route-3-dev libnuma-dev libgoogle-glog-dev
#
# Step 2: uv sync (automatically fetches from GitHub and compiles)
#   uv sync --extra moe-uccl
#
# Note: uv builds a PEP 517 wheel, so the import path is `import ep`
# (top-level). AutoModel code is compatible with both install methods.
#
# -- Manual install (recommended for Mellanox/Azure, gives control over EFA) -
# Usage:
#   bash scripts/setup_uccl_ep.sh [--uccl-src PATH] [--no-efa] [--skip-apt]
#
# Options:
#   --uccl-src PATH   Path to the UCCL source tree (default: clone from GitHub)
#   --no-efa          Build without AWS EFA support (required on Mellanox/Azure)
#   --skip-apt        Skip apt-get package installation (use if no sudo)
#
# After installation, set these environment variables before training:
#   export NCCL_IB_GID_INDEX=3              # Typical for Azure ND-series VMs
#
# Example training launch (2 nodes x 8 GPUs):
#   torchrun --nnodes=2 --nproc_per_node=8 \
#     -m nemo_automodel.train \
#     --config-path examples/llm_finetune/qwen \
#     --config-name qwen3_moe_30b_uccl_ep

set -euo pipefail

UCCL_SRC=""
NO_EFA=0
SKIP_APT=0
UCCL_REPO="https://github.com/uccl-project/uccl.git"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --uccl-src)   UCCL_SRC="$2"; shift 2 ;;
        --no-efa)     NO_EFA=1; shift ;;
        --skip-apt)   SKIP_APT=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper: run with sudo when available
# ---------------------------------------------------------------------------
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO_CMD=""
elif command -v sudo &>/dev/null && sudo -n true &>/dev/null; then
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
    SKIP_APT=1
fi

run_sudo() { ${SUDO_CMD:+$SUDO_CMD} "$@"; }

echo "================================================================"
echo " UCCL-EP setup for NeMo AutoModel"
echo " NO_EFA=${NO_EFA}  SKIP_APT=${SKIP_APT}"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
if [[ "$SKIP_APT" -eq 0 ]]; then
    echo "[1/5] Installing system packages..."
    run_sudo apt-get update -qq
    run_sudo apt-get install -y --no-install-recommends \
        build-essential \
        libibverbs-dev \
        librdmacm-dev \
        ibverbs-utils \
        rdma-core \
        libgoogle-glog-dev \
        clang-format-14 \
        git
else
    echo "[1/5] Skipping apt install (--skip-apt)"
fi

# ---------------------------------------------------------------------------
# 2. Python build dependencies
# ---------------------------------------------------------------------------
echo "[2/5] Installing Python build dependencies..."
pip install --quiet pybind11 nanobind --upgrade

# ---------------------------------------------------------------------------
# 3. Clone or locate UCCL source
# ---------------------------------------------------------------------------
echo "[3/5] Locating UCCL source..."
if [[ -n "$UCCL_SRC" ]]; then
    echo "    Using provided path: $UCCL_SRC"
    if [[ ! -d "$UCCL_SRC/ep" ]]; then
        echo "ERROR: $UCCL_SRC/ep does not exist. Is this the UCCL root?" >&2
        exit 1
    fi
else
    UCCL_SRC="$(mktemp -d)/uccl"
    echo "    Cloning $UCCL_REPO -> $UCCL_SRC ..."
    git clone --depth=1 "$UCCL_REPO" "$UCCL_SRC"
fi

EP_DIR="$UCCL_SRC/ep"

# ---------------------------------------------------------------------------
# 4. Build UCCL-EP Python extension
# ---------------------------------------------------------------------------
echo "[4/5] Building UCCL-EP (this may take a few minutes)..."
pushd "$EP_DIR" > /dev/null

# Clean stale build artifacts to avoid mixing EFA and non-EFA objects.
rm -rf build/

BUILD_EXTRA_ARGS=""
if [[ "$NO_EFA" -eq 1 ]]; then
    # Force EFA_HOME to a non-existent path so the build system cannot find
    # EFA headers.  This prevents the EFA SRD QP code-path from being
    # compiled in — it returns EOPNOTSUPP on Mellanox/Azure hardware.
    BUILD_EXTRA_ARGS="EFA_HOME=/nonexistent"
    echo "    Building without EFA (Mellanox / Azure mode)"
fi

# Install into the active Python environment.
TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" ${BUILD_EXTRA_ARGS:+env $BUILD_EXTRA_ARGS} python setup.py install

popd > /dev/null

# ---------------------------------------------------------------------------
# 5. Verify installation
# ---------------------------------------------------------------------------
echo "[5/5] Verifying installation..."
python -c "
import uccl.ep as ep
print(f'  uccl.ep imported successfully')
print(f'  is_sm90_compiled: {ep.is_sm90_compiled()}')

# Confirm no EFA symbols leaked in (Mellanox safety check)
import subprocess, sys, pathlib
so = next(iter(pathlib.Path(ep.__file__).parent.glob('*.so')), pathlib.Path(ep.__file__))
result = subprocess.run(['nm', '-D', str(so)], capture_output=True, text=True)
efa_sym = [l for l in result.stdout.splitlines() if 'efadv' in l]
if efa_sym:
    print('WARNING: EFA symbols found in binary — may fail on Mellanox hardware:')
    for l in efa_sym:
        print(' ', l)
else:
    print('  No EFA symbols detected (correct for Mellanox/Azure)')
"

echo ""
echo "================================================================"
echo " UCCL-EP installed successfully!"
echo ""
echo " Next steps:"
echo "   1. Set environment variables (Azure ND-series):"
echo "      export NCCL_IB_GID_INDEX=3       # typical for Azure ND-series"
echo ""
echo "   2. In your training config, set:"
echo "      model.backend.dispatcher: uccl_ep"
echo "      model.backend.experts: torch_mm"
echo ""
echo "   3. Or use the ready-made config:"
echo "      examples/llm_finetune/qwen/qwen3_moe_30b_uccl_ep.yaml"
echo "================================================================"
