#!/usr/bin/env bash
# install_media.sh — install the opt-in media stack (video/image decode).
#
# The FFmpeg-bearing media deps are kept OUT of the default container image so
# the image stays clear of FFmpeg CVEs. Run this in the container (or any
# AutoModel env) to enable VLM video reading, Mistral image tokenization, Qwen
# vision preprocessing, diffusion video export, and torchcodec.
#
# Usage:
#   bash scripts/install_media.sh
set -euo pipefail

TORCHCODEC_REF="${TORCHCODEC_REF:-v0.8.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# System FFmpeg: torchcodec links libav* at runtime; dev libs for its source build.
apt-get update
apt-get install -y --no-install-recommends ffmpeg libavdevice-dev pkg-config
apt-get clean
rm -rf /var/lib/apt/lists/*

# Add the media extra on top of the image's existing all-extra environment.
uv sync --extra all --extra media --all-groups --locked

# torchcodec from source to match the container PyTorch ABI (the PyPI wheel mismatches).
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    pip install --no-build-isolation --force-reinstall --no-deps \
        "git+https://github.com/pytorch/torchcodec.git@${TORCHCODEC_REF}"
