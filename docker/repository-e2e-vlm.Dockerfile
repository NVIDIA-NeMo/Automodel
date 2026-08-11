# CI overrides this with the exact primary-image digest. Keep the public,
# functional default immutable so direct builds reproduce the reviewed base.
ARG BASE_IMAGE=nvcr.io/nvidia/nemo-automodel:26.04@sha256:bd1287277a447edb1cc0b58219246740c22b6548f6d47b192a8604894e2bfc4b
FROM ${BASE_IMAGE}

WORKDIR /opt/Automodel

# Keep the published CI image as the runtime base while carrying deterministic
# test-only fixes from the dedicated E2E ref. Product sources stay unchanged.
COPY --chown=65532:65532 tests/unit_tests/speculative/test_dspark_gemma4.py tests/unit_tests/speculative/test_dspark_gemma4.py

# The native hf_transformer_vlm CI lane installs this opt-in extra at runtime.
# Populate both the venv and uv cache while the image build has package egress so
# the unchanged launcher can repeat the install in the offline GPU sandbox. Keep
# packages inherited from the official CI image, matching the additive native
# `uv pip install` step while still enforcing this checkout's lockfile.
RUN . /opt/venv/env.sh && uv sync --locked --extra vlm-media --inexact

ENV UV_OFFLINE=1
