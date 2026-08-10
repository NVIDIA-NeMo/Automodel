ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/Automodel

# Keep the published CI image as the runtime base while carrying deterministic
# test-only fixes from the dedicated E2E ref. Product sources stay unchanged.
COPY --chown=65532:65532 tests/unit_tests/speculative/test_dspark_gemma4.py tests/unit_tests/speculative/test_dspark_gemma4.py

# The native hf_transformer_vlm CI lane installs this opt-in extra at runtime.
# Populate both the venv and uv cache while the image build has package egress so
# the unchanged launcher can repeat the install in the offline GPU sandbox.
RUN . /opt/venv/env.sh && uv pip install ".[vlm-media]"

ENV UV_OFFLINE=1
