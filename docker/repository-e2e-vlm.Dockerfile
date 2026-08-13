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

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532
USER root
RUN if ! getent group "${RUNTIME_GID}" >/dev/null; then \
        groupadd --gid "${RUNTIME_GID}" nemo-runtime; \
    fi && \
    if ! getent passwd "${RUNTIME_UID}" >/dev/null; then \
        useradd --no-log-init --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
            --create-home --home-dir /home/nemo-runtime --shell /bin/bash nemo-runtime; \
    fi && \
    install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" /home/nemo-runtime /opt/uv_cache && \
    chown -R "${RUNTIME_UID}:${RUNTIME_GID}" /opt/Automodel /opt/venv /opt/uv_cache

ENV HOME=/home/nemo-runtime \
    XDG_CACHE_HOME=/tmp/regent-cache/xdg \
    UV_CACHE_DIR=/tmp/regent-cache/uv \
    UV_OFFLINE=1

USER ${RUNTIME_UID}:${RUNTIME_GID}
RUN test "$(id -u)" = "${RUNTIME_UID}" && \
    test "$(id -g)" = "${RUNTIME_GID}" && \
    test ! -e /nemo_run && \
    test -x /opt/venv/bin/python && \
    test -z "$(find /opt/venv -xdev -type l -lname '/root/*' -print -quit)" && \
    mkdir -p "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" && \
    touch /opt/Automodel/.nonroot-write-probe /opt/venv/.nonroot-write-probe && \
    rm /opt/Automodel/.nonroot-write-probe /opt/venv/.nonroot-write-probe
