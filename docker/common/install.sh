#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --base-image)
        BASE_IMAGE="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --base-image {pytorch|cuda}"
        exit 1
        ;;
    esac
done

# Validate base image argument
if [[ -z "${BASE_IMAGE:-}" ]]; then
    echo "Error: --base-image argument is required"
    echo "Usage: $0 --base-image {pytorch|cuda}"
    exit 1
fi

if [[ "$BASE_IMAGE" != "pytorch" && "$BASE_IMAGE" != "cuda" ]]; then
    echo "Error: --base-image must be either 'pytorch' or 'cuda'"
    echo "Usage: $0 --base-image {pytorch|cuda}"
    exit 1
fi

main() {
    if [[ -n "${PAT:-}" ]]; then
        echo -e "machine github.com\n  login token\n  password $PAT" >~/.netrc
        chmod 600 ~/.netrc
    fi

    # Install dependencies
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y curl git libopenmpi-dev libpython3.12

    # Install uv
    UV_VERSION="0.7.2"
    curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
    export UV_PROJECT_ENVIRONMENT=/opt/venv
    export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
    export UV_LINK_MODE=copy

    UV_ARGS=()
    if [[ "$BASE_IMAGE" == "pytorch" ]]; then
        UV_ARGS=(
            "--system-site-packages"
            "--no-install-package" "torch"
            "--no-install-package" "torchvision"
            "--no-install-package" "triton"
            "--no-install-package" "nvidia-cublas-cu12"
            "--no-install-package" "nvidia-cuda-cupti-cu12"
            "--no-install-package" "nvidia-cuda-nvrtc-cu12"
            "--no-install-package" "nvidia-cuda-runtime-cu12"
            "--no-install-package" "nvidia-cudnn-cu12"
            "--no-install-package" "nvidia-cufft-cu12"
            "--no-install-package" "nvidia-cufile-cu12"
            "--no-install-package" "nvidia-curand-cu12"
            "--no-install-package" "nvidia-cusolver-cu12"
            "--no-install-package" "nvidia-cusparse-cu12"
            "--no-install-package" "nvidia-cusparselt-cu12"
            "--no-install-package" "nvidia-nccl-cu12"
        )
    fi

    # Create virtual environment and install dependencies
    uv venv ${UV_PROJECT_ENVIRONMENT} $([[ "$BASE_IMAGE" == "pytorch" ]] && echo "--system-site-packages")

    # Install dependencies
    uv sync --locked --only-group build ${UV_ARGS[@]}
    uv sync \
        --link-mode copy \
        --locked \
        --extra fa \
        --all-groups ${UV_ARGS[@]}

    # Run install overrides
    bash docker/common/install_conflicting_deps.sh

    # Install the package
    uv pip install --no-deps -e .

    # Write environment variables to a file for later sourcing
    cat >/opt/venv/env.sh <<'EOF'
#!/bin/bash
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
EOF

    chmod +x /opt/venv/env.sh
}

# Call the main function
main "$@"
