#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The project pin predates the Pixtral regression. Use the real affected release,
# isolated from the base environment, so the backport executes in CI coverage.
# Remove this lane when the project pin includes HF #49373 and its tests pass.
# Usage: bash tests/ci_tests/scripts/test_pixtral_compat.sh cpu|gpu [transformers-version]
set -euo pipefail

case "${1:-}" in
    cpu) test_args=(--cpu tests/unit_tests/models/pixtral/test_pixtral_compat.py) ;;
    gpu) test_args=(tests/functional_tests/retrieval/test_mistral3_vl_flash_attention.py) ;;
    *) echo "Usage: $0 cpu|gpu [transformers-version]" >&2; exit 2 ;;
esac
transformers_version=${2:-5.18.0}
compat_dir=$(mktemp -d)
trap 'rm -rf -- "$compat_dir"' EXIT
# Exact direct prerequisites for the affected release. --no-deps prevents the
# compatibility run from replacing the container's PyTorch/CUDA stack.
uv pip install --target "$compat_dir" --no-deps \
    "transformers==$transformers_version" "huggingface-hub==1.33.0" \
    "tokenizers==0.23.2" "safetensors==0.8.0"
export PYTHONPATH="$compat_dir${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys, transformers; assert transformers.__version__ == sys.argv[1]; print("Pixtral compatibility:", transformers.__version__, transformers.__file__)' "$transformers_version"
python -m coverage run -m pytest -q --tb=short "${test_args[@]}"
