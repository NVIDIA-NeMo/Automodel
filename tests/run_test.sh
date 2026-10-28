# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

UNIT_TEST=false
CPU=false
TEST_NAME=""
ADDITIONAL_ARGS=""
SHARD_ID=""
NUM_SHARDS=""

for i in "$@"; do
    case $i in
        --UNIT_TEST=?*) UNIT_TEST="${i#*=}";;
        --CPU=?*) CPU="${i#*=}";;
        --TEST_NAME=?*) TEST_NAME="${i#*=}";;
        --SHARD_ID=?*) SHARD_ID="${i#*=}";;
        --NUM_SHARDS=?*) NUM_SHARDS="${i#*=}";;
        *) ;;
    esac
    shift
done

# Optionally split the unit suite across several parallel CI runners. Whole test
# files are assigned to shards, balanced by tests/unit_tests/gpu_shard_durations.json.
# Only takes effect when both a shard index and a shard count (>1) are provided.
SHARD_ARGS=""
if [[ -n "$NUM_SHARDS" && "$NUM_SHARDS" -gt 1 && -n "$SHARD_ID" ]]; then
    SHARD_ARGS="--file-shard-id=$SHARD_ID --file-num-shards=$NUM_SHARDS"
fi

if [[ "$CPU" == "false" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
    # On GCP hosts the NGC shell init (/etc/shinit_v2) enables Google's NCCL profiler plugin.
    # Its gRPC telemetry client times out during interpreter shutdown, so a pytest process
    # that created an NCCL communicator segfaults on exit after every test passed.
    export NCCL_PROFILER_PLUGIN=none
else
    export ADDITIONAL_ARGS="--cpu"
fi

if [[ "$UNIT_TEST" == "true" ]]; then
    TEST_DIRS=("tests/unit_tests")
else
    IFS=',' read -r -a TEST_FOLDERS <<< "$TEST_NAME"
    TEST_DIRS=()
    for TEST_FOLDER in "${TEST_FOLDERS[@]}"; do
        if [[ -z "$TEST_FOLDER" ]]; then
            echo "Functional test folder names must not be empty" >&2
            exit 2
        fi
        TEST_DIRS+=("tests/functional_tests/$TEST_FOLDER")
    done
fi

# Test dependencies, including vlm-media for VLM and parallelism suites, are
# installed when building the CI image.

# Default pytest capture reports module progress and includes stdout, stderr, and logs only for failed tests.
TEST_COMMAND=(coverage run -m pytest)
MARKER_EXPRESSION="not pleasefixme"
if [[ "$UNIT_TEST" == "true" ]]; then
    # CPU/Gloo tests also run in the GPU shards. Bound their host thread pools
    # consistently so spawned ranks do not oversubscribe the runner.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    # Measure line coverage with Python 3.12's sys.monitoring. The default settrace
    # tracer made the unit suite about 40% slower; sys.monitoring adds about 1%.
    export COVERAGE_CORE=sysmon
fi
if [[ "$UNIT_TEST" == "true" && "$CPU" == "false" ]]; then
    # Tests marked cpu_only never use CUDA, and L0_Unit_Tests_CPU already runs them.
    MARKER_EXPRESSION="not pleasefixme and not cpu_only"
fi
if [[ "$UNIT_TEST" == "true" && "$CPU" == "true" ]]; then
    # Keep one CI job, with bounded concurrency for tests that spawn CPU ranks.
    # File scheduling keeps module fixtures and fixed-port tests on one worker.
    # pytest-cov collects worker coverage; coverage run alone only traces the controller.
    TEST_COMMAND=(python -m pytest -n 2 --dist loadfile --cov --cov-config=pyproject.toml --cov-report=)
fi

"${TEST_COMMAND[@]}" \
    --durations 32 \
    --durations-min=0 \
    "${TEST_DIRS[@]}" \
    -m "$MARKER_EXPRESSION" --tb=short -rfE \
    $SHARD_ARGS \
    $ADDITIONAL_ARGS
if [[ "$UNIT_TEST" != "true" || "$CPU" != "true" ]]; then
    coverage combine -q
fi
