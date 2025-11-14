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
TEST_DIR="tests/"
TEST_NAME=""
ADDITIONAL_ARGS=""

for i in "$@"; do
    case $i in
        --UNIT_TEST=?*) UNIT_TEST="${i#*=}";;
        --CPU=?*) CPU="${i#*=}";;
        --TEST_NAME=?*) TEST_NAME="${i#*=}";;
        *) ;;
    esac
    shift
done

if [[ "$CPU" == "false" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
else
    export ADDITIONAL_ARGS="--cpu --with_downloads"
fi

if [[ "$UNIT_TEST" == "true" ]]; then
    export TEST_DIR="tests/unit_tests"
else
    export TEST_DIR="tests/functional_tests/$TEST_NAME"
fi

coverage run \
    --data-file=/workspace/.coverage \
    --source=/workspace/ \
    --parallel-mode \
    -m pytest \
    $TEST_DIR \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -vs -m "not pleasefixme" --tb=short -rA \
    $ADDITIONAL_ARGS
coverage combine -q
