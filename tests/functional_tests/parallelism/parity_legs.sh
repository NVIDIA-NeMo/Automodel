# Copyright (c) 2026, NVIDIA CORPORATION.
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

# Helpers for the L2_Parallelism_* scripts: run the compared legs of a parity test at the
# same time instead of one after another. Each leg is a separate torchrun, most of its
# time is start-up, and the legs are independent once they share the same initial state.
#
# Source this file after RUN_DIR is set. Start each leg with torchrun's --standalone so
# concurrent launches pick their own rendezvous port.

LEG_PIDS=()
LEG_NAMES=()

# start_leg <name> <command...>: run a leg in the background, logging to $RUN_DIR/<name>.log.
start_leg() {
    local name=$1
    shift
    "$@" > "$RUN_DIR/$name.log" 2>&1 &
    LEG_PIDS+=($!)
    LEG_NAMES+=("$name")
}

# wait_legs: wait for every started leg, print its log, and fail if any leg failed.
wait_legs() {
    local failed=0 i
    for i in "${!LEG_PIDS[@]}"; do
        if ! wait "${LEG_PIDS[$i]}"; then
            failed=1
            echo "ERROR: leg ${LEG_NAMES[$i]} failed"
        fi
        cat "$RUN_DIR/${LEG_NAMES[$i]}.log"
    done
    LEG_PIDS=()
    LEG_NAMES=()
    return "$failed"
}
