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

# Install opt-in media extras (kept out of the default media-free image) per folder.
case "$TEST_NAME" in
    hf_transformer_vlm) MEDIA_EXTRA="vlm-media" ;;
    *) MEDIA_EXTRA="" ;;
esac
if [[ -n "$MEDIA_EXTRA" ]]; then
    uv pip install ".[$MEDIA_EXTRA]"
fi

coverage run \
    -m pytest \
    --durations 32 \
    --durations-min=0 \
    $TEST_DIR \
    --junitxml=junit.xml \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -vs -m "not pleasefixme" --tb=short -rA \
    $ADDITIONAL_ARGS
PYTEST_RC=$?
set -e

# Emit failed-tests reports that the CI "Failed tests" log sections cat:
#   failed_summary.txt  -- one line per failing test
#   failed_traces.txt   -- full traceback per failing test
python3 - <<'PY' 2>/dev/null || true
import glob
import xml.etree.ElementTree as ET

fails = []
for path in sorted(glob.glob("junit*.xml")):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        continue
    for tc in root.iter("testcase"):
        bad = tc.find("failure")
        if bad is None:
            bad = tc.find("error")
        if bad is None:
            continue
        loc = tc.get("file") or tc.get("classname", "").replace(".", "/")
        nodeid = loc + "::" + (tc.get("name") or "")
        msg = (bad.get("message") or "").strip().splitlines()
        msg1 = msg[0][:200] if msg else ""
        # element text is pytest's longrepr (the traceback); fall back to the summary message
        detail = (bad.text or bad.get("message") or "").rstrip()
        fails.append((nodeid, msg1, detail))

header = ("%d failing test(s):" % len(fails)) if fails else "No failing tests."

with open("failed_summary.txt", "w") as fh:
    fh.write(header + "\n")
    for nodeid, msg1, _ in fails:
        fh.write("FAILED " + nodeid + (("  -  " + msg1) if msg1 else "") + "\n")

with open("failed_traces.txt", "w") as fh:
    fh.write(header + "\n")
    for nodeid, _, detail in fails:
        fh.write("\n" + "_" * 78 + "\n")
        fh.write("FAILED " + nodeid + "\n\n")
        fh.write(detail + "\n")
PY

if [ "$PYTEST_RC" -eq 0 ]; then
    coverage combine -q
fi
exit "$PYTEST_RC"
