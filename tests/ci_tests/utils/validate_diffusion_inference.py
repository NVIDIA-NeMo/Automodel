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

"""Validate diffusion model inference outputs for CI.

Checks that generated video files exist and are non-degenerate:
  - At least one output file (MP4) exists in the output directory
  - Each file is larger than a minimum size threshold (not corrupted/empty)

Exit code 0 on success, 1 on failure.

Usage::

    python tests/ci_tests/utils/validate_diffusion_inference.py \\
        --output-dir /path/to/inference_outputs
"""

import argparse
import sys
from pathlib import Path


def validate(output_dir: str, min_file_size: int = 1024) -> bool:
    """Validate that inference produced non-empty video outputs.

    Args:
        output_dir: Directory containing generated MP4 files.
        min_file_size: Minimum acceptable file size in bytes (default: 1KB).

    Returns:
        True if all checks pass, False otherwise.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"FAIL: Output directory does not exist: {output_dir}")
        return False

    outputs = list(output_path.glob("*.mp4"))
    if not outputs:
        print(f"FAIL: No *.mp4 files found in {output_dir}")
        return False

    all_ok = True
    for f in outputs:
        size = f.stat().st_size
        if size < min_file_size:
            print(f"FAIL: Output video too small ({size} bytes, minimum {min_file_size}): {f}")
            all_ok = False
        else:
            print(f"OK: {f.name} ({size:,} bytes)")

    if all_ok:
        print(f"PASS: {len(outputs)} video(s) validated successfully")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate diffusion inference outputs")
    parser.add_argument("--output-dir", required=True, help="Directory containing generated outputs")
    parser.add_argument(
        "--min-file-size", type=int, default=1024, help="Minimum file size in bytes (default: 1024)"
    )
    args = parser.parse_args()
    success = validate(args.output_dir, args.min_file_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
