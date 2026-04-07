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

"""Download modal-labs/dissolve dataset and extract videos + captions for CI.

Downloads the HuggingFace dataset and writes video files with JSON sidecar
captions in the format expected by ``tools/diffusion/preprocessing_multiprocess.py``
with ``--caption_format sidecar``.

Output structure::

    output_dir/
        dissolve_000.mp4
        dissolve_000.json   # {"caption": "..."}
        dissolve_001.mp4
        dissolve_001.json
        ...

Usage::

    python tests/ci_tests/scripts/download_diffusion_test_data.py \\
        --output-dir /tmp/dissolve_raw
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Download modal-labs/dissolve dataset for CI")
    parser.add_argument("--output-dir", required=True, help="Directory to write videos and captions")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import Video, load_dataset

    ds = load_dataset("modal-labs/dissolve", split="train")
    # Disable video decoding so we get raw bytes/path dicts instead of VideoDecoder objects
    ds = ds.cast_column("video", Video(decode=False))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    for i, sample in enumerate(ds):
        video_path = os.path.join(args.output_dir, f"dissolve_{i:03d}.mp4")
        caption_path = os.path.join(args.output_dir, f"dissolve_{i:03d}.json")

        # Save video file — HF datasets returns video as a dict with path or bytes
        video_data = sample["video"]
        if isinstance(video_data, dict) and "path" in video_data and video_data["path"]:
            shutil.copy(video_data["path"], video_path)
        elif isinstance(video_data, dict) and "bytes" in video_data and video_data["bytes"]:
            with open(video_path, "wb") as f:
                f.write(video_data["bytes"])
        elif isinstance(video_data, bytes):
            with open(video_path, "wb") as f:
                f.write(video_data)
        else:
            print(f"WARNING: Could not extract video for sample {i}, type={type(video_data)}")
            continue

        # Save caption as JSON sidecar (expected by JSONSidecarCaptionLoader)
        with open(caption_path, "w") as f:
            json.dump({"caption": sample["prompt"]}, f)

    print(f"Extracted {len(ds)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
