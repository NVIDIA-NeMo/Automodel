#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Ensure the Tulu-3 dataset is prefiltered before an LLM convergence run.

The LLM convergence recipes (moonlight/qwen) train on ``allenai/tulu-3-sft-mixture``
with ``truncation: false``; over-length samples spike memory on the large-vocab MoEs
and OOM. The run docs and the measured baselines prefilter the dataset (drop samples
longer than ``seq_length``) first. This helper reproduces that for CI:

  * reuse an existing filtered cache for this (dataset, split, seq_length, model) if
    present -- idempotent, no repeated multi-minute filtering; otherwise
  * run ``examples/convergence/tulu3/data/prefilter.sh`` once to create it.

It prints ONLY the resolved cache path to stdout (progress goes to stderr) so the
launcher can point ``--dataset.path_or_dataset_id`` / ``--validation_dataset.path_or_dataset_id``
at it. VLM recipes (gemma4) pack with ``drop_long_samples`` and do not use this.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

AUTOMODEL_DIR = os.environ.get("AUTOMODEL_DIR", "/opt/Automodel")
PREFILTER_SH = os.path.join(AUTOMODEL_DIR, "examples/convergence/tulu3/data/prefilter.sh")
DEFAULT_CACHE_DIR = os.path.join(AUTOMODEL_DIR, "examples/convergence/tulu3/data/cached")


def main() -> int:
    p = argparse.ArgumentParser(description="Resolve (or build) the prefiltered Tulu-3 cache for a recipe.")
    p.add_argument("--config", required=True, help="Resolved finetune config YAML")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Directory holding filtered dataset caches")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset") or {}
    model = (cfg.get("model") or {}).get("pretrained_model_name_or_path")
    dataset_id = ds.get("path_or_dataset_id", "allenai/tulu-3-sft-mixture")
    seq_length = int(ds.get("seq_length", 2048))
    split = str(ds.get("split", "train"))
    if not model:
        sys.exit("[prefilter] model.pretrained_model_name_or_path missing from config")

    # Stable prefix of prefilter_dataset.get_cache_path (dirname ends with an 8-char config
    # hash we intentionally do not recompute -- glob on the stable prefix so we reuse the
    # baseline cache regardless of shuffle_seed/chat_template hashing.
    ds_name = dataset_id.replace("/", "_").replace("\\", "_")
    model_short = model.split("/")[-1]
    split_name = split.replace("[", "_").replace("]", "_").replace(":", "-")
    pattern = os.path.join(args.cache_dir, f"{ds_name}_{split_name}_seq{seq_length}_{model_short}_*")

    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if matches:
        print(f"[prefilter] reusing cache: {matches[0]}", file=sys.stderr)
        print(matches[0])
        return 0

    print(f"[prefilter] no cache for {model_short} seq{seq_length}; running prefilter.sh", file=sys.stderr)
    env = dict(
        os.environ,
        MODEL=model,
        SEQ_LENGTH=str(seq_length),
        DATASET=dataset_id,
        SPLIT=split,
        CACHE_DIR=args.cache_dir,
        # Allow the tokenizer to be fetched if not cached yet (prefilter.sh forces offline).
        HF_HUB_OFFLINE="0",
    )
    proc = subprocess.run(
        ["bash", PREFILTER_SH],
        env=env,
        cwd=AUTOMODEL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stderr.write(proc.stdout)
    if proc.returncode != 0:
        sys.exit(f"[prefilter] prefilter.sh failed (exit {proc.returncode})")

    m = re.search(r"Cached to:\s*(\S+)", proc.stdout)
    if m:
        print(m.group(1))
        return 0
    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if matches:
        print(matches[0])
        return 0
    sys.exit("[prefilter] could not determine cache path after prefilter.sh")


if __name__ == "__main__":
    raise SystemExit(main())
