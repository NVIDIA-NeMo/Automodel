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

"""FlashOptim DCP checkpoint roundtrip validator.

Compares training.jsonl logs from a full training run and a resumed run.
Verifies that:
1. num_label_tokens match (dataloader state restored correctly)
2. Training losses are within threshold (optimizer state preserved)

Called by L2_FlashOptim_DCP_Roundtrip.sh after two separate torchrun
invocations (train + resume).

Usage:
    python test_flashoptim_dcp_roundtrip.py full.jsonl resumed.jsonl --ckpt_step 6
"""

import argparse
import json
import sys

LOSS_THRESHOLD = 0.05


def read_losses(jsonl_path: str) -> dict[int, dict]:
    """Read step -> {loss, num_label_tokens} from training.jsonl."""
    entries = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            entries[e["step"]] = {
                "loss": e["loss"],
                "num_label_tokens": e["num_label_tokens"],
            }
    return entries


def main():
    parser = argparse.ArgumentParser(description="Compare training losses from full vs resumed run")
    parser.add_argument("full_jsonl", help="training.jsonl from the full run")
    parser.add_argument("resumed_jsonl", help="training.jsonl from the resumed run")
    parser.add_argument("--ckpt_step", type=int, required=True, help="Checkpoint step (compare steps >= this)")
    args = parser.parse_args()

    full = read_losses(args.full_jsonl)
    resumed = read_losses(args.resumed_jsonl)

    assert len(full) > 0, f"No entries in {args.full_jsonl}"
    assert len(resumed) > 0, f"No entries in {args.resumed_jsonl}"

    common_steps = sorted(s for s in full if s >= args.ckpt_step and s in resumed)
    assert len(common_steps) > 0, (
        f"No common steps >= {args.ckpt_step}. "
        f"Full: {sorted(full.keys())}, Resumed: {sorted(resumed.keys())}"
    )

    max_delta = 0.0
    token_mismatches = []
    print(f"\nFlashOptim DCP roundtrip (ckpt at step {args.ckpt_step}):")
    print(f"  {'step':<6} {'full_loss':<12} {'res_loss':<12} {'delta':<12} {'full_tok':<10} {'res_tok':<10}")
    for step in common_steps:
        fl, rl = full[step]["loss"], resumed[step]["loss"]
        ft, rt = full[step]["num_label_tokens"], resumed[step]["num_label_tokens"]
        delta = abs(fl - rl)
        max_delta = max(max_delta, delta)
        if ft != rt:
            token_mismatches.append(step)
        flag = " <<<" if ft != rt else ""
        print(f"  {step:<6} {fl:<12.6f} {rl:<12.6f} {delta:<12.6f} {ft:<10} {rt:<10}{flag}")

    print(f"\n  max loss delta: {max_delta:.6f}")

    assert len(token_mismatches) == 0, (
        f"Dataloader state not restored: num_label_tokens mismatch at steps {token_mismatches}"
    )
    assert max_delta < LOSS_THRESHOLD, (
        f"Training loss mismatch: max delta {max_delta:.6f} >= {LOSS_THRESHOLD}"
    )
    print("  PASSED")


if __name__ == "__main__":
    main()
